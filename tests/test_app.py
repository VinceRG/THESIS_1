import os
import re
import tempfile
import unittest
from io import BytesIO

# The app binds its SQLAlchemy engine to SQLALCHEMY_DATABASE_URI the moment
# app.py is imported (create_app() -> init_db() run at module load time).
# Point it at an isolated on-disk sqlite file *before* importing app, so the
# test suite's db.drop_all()/db.create_all() calls never touch the real
# instance/clinic.db.
_TEST_DB_PATH = os.path.join(tempfile.gettempdir(), 'smart_clinic_test.db')
os.environ['DATABASE_URL'] = 'sqlite:///' + _TEST_DB_PATH.replace('\\', '/')

import pandas as pd

from app import (
    Appointment,
    Branch,
    ConsultationRecord,
    Patient,
    StaffMember,
    User,
    app,
    build_forecasting_training_frame,
    build_training_frame,
    db,
    init_db,
    _rate_limit_attempts,
)


class SmartClinicAppTests(unittest.TestCase):
    def setUp(self):
        self.app = app
        self.app.config.update(
            TESTING=True,
            SECRET_KEY='test-secret',
            # Likewise isolate uploaded files / generated reports from the
            # real uploads/ folder used by the running app.
            UPLOAD_FOLDER=tempfile.mkdtemp(prefix='clinic_test_uploads_'),
        )
        self.client = self.app.test_client()
        self.csrf_token = None
        _rate_limit_attempts.clear()
        with self.app.app_context():
            db.drop_all()
            db.create_all()
            init_db()

    @staticmethod
    def _extract_csrf_token(html):
        match = re.search(rb'name="_csrf_token"\s+value="([^"]*)"', html)
        return match.group(1).decode() if match else ''

    def _login(self, username='admin', password='admin123'):
        """Log in through the real /login flow, including the CSRF token
        every POST route now requires (see app.py's require_login/
        validate_csrf_token), and complete the forced password change that
        seeded accounts (must_change_password=True) are routed through on
        first login."""
        login_page = self.client.get('/login')
        self.csrf_token = self._extract_csrf_token(login_page.data)
        response = self.client.post(
            '/login',
            data={'username': username, 'password': password, '_csrf_token': self.csrf_token},
            follow_redirects=True,
        )
        if b'name="new_password"' in response.data:
            response = self.client.post(
                '/change-password',
                data={
                    '_csrf_token': self.csrf_token,
                    'new_password': 'test-password-123',
                    'confirm_password': 'test-password-123',
                },
                follow_redirects=True,
            )
        return response

    def test_login_page_renders(self):
        response = self.client.get('/login')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Login', response.data)

    def test_dashboard_requires_login(self):
        response = self.client.get('/dashboard', follow_redirects=True)
        self.assertIn(b'Login', response.data)

    def test_login_locks_out_after_repeated_failures(self):
        login_page = self.client.get('/login')
        token = self._extract_csrf_token(login_page.data)
        bad_login = {'username': 'admin', 'password': 'wrong-password', '_csrf_token': token}

        for _ in range(5):
            response = self.client.post('/login', data=bad_login)
            self.assertIn(b'Invalid credentials', response.data)

        locked_out = self.client.post('/login', data=bad_login)
        self.assertIn(b'Too many login attempts', locked_out.data)

        # Even the correct password is now blocked until the window expires.
        good_login = {'username': 'admin', 'password': 'admin123', '_csrf_token': token}
        still_blocked = self.client.post('/login', data=good_login)
        self.assertIn(b'Too many login attempts', still_blocked.data)

    def test_seeded_admin_is_forced_to_change_password(self):
        login_page = self.client.get('/login')
        token = self._extract_csrf_token(login_page.data)
        login_response = self.client.post(
            '/login',
            data={'username': 'admin', 'password': 'admin123', '_csrf_token': token},
            follow_redirects=True,
        )
        self.assertIn(b'name="new_password"', login_response.data)

        # The dashboard must stay locked until the password is actually changed.
        blocked = self.client.get('/dashboard', follow_redirects=True)
        self.assertIn(b'name="new_password"', blocked.data)

        changed = self.client.post(
            '/change-password',
            data={'_csrf_token': token, 'new_password': 'new-secure-pass1', 'confirm_password': 'new-secure-pass1'},
            follow_redirects=True,
        )
        self.assertEqual(changed.status_code, 200)
        self.assertNotIn(b'name="new_password"', changed.data)

        # Logging in again with the new password should reach the dashboard
        # directly, with no further forced change.
        self.client.get('/logout')
        relogin_page = self.client.get('/login')
        relogin_token = self._extract_csrf_token(relogin_page.data)
        relogin = self.client.post(
            '/login',
            data={'username': 'admin', 'password': 'new-secure-pass1', '_csrf_token': relogin_token},
            follow_redirects=True,
        )
        self.assertNotIn(b'name="new_password"', relogin.data)

    def test_build_forecasting_training_frame_with_no_records(self):
        empty_df = pd.DataFrame(columns=[
            'consultation_date', 'age_group', 'gender', 'diagnosis',
            'department', 'physician', 'consultation_type',
        ])
        result = build_forecasting_training_frame(empty_df)
        self.assertTrue(result.empty)

    def test_dashboard_shows_prediction_month_labels(self):
        self._login()
        with self.app.app_context():
            db.session.add(ConsultationRecord(
                consultation_date='2024-01-01',
                age_group='Adult',
                gender='Male',
                diagnosis='Hypertension',
                department='General Medicine',
                physician='Dr. Ada',
                consultation_type='New',
            ))
            db.session.commit()
        response = self.client.get('/dashboard')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Predicted Month', response.data)

    @staticmethod
    def _sample_consultation_csv():
        # The live training pipeline (train_and_evaluate_model) requires at
        # least 8 distinct months of data, and REQUIRE_COMPLETE_TRAINING_YEARS
        # (app.py:92) discards any calendar year that isn't fully populated
        # with all 12 months -- so the fixture has to span one full year,
        # not the 3 months the original fixture used.
        diagnoses = ['Hypertension', 'Diabetes', 'Upper Respiratory Infection']
        rows = ['consultation_date,age_group,gender,diagnosis,department,physician,consultation_type']
        for month in range(1, 13):
            for index, diagnosis in enumerate(diagnoses):
                gender = 'Male' if index % 2 == 0 else 'Female'
                consultation_type = 'Follow-up' if index % 2 == 0 else 'New'
                rows.append(
                    f'2024-{month:02d}-{index + 1:02d},Adult,{gender},{diagnosis},'
                    f'General Medicine,Dr. Ada,{consultation_type}'
                )
        return ('\n'.join(rows) + '\n').encode('utf-8')

    def test_upload_trains_model_and_writes_metrics_report(self):
        self._login()
        data = {'file': (BytesIO(self._sample_consultation_csv()), 'sample.csv'), '_csrf_token': self.csrf_token}
        response = self.client.post('/upload', data=data, content_type='multipart/form-data', follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        report_path = os.path.join(self.app.config['UPLOAD_FOLDER'], 'training_report.txt')
        self.assertTrue(os.path.exists(report_path))

    def test_predict_page_uses_metrics_from_report(self):
        self._login()
        data = {'file': (BytesIO(self._sample_consultation_csv()), 'sample.csv'), '_csrf_token': self.csrf_token}
        self.client.post('/upload', data=data, content_type='multipart/form-data', follow_redirects=True)
        # /upload retrains the model but does not force-refresh the cached
        # dashboard/predict summary (see audit finding: "normal page loads
        # don't retrain"), so metrics only become fresh after an explicit
        # retrain -- exactly what the "Retrain Model" button on this page
        # does. Mirroring that here keeps the test honest about current
        # behavior instead of asserting on the un-refreshed zero values.
        self.client.post('/retrain', data={'_csrf_token': self.csrf_token}, follow_redirects=True)
        response = self.client.get('/predict')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Validation R2', response.data)
        self.assertIn(b'MAE', response.data)
        self.assertIn(b'MSE', response.data)
        self.assertIn(b'RMSE', response.data)
        self.assertNotIn(b'0.0', response.data)

    @staticmethod
    def _consultation_csv_with_physicians(physician_by_diagnosis):
        """Like _sample_consultation_csv but lets each diagnosis carry its own
        physician, so evidence-based diagnosis->role attribution (historical
        physician match vs. keyword fallback) can be exercised deterministically."""
        diagnoses = list(physician_by_diagnosis.keys())
        rows = ['consultation_date,age_group,gender,diagnosis,department,physician,consultation_type']
        for month in range(1, 13):
            for index, diagnosis in enumerate(diagnoses):
                gender = 'Male' if index % 2 == 0 else 'Female'
                consultation_type = 'Follow-up' if index % 2 == 0 else 'New'
                physician = physician_by_diagnosis[diagnosis]
                rows.append(
                    f'2024-{month:02d}-{index + 1:02d},Adult,{gender},{diagnosis},'
                    f'General Medicine,{physician},{consultation_type}'
                )
        return ('\n'.join(rows) + '\n').encode('utf-8')

    def test_resources_page_uses_historical_mapping_and_falls_back_when_unmatched(self):
        """Diagnosis->role attribution on the Resources page should prefer a
        physician-derived historical mapping (ConsultationRecord.physician
        matched against the live StaffMember roster) and only fall back to the
        static keyword rule when no current staff member matches the diagnosis's
        historical physician on record."""
        self._login()
        with self.app.app_context():
            branch = Branch.query.first()
            db.session.add(StaffMember(
                branch_id=branch.id, name='Dr. Ada', role='Internal Medicine Physicians', is_active=True,
            ))
            db.session.commit()

        csv_bytes = self._consultation_csv_with_physicians({
            'Hypertension': 'Dr. Ada',                    # matches a real StaffMember -> historical
            'Diabetes': 'Dr. Ada',                         # matches a real StaffMember -> historical
            'Upper Respiratory Infection': 'Dr. Unlisted',  # no matching StaffMember -> fallback
        })
        data = {'file': (BytesIO(csv_bytes), 'sample.csv'), '_csrf_token': self.csrf_token}
        self.client.post('/upload', data=data, content_type='multipart/form-data', follow_redirects=True)
        self.client.post('/retrain', data={'_csrf_token': self.csrf_token}, follow_redirects=True)

        response = self.client.get('/resources')
        self.assertEqual(response.status_code, 200)
        body = response.data.decode('utf-8')
        match = re.search(
            r'<strong>(\d+)</strong>\s*diagnos\w+ used historical staffing data.*?'
            r'<strong>(\d+)</strong>\s*used the fallback keyword rule',
            body, re.S,
        )
        self.assertIsNotNone(match, 'Expected the diagnosis-to-role mapping transparency line on the Resources page')
        historical_count, fallback_count = int(match.group(1)), int(match.group(2))
        self.assertGreaterEqual(historical_count, 1, 'Hypertension/Diabetes should be mapped from historical physician data')
        self.assertGreaterEqual(fallback_count, 1, 'Upper Respiratory Infection has no matching staff member and should fall back')

    def test_retrain_runs_real_hyperparameter_search(self):
        """/retrain now trains with fast=False, so RandomizedSearchCV actually
        runs in production instead of always returning the fixed fast=True
        parameter dict -- assert the reported params vary with n_iter search
        space options that the fixed dict never uses (e.g. max_features=0.5
        or min_samples_split=2 alongside a non-default n_estimators)."""
        self._login()
        data = {'file': (BytesIO(self._sample_consultation_csv()), 'sample.csv'), '_csrf_token': self.csrf_token}
        self.client.post('/upload', data=data, content_type='multipart/form-data', follow_redirects=True)
        response = self.client.post('/retrain', data={'_csrf_token': self.csrf_token}, follow_redirects=True)
        self.assertEqual(response.status_code, 200)

        report_path = os.path.join(self.app.config['UPLOAD_FOLDER'], 'training_report.txt')
        with open(report_path, encoding='utf-8') as handle:
            report_text = handle.read()
        self.assertIn('Best Random Forest Parameters', report_text)
        # The old fast=True path always reported this exact fixed combination;
        # fast=False explores {80,120,150} x {4,6,8} x {2,5} x {1,2} x
        # {sqrt,log2,0.5} via RandomizedSearchCV, so it should not always land
        # back on the fixed-path defaults.
        fixed_fast_params = (
            'n_estimators      : 80\n'
            'max_depth         : 6\n'
            'min_samples_split : 5\n'
            'min_samples_leaf  : 2\n'
            'max_features      : sqrt'
        )
        self.assertNotIn(fixed_fast_params, report_text)

    def test_build_training_frame_cleans_and_encodes(self):
        # build_training_frame (app.py:337) is unused by any live route today
        # (the active pipeline is build_forecasting_training_frame), but it's
        # still imported/tested here, so this asserts what it actually does:
        # drop unparseable rows and label-encode the categorical columns.
        # The previous version of this test asserted a row-count "augmentation"
        # behavior the function has never implemented.
        sparse_df = pd.DataFrame([
            {'consultation_date': '2024-01-01', 'age_group': 'Adult', 'gender': 'Male', 'diagnosis': 'Hypertension', 'department': 'General Medicine', 'physician': 'Dr. Ada', 'consultation_type': 'New'},
            {'consultation_date': '2024-02-01', 'age_group': 'Adult', 'gender': 'Female', 'diagnosis': 'Diabetes', 'department': 'General Medicine', 'physician': 'Dr. Ada', 'consultation_type': 'New'},
            {'consultation_date': 'not-a-date', 'age_group': 'Adult', 'gender': 'Male', 'diagnosis': 'Hypertension', 'department': 'General Medicine', 'physician': 'Dr. Ada', 'consultation_type': 'New'},
        ])
        cleaned_df = build_training_frame(sparse_df)
        self.assertEqual(len(cleaned_df), 2)
        self.assertTrue(pd.api.types.is_numeric_dtype(cleaned_df['diagnosis']))

    # ------------------------------------------------------------------
    # Regression tests for the two most severe findings in the codebase
    # audit. Both are expected to FAIL on the current code and start
    # passing once the corresponding Phase 1 fix lands.
    # ------------------------------------------------------------------

    def test_clear_records_blocked_without_login(self):
        """/records/clear is missing from the before_request protected_endpoints
        allow-list (app.py ~2968-2981), so an anonymous request that only
        carries a valid CSRF token can currently wipe consultation records."""
        with self.app.app_context():
            branch = Branch.query.first()
            db.session.add(ConsultationRecord(
                branch_id=branch.id,
                consultation_date='2024-01-01',
                age_group='Adult',
                gender='Male',
                diagnosis='Hypertension',
                department='General Medicine',
                physician='Dr. Ada',
                consultation_type='New',
            ))
            db.session.commit()

        # Obtain a CSRF token the way an anonymous visitor legitimately can,
        # without ever logging in.
        login_page = self.client.get('/login')
        token = self._extract_csrf_token(login_page.data)
        self.client.post('/records/clear', data={'_csrf_token': token})

        with self.app.app_context():
            remaining = ConsultationRecord.query.count()
        self.assertEqual(
            remaining, 1,
            'An unauthenticated request should not be able to clear consultation records.'
        )

    def test_completing_appointment_twice_does_not_duplicate_records(self):
        """/appointments/<id>/complete never checks converted_to_records
        before processing a POST (app.py ~4175-4256), so resubmitting the
        form against an already-completed appointment currently creates a
        second full set of consultation records."""
        with self.app.app_context():
            branch = Branch.query.first()
            patient = Patient(
                branch_id=branch.id,
                patient_number='P-TEST-0001',
                full_name='Test Patient',
                birthdate='1990-01-01',
                age=34,
                age_group='Adult',
                gender='Male',
            )
            db.session.add(patient)
            db.session.commit()
            appointment = Appointment(
                branch_id=branch.id,
                patient_id=patient.id,
                appointment_date='2024-01-01',
                status='Confirmed',
            )
            db.session.add(appointment)
            db.session.commit()
            appointment_id = appointment.id
            patient_id = patient.id

        self._login()
        complete_data = {
            '_csrf_token': self.csrf_token,
            'completed_date': '2024-01-02',
            'assigned_staff_0': 'Dr. Ada',
            'final_diagnosis_0': 'Check-up',
            'service_notes_0': '',
        }
        complete_url = f'/appointments/{appointment_id}/complete'
        self.client.post(complete_url, data=complete_data, follow_redirects=True)
        self.client.post(complete_url, data=complete_data, follow_redirects=True)

        with self.app.app_context():
            record_count = ConsultationRecord.query.filter_by(patient_id=patient_id).count()
        self.assertEqual(
            record_count, 1,
            'Completing an already-completed appointment should not create duplicate consultation records.'
        )


if __name__ == '__main__':
    unittest.main()
