import json
import os
import re
import tempfile
import unittest
from datetime import datetime
from io import BytesIO
from unittest.mock import patch

# The app binds its SQLAlchemy engine to SQLALCHEMY_DATABASE_URI the moment
# app.py is imported (create_app() -> init_db() run at module load time).
# Point it at an isolated on-disk sqlite file *before* importing app, so the
# test suite's db.drop_all()/db.create_all() calls never touch the real
# instance/clinic.db.
_TEST_DB_PATH = os.path.join(tempfile.gettempdir(), 'smart_clinic_test.db')
os.environ['DATABASE_URL'] = 'sqlite:///' + _TEST_DB_PATH.replace('\\', '/')

import pandas as pd
from flask import session
from werkzeug.security import generate_password_hash

from chatbot import GeminiError

from app import (
    Appointment,
    AuditLog,
    Branch,
    ChatbotInteraction,
    ConsultationRecord,
    Patient,
    StaffMember,
    User,
    app,
    apply_rare_diagnosis_grouping,
    build_forecasting_training_frame,
    build_training_frame,
    db,
    generate_forecast_for_month,
    init_db,
    train_and_evaluate_model,
    RARE_DIAGNOSIS_BUCKET,
    _predict_consultation_count_interval,
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

    @staticmethod
    def _recent_consultation_csv():
        """Like _sample_consultation_csv, but the 10-month window ends in the
        current real-world month instead of a fixed past year (2024).
        Needed for tests that check the *live* forecast rather than the
        naive fallback: build_dashboard_context always forecasts "next
        calendar month" relative to wall-clock time, so lag/rolling
        features only carry real signal when training data extends close
        to "now" -- a fixed-past-year fixture drifts further from "now"
        every year the test suite is run, silently degrading into the
        fallback path (verified empirically: with 2024-fixed dates, the RF
        predicts a total of exactly 0 once "now" is far enough past 2024,
        which build_dashboard_context then treats as "no usable forecast"
        and discards rf_metrics entirely). A rolling window that doesn't
        align to a Jan-Dec calendar year also sidesteps
        REQUIRE_COMPLETE_TRAINING_YEARS filtering, which would otherwise
        keep only a single complete calendar year and reintroduce the same
        gap."""
        diagnoses = [
            'Hypertension', 'Diabetes', 'Upper Respiratory Infection', 'Asthma Monitoring',
            'Gastritis', 'Dermatitis and Allergy', 'Fever', 'Routine Medical Consultation',
            'Headache', 'Preventive Health Check-up',
        ]
        now = datetime.now()
        months = []
        year, month = now.year, now.month
        for _ in range(10):
            months.append((year, month))
            month -= 1
            if month == 0:
                month = 12
                year -= 1
        months.reverse()
        rows = ['consultation_date,age_group,gender,diagnosis,department,physician,consultation_type']
        for year, month in months:
            for index, diagnosis in enumerate(diagnoses):
                record_count = 3 if index < 5 else 1
                for k in range(record_count):
                    gender = 'Male' if index % 2 == 0 else 'Female'
                    consultation_type = 'Follow-up' if index % 2 == 0 else 'New'
                    day = (index + k) % 28 + 1
                    rows.append(
                        f'{year}-{month:02d}-{day:02d},Adult,{gender},{diagnosis},'
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
        # /upload now force-refreshes the cached dashboard/predict summary as
        # part of the forecast-freshness fix, so metrics are already fresh at
        # this point -- the explicit /retrain call below is kept only to
        # exercise the "Retrain Model" button's own code path, not because
        # it's required for fresh metrics to appear.
        self.client.post('/retrain', data={'_csrf_token': self.csrf_token}, follow_redirects=True)
        response = self.client.get('/predict')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Validation R2', response.data)
        self.assertIn(b'MAE', response.data)
        self.assertIn(b'MSE', response.data)
        self.assertIn(b'RMSE', response.data)
        # /predict falls back to {'r2_score': 0, 'mae': 0, ...} when no forecast
        # exists, which renders as a bare "0" in the metric cards. Match that
        # placeholder exactly -- scanning for "0.0" also matches a genuinely
        # small error such as an MAE of 0.0444.
        self.assertNotIn(b'<div class="value">0</div>', response.data)

    def test_predict_page_shows_demographics_comparison_section(self):
        """RQ3: the with-vs-without-demographics model comparison already
        computed by evaluate_model_without_demographics should be visible on
        the live Predictions page, not only in the offline training report."""
        self._login()
        data = {'file': (BytesIO(self._recent_consultation_csv()), 'sample.csv'), '_csrf_token': self.csrf_token}
        self.client.post('/upload', data=data, content_type='multipart/form-data', follow_redirects=True)
        response = self.client.get('/predict')
        self.assertEqual(response.status_code, 200)
        body = response.data.decode('utf-8')
        self.assertIn('With vs. Without Demographics', body)
        self.assertIn('Model B (Without Demographics)', body)
        self.assertNotIn('Demographic comparison not available yet', body)

    def test_predict_page_handles_missing_demographics_comparison_gracefully(self):
        """Before any data has ever been uploaded, rf_metrics is the empty
        fallback stub with no model_b_without_demographics key -- the new
        sections must degrade gracefully instead of raising a template error."""
        self._login()
        response = self.client.get('/predict')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Demographic comparison not available yet', response.data)

    @staticmethod
    def _complete_years_plus_recent_gap_csv():
        """Two full complete calendar years (safely in the past) for real
        training signal, plus several more recent months of the current,
        still-in-progress year extending up to today -- mirrors the real
        production data's actual shape (which is exactly why the
        training-year-cutoff bug went unnoticed until tested against real
        data instead of a single-fixed-year fixture)."""
        diagnoses = ['Hypertension', 'Diabetes', 'Upper Respiratory Infection']
        rows = ['consultation_date,age_group,gender,diagnosis,department,physician,consultation_type']
        now = datetime.now()

        for year in (now.year - 3, now.year - 2):
            for month in range(1, 13):
                for index, diagnosis in enumerate(diagnoses):
                    gender = 'Male' if index % 2 == 0 else 'Female'
                    consultation_type = 'Follow-up' if index % 2 == 0 else 'New'
                    rows.append(
                        f'{year}-{month:02d}-{index + 1:02d},Adult,{gender},{diagnosis},'
                        f'General Medicine,Dr. Ada,{consultation_type}'
                    )

        # Recent months of the current (incomplete) year, with a deliberately
        # higher, distinctive volume so a real lag_1 signal is unambiguous
        # once it's actually visible to the forecast.
        for offset in range(6):
            target_month = now.month - offset
            target_year = now.year
            if target_month <= 0:
                target_month += 12
                target_year -= 1
            for k in range(10):
                day = (k % 27) + 1
                rows.append(
                    f'{target_year}-{target_month:02d}-{day:02d},Adult,Male,Hypertension,'
                    f'General Medicine,Dr. Ada,Follow-up'
                )
        return ('\n'.join(rows) + '\n').encode('utf-8')

    def test_forecast_uses_recent_lag_signal_beyond_training_year_cutoff(self):
        """RQ1 regression: forecasting 'next month' must be able to see lag
        context from the current, still-in-progress year, not just fully
        complete calendar years -- otherwise near-term forecasts
        systematically collapse toward zero once 'now' drifts far enough
        past the last complete year (verified against real production data
        this session: Upper Respiratory Infection ran 40-58 cases/month as
        of July 2026, forecast showed 4)."""
        df = pd.read_csv(BytesIO(self._complete_years_plus_recent_gap_csv()))

        now = datetime.now()
        target_month = now.month + 1
        target_year = now.year
        if target_month > 12:
            target_month = 1
            target_year += 1
        target_period = pd.Period(year=target_year, month=target_month, freq='M')

        # Contrast: the OLD behavior (year-filtered lag context) cannot see
        # the month immediately before the target at all.
        filtered = build_forecasting_training_frame(df, apply_year_filter=True)
        self.assertNotIn(target_period - 1, set(filtered['period']))

        # The fix: an unfiltered lag-context frame CAN see it.
        unfiltered = build_forecasting_training_frame(df, apply_year_filter=False)
        self.assertIn(target_period - 1, set(unfiltered['period']))

        model, metrics, feature_cols, label_mapping = train_and_evaluate_model(df, fast=True)
        forecast = generate_forecast_for_month(model, feature_cols, label_mapping, df, target_month, target_year)
        forecast_by_diagnosis = dict(forecast)
        self.assertGreater(
            forecast_by_diagnosis.get('Hypertension', 0), 0,
            'Forecast should use recent lag signal instead of collapsing to zero across the training-year gap',
        )

    def test_apply_rare_diagnosis_grouping_respects_keep_diagnoses(self):
        # Needs >= MIN_MODELED_DIAGNOSES_FOR_RF (10) distinct diagnoses, or
        # the "keep at least 10" fallback rescues every diagnosis regardless
        # of count and nothing gets grouped to demonstrate against.
        diagnoses = {'A': 60}
        diagnoses.update({f'Rare{i}': 2 for i in range(15)})
        df = pd.DataFrame({'diagnosis': [d for d, count in diagnoses.items() for _ in range(count)]})

        grouped_default, _ = apply_rare_diagnosis_grouping(df.copy())
        self.assertIn('A', set(grouped_default['diagnosis']))
        self.assertIn(RARE_DIAGNOSIS_BUCKET, set(grouped_default['diagnosis']))

        # Pinning keep_diagnoses to exclude 'A' must group it regardless of
        # its own count in this call's data -- this is what keeps a wider
        # (unfiltered) lag-context frame from ever disagreeing with the
        # trained model's own diagnosis vocabulary.
        grouped_pinned, _ = apply_rare_diagnosis_grouping(df.copy(), keep_diagnoses={'Rare0'})
        self.assertNotIn('A', set(grouped_pinned['diagnosis']))
        self.assertIn(RARE_DIAGNOSIS_BUCKET, set(grouped_pinned['diagnosis']))

    def test_time_index_anchor_consistent_between_filtered_and_unfiltered_frames(self):
        df = pd.read_csv(BytesIO(self._complete_years_plus_recent_gap_csv()))
        filtered = build_forecasting_training_frame(df, apply_year_filter=True)
        unfiltered = build_forecasting_training_frame(df, apply_year_filter=False)
        self.assertEqual(
            filtered.attrs['time_index_anchor_year'],
            unfiltered.attrs['time_index_anchor_year'],
            'Both frames must anchor time_index to the same year so a trained model sees a '
            'consistent numeric scale between training and prediction time',
        )

    def test_prediction_interval_bounds_are_sensible(self):
        """Unit-level check of the new tree-spread interval helper: bounds
        must never be negative and low must never exceed high."""
        df = pd.read_csv(BytesIO(self._recent_consultation_csv()))
        model, metrics, feature_cols, label_mapping = train_and_evaluate_model(df, fast=True)
        training_df = build_forecasting_training_frame(df, apply_year_filter=False)
        low, high = _predict_consultation_count_interval(model, training_df[feature_cols])
        self.assertTrue((low >= 0).all())
        self.assertTrue((high >= low).all())

    @staticmethod
    def _recent_consultation_csv_with_variance():
        """Like _recent_consultation_csv, but with genuine month-to-month
        variability in record counts (instead of an identical fixed pattern
        every month) -- a perfectly deterministic fixture causes every tree
        in the forest to agree exactly, which correctly collapses the
        prediction interval to zero width and can't demonstrate the range
        display. Real production data naturally has this kind of variance."""
        import random
        rng = random.Random(42)
        diagnoses = [
            'Hypertension', 'Diabetes', 'Upper Respiratory Infection', 'Asthma Monitoring',
            'Gastritis', 'Dermatitis and Allergy', 'Fever', 'Routine Medical Consultation',
            'Headache', 'Preventive Health Check-up',
        ]
        now = datetime.now()
        months = []
        year, month = now.year, now.month
        for _ in range(10):
            months.append((year, month))
            month -= 1
            if month == 0:
                month = 12
                year -= 1
        months.reverse()
        rows = ['consultation_date,age_group,gender,diagnosis,department,physician,consultation_type']
        for year, month in months:
            for index, diagnosis in enumerate(diagnoses):
                base_count = 6 if index < 5 else 2
                record_count = max(1, base_count + rng.randint(-2, 3))
                for k in range(record_count):
                    gender = 'Male' if index % 2 == 0 else 'Female'
                    consultation_type = 'Follow-up' if index % 2 == 0 else 'New'
                    day = (index + k) % 27 + 1
                    rows.append(
                        f'{year}-{month:02d}-{day:02d},Adult,{gender},{diagnosis},'
                        f'General Medicine,Dr. Ada,{consultation_type}'
                    )
        return ('\n'.join(rows) + '\n').encode('utf-8')

    def test_predict_page_shows_forecast_range(self):
        """RQ2-adjacent: a bare point forecast implies a precision the model
        doesn't have. After a real retrain, the Predictions page should show
        a (low-high) range alongside at least one forecasted count."""
        self._login()
        data = {'file': (BytesIO(self._recent_consultation_csv_with_variance()), 'sample.csv'), '_csrf_token': self.csrf_token}
        self.client.post('/upload', data=data, content_type='multipart/form-data', follow_redirects=True)
        response = self.client.get('/predict')
        self.assertEqual(response.status_code, 200)
        body = response.data.decode('utf-8')
        self.assertIn('muted-copy" style="font-size:11px;">(', body)

    def test_predict_page_naive_fallback_has_no_crash_and_no_fabricated_range(self):
        """With too little data for real training (fewer than 8 months), the
        system falls back to a naive trend estimate -- it must render without
        error and must not show a range it has no model behind."""
        self._login()
        with self.app.app_context():
            branch = Branch.query.first()
            for month in range(1, 4):
                db.session.add(ConsultationRecord(
                    branch_id=branch.id,
                    consultation_date=f'2024-{month:02d}-01',
                    age_group='Adult',
                    gender='Male',
                    diagnosis='Hypertension',
                    department='General Medicine',
                    physician='Dr. Ada',
                    consultation_type='Follow-up',
                ))
            db.session.commit()
        self.client.post('/retrain', data={'_csrf_token': self.csrf_token}, follow_redirects=True)
        response = self.client.get('/predict')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Showing a cached or trend-based estimate', response.data)
        self.assertNotIn(b'muted-copy" style="font-size:11px;">(', response.data)

    def test_mapped_staff_role_demand_shows_required_range(self):
        self._login()
        data = {'file': (BytesIO(self._recent_consultation_csv_with_variance()), 'sample.csv'), '_csrf_token': self.csrf_token}
        self.client.post('/upload', data=data, content_type='multipart/form-data', follow_redirects=True)
        response = self.client.get('/predict')
        self.assertEqual(response.status_code, 200)
        body = response.data.decode('utf-8')
        idx = body.find('Mapped Staff Role Demand')
        self.assertGreater(idx, -1)
        section = body[idx:idx + 3000]
        self.assertIn('muted-copy" style="font-size:11px;">(', section)

    def test_common_diagnosis_metrics_exclude_rare_bucket(self):
        """Unit-level check that common_diagnoses is computed from real
        per-diagnosis volume and explicitly excludes the rare-diagnosis
        aggregate bucket, which is a grab-bag of many diagnoses, not one
        real common diagnosis."""
        rows = []
        for month in range(1, 13):
            for day in range(1, 6):
                rows.append({
                    'consultation_date': f'2024-{month:02d}-{day:02d}',
                    'age_group': 'Adult',
                    'gender': 'Male' if day % 2 == 0 else 'Female',
                    'diagnosis': 'Hypertension',
                    'department': 'General Medicine',
                    'physician': 'Dr. Ada',
                    'consultation_type': 'Follow-up',
                })
            rows.append({
                'consultation_date': f'2024-{month:02d}-15',
                'age_group': 'Adult',
                'gender': 'Male',
                'diagnosis': 'Diabetes',
                'department': 'General Medicine',
                'physician': 'Dr. Ada',
                'consultation_type': 'Follow-up',
            })
        # 20 distinct rare diagnoses, one record each -- well under
        # MIN_DIAGNOSIS_RECORDS_FOR_RF, and enough distinct labels that the
        # MIN_MODELED_DIAGNOSES_FOR_RF fallback doesn't keep them individually.
        for i in range(20):
            rows.append({
                'consultation_date': '2024-06-01',
                'age_group': 'Adult',
                'gender': 'Female',
                'diagnosis': f'Rare Condition {i}',
                'department': 'General Medicine',
                'physician': 'Dr. Ada',
                'consultation_type': 'New',
            })
        df = pd.DataFrame(rows)
        _, metrics, _, _ = train_and_evaluate_model(df, fast=True)
        self.assertNotIn(RARE_DIAGNOSIS_BUCKET, metrics['common_diagnoses'])
        self.assertIn('Hypertension', metrics['common_diagnoses'])
        self.assertLessEqual(metrics['common_diagnoses_count'], 10)

    def test_dashboard_shows_section_headings_and_pipeline_explainer(self):
        self._login()
        response = self.client.get('/dashboard')
        self.assertEqual(response.status_code, 200)
        body = response.data.decode('utf-8')
        for heading in ('Forecast Overview', 'Consultation Trend', 'Diagnosis Forecast', 'Demographic Forecast Detail', 'How the Prediction Works'):
            self.assertIn(heading, body)

    def test_resources_page_shows_source_badges(self):
        self._login()
        with self.app.app_context():
            branch = Branch.query.first()
            db.session.add(StaffMember(
                branch_id=branch.id, name='Dr. Ada', role='Internal Medicine Physicians', is_active=True,
            ))
            db.session.commit()
        csv_bytes = self._consultation_csv_with_physicians({
            'Hypertension': 'Dr. Ada',
            'Upper Respiratory Infection': 'Dr. Unlisted',
        })
        data = {'file': (BytesIO(csv_bytes), 'sample.csv'), '_csrf_token': self.csrf_token}
        self.client.post('/upload', data=data, content_type='multipart/form-data', follow_redirects=True)
        response = self.client.get('/resources')
        self.assertEqual(response.status_code, 200)
        body = response.data.decode('utf-8')
        self.assertIn('pill-source-historical', body)
        self.assertIn('pill-source-fallback', body)

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
        historical_match = re.search(r'pill-source-historical">\s*(\d+)\s*historical', body)
        fallback_match = re.search(r'pill-source-fallback">\s*(\d+)\s*fallback keyword', body)
        self.assertIsNotNone(historical_match, 'Expected the historical source badge on the Resources page')
        self.assertIsNotNone(fallback_match, 'Expected the fallback source badge on the Resources page')
        historical_count, fallback_count = int(historical_match.group(1)), int(fallback_match.group(1))
        self.assertGreaterEqual(historical_count, 1, 'Hypertension/Diabetes should be mapped from historical physician data')
        self.assertGreaterEqual(fallback_count, 1, 'Upper Respiratory Infection has no matching staff member and should fall back')

    def test_diagnosis_classifier_matches_unseen_diagnosis_by_text_similarity(self):
        """A diagnosis with no matching physician (so it never enters the
        historical map) should still be routed to a sensible role via TF-IDF
        text-similarity to a diagnosis that *does* have a confident historical
        mapping, instead of immediately collapsing to the generic keyword
        fallback."""
        self._login()
        with self.app.app_context():
            branch = Branch.query.first()
            db.session.add_all([
                StaffMember(branch_id=branch.id, name='Dr. Ada', role='Internal Medicine Physicians', is_active=True),
                StaffMember(branch_id=branch.id, name='Dr. Ben', role='General Physicians', is_active=True),
            ])
            db.session.commit()

        csv_bytes = self._consultation_csv_with_physicians({
            'Hypertension Monitoring': 'Dr. Ada',       # matches -> historical, Internal Medicine Physicians
            'Routine Medical Consultation': 'Dr. Ben',  # matches -> historical, General Physicians
            'Hypertensive Urgency': 'Dr. Unlisted',     # no matching staff -> absent from historical map,
                                                         # but textually close to 'Hypertension Monitoring'
        })
        data = {'file': (BytesIO(csv_bytes), 'sample.csv'), '_csrf_token': self.csrf_token}
        self.client.post('/upload', data=data, content_type='multipart/form-data', follow_redirects=True)
        self.client.post('/retrain', data={'_csrf_token': self.csrf_token}, follow_redirects=True)

        response = self.client.get('/resources')
        self.assertEqual(response.status_code, 200)
        body = response.data.decode('utf-8')
        historical_match = re.search(r'pill-source-historical">\s*(\d+)\s*historical', body)
        classifier_match = re.search(r'pill-source-classifier">\s*(\d+)\s*text-similarity', body)
        self.assertIsNotNone(historical_match, 'Expected the historical source badge on the Resources page')
        self.assertIsNotNone(classifier_match, 'Expected the text-similarity source badge on the Resources page')
        historical_count, classifier_count = int(historical_match.group(1)), int(classifier_match.group(1))
        self.assertGreaterEqual(historical_count, 1)
        self.assertGreaterEqual(
            classifier_count, 1,
            'Hypertensive Urgency should be routed by text-similarity to Hypertension Monitoring, not immediately fall back',
        )

    def test_attribution_gap_report_lists_non_historical_diagnoses_and_invisible_roles(self):
        """The Diagnosis Attribution Gap Report should surface diagnoses that were
        not confidently matched from history, and should dynamically list staff
        roles that never appear as a diagnosis's attributed role -- roles no
        diagnosis-based method (historical, classifier, or fallback) can ever
        recommend from this data."""
        self._login()
        with self.app.app_context():
            branch = Branch.query.first()
            db.session.add_all([
                StaffMember(branch_id=branch.id, name='Dr. Ada', role='Internal Medicine Physicians', is_active=True),
                StaffMember(branch_id=branch.id, name='Dr. Ben', role='General Physicians', is_active=True),
                # Never referenced as any diagnosis's physician -> a diagnosis-invisible role.
                StaffMember(branch_id=branch.id, name='Liza Ramos', role='Laboratory Technicians', is_active=True),
            ])
            db.session.commit()

        csv_bytes = self._consultation_csv_with_physicians({
            'Hypertension Monitoring': 'Dr. Ada',
            'Routine Medical Consultation': 'Dr. Ben',
            'Hypertensive Urgency': 'Dr. Unlisted',
        })
        data = {'file': (BytesIO(csv_bytes), 'sample.csv'), '_csrf_token': self.csrf_token}
        self.client.post('/upload', data=data, content_type='multipart/form-data', follow_redirects=True)
        self.client.post('/retrain', data={'_csrf_token': self.csrf_token}, follow_redirects=True)

        response = self.client.get('/reports/attribution-gaps')
        self.assertEqual(response.status_code, 200)
        body = response.data.decode('utf-8')
        self.assertIn('Diagnosis Attribution Gap Report', body)
        self.assertIn('Laboratory Technicians', body)
        self.assertIn('not something either attribution improvement is expected to fix', body)

    def test_attribution_gap_report_listed_on_reports_index(self):
        self._login()
        response = self.client.get('/reports')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Diagnosis Attribution Gap Report', response.data)

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

    # ------------------------------------------------------------------
    # RQ1 forecast-freshness fix: routes that add/remove ConsultationRecord
    # rows should force a real dashboard/forecast refresh, not just
    # invalidate the cache and leave the next viewer with a stale fallback.
    # ------------------------------------------------------------------

    def test_upload_force_refreshes_dashboard_without_manual_retrain(self):
        self._login()
        data = {'file': (BytesIO(self._recent_consultation_csv()), 'sample.csv'), '_csrf_token': self.csrf_token}
        self.client.post('/upload', data=data, content_type='multipart/form-data', follow_redirects=True)
        # No explicit /retrain call -- the fix is that /upload itself should
        # already force a real refresh.
        response = self.client.get('/predict')
        self.assertEqual(response.status_code, 200)
        self.assertNotIn(b'Showing a cached or trend-based estimate', response.data)
        self.assertIn(b'Model Validation Results', response.data)

    def test_completing_appointment_refreshes_dashboard_forecast(self):
        self._login()
        data = {'file': (BytesIO(self._recent_consultation_csv()), 'sample.csv'), '_csrf_token': self.csrf_token}
        self.client.post('/upload', data=data, content_type='multipart/form-data', follow_redirects=True)
        baseline = self.client.get('/dashboard')
        self.assertNotIn(b'Showing a cached or trend-based estimate', baseline.data)

        with self.app.app_context():
            branch = Branch.query.first()
            patient = Patient(
                branch_id=branch.id,
                patient_number='P-TEST-FRESH-0001',
                full_name='Freshness Test Patient',
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
                appointment_date='2024-01-02',
                status='Confirmed',
            )
            db.session.add(appointment)
            db.session.commit()
            appointment_id = appointment.id

        self.client.post(f'/appointments/{appointment_id}/complete', data={
            '_csrf_token': self.csrf_token,
            'completed_date': '2024-01-02',
            'assigned_staff_0': 'Dr. Ada',
            'final_diagnosis_0': 'Check-up',
            'service_notes_0': '',
        }, follow_redirects=True)

        # No manual retrain -- completing the appointment should have already
        # force-refreshed the dashboard with the newly added consultation record.
        response = self.client.get('/dashboard')
        self.assertEqual(response.status_code, 200)
        self.assertNotIn(b'Showing a cached or trend-based estimate', response.data)

    def test_clear_records_resets_dashboard_to_empty_state(self):
        self._login()
        data = {'file': (BytesIO(self._sample_consultation_csv()), 'sample.csv'), '_csrf_token': self.csrf_token}
        self.client.post('/upload', data=data, content_type='multipart/form-data', follow_redirects=True)
        with self.app.app_context():
            self.assertGreater(ConsultationRecord.query.count(), 0)

        self.client.post('/records/clear', data={'_csrf_token': self.csrf_token}, follow_redirects=True)

        response = self.client.get('/dashboard')
        self.assertEqual(response.status_code, 200)
        with self.app.app_context():
            remaining = ConsultationRecord.query.count()
        self.assertEqual(remaining, 0)
        self.assertIn(b'<div class="value">0</div>', response.data)

    @staticmethod
    def _csv_with_dominant_rare_bucket():
        """A rolling window ending in the current real month (so the RF
        forecast branch, not the naive fallback, is exercised) with 2
        clearly-common diagnoses and 25 individually-rare ones whose combined
        volume dominates -- guarantees 'Other Services/Cases' both forms and
        lands near the top of the forecasted-diagnosis list."""
        rows = ['consultation_date,age_group,gender,diagnosis,department,physician,consultation_type']
        now = datetime.now()
        months = []
        year, month = now.year, now.month
        for _ in range(10):
            months.append((year, month))
            month -= 1
            if month == 0:
                month = 12
                year -= 1
        months.reverse()

        common_diagnoses = ['Hypertension', 'Diabetes']
        rare_diagnoses = [f'Rare Condition {i}' for i in range(25)]
        for year, month in months:
            for index, diagnosis in enumerate(common_diagnoses):
                for k in range(6):
                    day = (index * 6 + k) % 27 + 1
                    rows.append(f'{year}-{month:02d}-{day:02d},Adult,Male,{diagnosis},General Medicine,Dr. Ada,Follow-up')
            for index, diagnosis in enumerate(rare_diagnoses):
                for k in range(2):
                    day = (index * 2 + k) % 27 + 1
                    rows.append(f'{year}-{month:02d}-{day:02d},Adult,Female,{diagnosis},General Medicine,Dr. Ada,New')
        return ('\n'.join(rows) + '\n').encode('utf-8')

    def test_other_services_cases_hidden_from_forecast_table(self):
        """The rare-diagnosis aggregate bucket is a synthetic training-time
        label -- a grab-bag of unrelated low-volume diagnoses, not a real
        clinical need -- so it's excluded from the forecasted-diagnosis
        table entirely rather than shown with a fabricated trend."""
        self._login()
        data = {'file': (BytesIO(self._csv_with_dominant_rare_bucket()), 'sample.csv'), '_csrf_token': self.csrf_token}
        self.client.post('/upload', data=data, content_type='multipart/form-data', follow_redirects=True)
        response = self.client.get('/predict')
        self.assertEqual(response.status_code, 200)
        body = response.data.decode('utf-8')
        row_match = re.search(r'<tr>\s*<td>Other Services/Cases.*?</tr>', body, re.S)
        self.assertIsNone(row_match, "Expected 'Other Services/Cases' to be excluded from the forecasted diagnosis table")

    def test_appointment_completion_carries_forward_demographic_comparison(self):
        """After completing an appointment (which skips the redundant Model B
        fit for speed), the Predictions page should still show the last real
        With-vs-Without-Demographics comparison instead of going dark until
        the next manual retrain."""
        self._login()
        data = {'file': (BytesIO(self._recent_consultation_csv()), 'sample.csv'), '_csrf_token': self.csrf_token}
        self.client.post('/upload', data=data, content_type='multipart/form-data', follow_redirects=True)
        self.client.post('/retrain', data={'_csrf_token': self.csrf_token}, follow_redirects=True)
        baseline = self.client.get('/predict')
        self.assertIn(b'Model B (Without Demographics)', baseline.data)

        with self.app.app_context():
            branch = Branch.query.first()
            patient = Patient(
                branch_id=branch.id,
                patient_number='P-TEST-CARRY-0001',
                full_name='Carry Forward Test Patient',
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
                appointment_date=datetime.now().strftime('%Y-%m-%d'),
                status='Confirmed',
            )
            db.session.add(appointment)
            db.session.commit()
            appointment_id = appointment.id

        self.client.post(f'/appointments/{appointment_id}/complete', data={
            '_csrf_token': self.csrf_token,
            'completed_date': datetime.now().strftime('%Y-%m-%d'),
            'assigned_staff_0': 'Dr. Ada',
            'final_diagnosis_0': 'Check-up',
            'service_notes_0': '',
        }, follow_redirects=True)

        response = self.client.get('/predict')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Model B (Without Demographics)', response.data)
        self.assertIn(b'from the last full retrain', response.data)


class ChatbotApiTests(unittest.TestCase):
    """The admin-only analytics chatbot: /api/chatbot/ask and its read-only
    tool registry (exposed on the app instance as app.chatbot_tools since the
    tools are closures over create_app()'s locals). LLM backend is the Gemini
    API -- run_chatbot_turn is mocked in every test here so the suite never
    makes a real API call or depends on a configured GEMINI_API_KEY."""

    def setUp(self):
        self.app = app
        self.app.config.update(
            TESTING=True,
            SECRET_KEY='test-secret',
            UPLOAD_FOLDER=tempfile.mkdtemp(prefix='clinic_test_uploads_'),
            GEMINI_API_KEY='test-key',
            GEMINI_MODEL='gemini-3.6-flash',
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

    def _fresh_csrf_token(self):
        """A valid CSRF token without logging in (just enough to reach the
        route's own auth check instead of getting rejected earlier by the
        blanket CSRF check that runs before it)."""
        return self._extract_csrf_token(self.client.get('/login').data)

    def _login(self, username='admin', password='admin123'):
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

    def _ask(self, question, conversation_id=None, csrf_token=None):
        return self.client.post(
            '/api/chatbot/ask',
            json={'question': question, 'conversation_id': conversation_id},
            headers={'X-CSRFToken': csrf_token if csrf_token is not None else self.csrf_token},
        )

    def test_unauthenticated_request_is_rejected(self):
        token = self._fresh_csrf_token()
        response = self._ask('What is the top diagnosis?', csrf_token=token)
        # Not logged in: require_login()'s protected_endpoints check redirects
        # to /login before the route body's own role check ever runs.
        self.assertEqual(response.status_code, 302)
        self.assertIn('/login', response.headers.get('Location', ''))

    def test_missing_csrf_token_is_rejected_as_json(self):
        self._login()
        response = self._ask('What is the top diagnosis?', csrf_token='not-the-real-token')
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json().get('error'), 'csrf')

    def test_non_main_admin_role_is_rejected(self):
        with self.app.app_context():
            branch = Branch.query.first()
            db.session.add(User(
                username='branchstaff', password=generate_password_hash('pass12345'),
                role='staff', branch_id=branch.id, must_change_password=False,
            ))
            db.session.commit()
        self._login(username='branchstaff', password='pass12345')
        response = self._ask('What is the top diagnosis?')
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.get_json().get('error'), 'forbidden')

    def test_gemini_unreachable_returns_friendly_message(self):
        """Reproduces what happens when the Gemini API can't be reached or
        the API key is missing/invalid -- the route should degrade
        gracefully, not 500."""
        self._login()
        with patch('app.run_chatbot_turn') as mock_turn:
            mock_turn.side_effect = GeminiError('GEMINI_API_KEY is not configured.')
            response = self._ask('What is the top diagnosis?')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('not reachable', data['answer'])
        self.assertIn('An internal error occurred', data['warnings'][0])

    def test_real_tool_backed_round_trip_is_logged(self):
        self._login()
        with self.app.app_context():
            db.session.add(ConsultationRecord(
                consultation_date='2024-01-01', age_group='Adult', gender='Male',
                diagnosis='Hypertension Monitoring', department='General Medicine',
                physician='Dr. Ada', consultation_type='New',
            ))
            db.session.commit()

        with patch('app.run_chatbot_turn') as mock_turn:
            mock_turn.return_value = (
                'Hypertension Monitoring has the most consultations.',
                [{'tool': 'tool_dashboard_summary', 'args': {}}],
            )
            response = self._ask('What is the most common diagnosis?')

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['answer'], 'Hypertension Monitoring has the most consultations.')
        self.assertTrue(data['conversation_id'])
        self.assertEqual(data['tools_used'], ['tool_dashboard_summary'])

        with self.app.app_context():
            interaction = ChatbotInteraction.query.order_by(ChatbotInteraction.id.desc()).first()
            self.assertIsNotNone(interaction)
            self.assertEqual(interaction.answer, 'Hypertension Monitoring has the most consultations.')
            self.assertIn('tool_dashboard_summary', interaction.tools_used)
            audit_row = AuditLog.query.filter_by(action='chatbot_query').order_by(AuditLog.id.desc()).first()
            self.assertIsNotNone(audit_row)
            self.assertEqual(audit_row.entity_type, 'ChatbotInteraction')

    def test_followup_question_receives_prior_turn_as_history(self):
        self._login()

        with patch('app.run_chatbot_turn') as mock_turn:
            mock_turn.return_value = ('Hypertension Monitoring.', [{'tool': 'tool_predictions', 'args': {}}])
            first = self._ask('What diagnosis is predicted to be highest?')
        conversation_id = first.get_json()['conversation_id']

        with patch('app.run_chatbot_turn') as mock_turn:
            mock_turn.return_value = (
                'Because it has the highest predicted case count with strong historical support.',
                [{'tool': 'tool_staff_recommendation', 'args': {'diagnosis': 'Hypertension Monitoring'}}],
            )
            second = self._ask('Why?', conversation_id=conversation_id)

        self.assertEqual(second.status_code, 200)
        _args, kwargs = mock_turn.call_args
        history = kwargs.get('history')
        self.assertTrue(history, 'expected the prior turn to be passed as history')
        self.assertEqual(history[-1]['q'], 'What diagnosis is predicted to be highest?')
        self.assertEqual(history[-1]['a'], 'Hypertension Monitoring.')

    def test_tool_outputs_never_contain_patient_identifying_info(self):
        """The most important guardrail: whatever the LLM decides to call,
        no tool may ever surface patient name/contact/address, regardless of
        the arguments passed in."""
        with self.app.app_context():
            branch = Branch.query.first()
            patient = Patient(
                branch_id=branch.id, patient_number='P-CHATBOT-0001',
                full_name='Extremely Unique Patient Name Zzyzx', birthdate='1990-01-01', age=34,
                age_group='Adult', gender='Male', contact_number='0917-555-0199',
                email='zzyzx.patient@example.com',
            )
            db.session.add(patient)
            db.session.commit()
            db.session.add(ConsultationRecord(
                branch_id=branch.id, consultation_date='2024-01-01', age_group='Adult', gender='Male',
                diagnosis='Hypertension Monitoring', department='General Medicine', physician='Dr. Ada',
                consultation_type='New', patient_id=patient.id,
            ))
            db.session.commit()
            branch_id = branch.id

        forbidden_strings = ['Extremely Unique Patient Name Zzyzx', '0917-555-0199', 'zzyzx.patient@example.com']

        with self.app.test_request_context('/'):
            session['user_id'] = 1
            session['role'] = 'superadmin'
            session['branch_id'] = branch_id

            calls = [
                ('tool_dashboard_summary', {}),
                ('tool_predictions', {}),
                ('tool_diagnosis_trends', {'diagnosis': 'Hypertension Monitoring'}),
                ('tool_staff_recommendation', {'diagnosis': 'Hypertension Monitoring'}),
                ('tool_department_demand', {}),
                ('tool_resource_capacity', {}),
                ('tool_query_consultations', {'diagnosis': 'Hypertension'}),
                ('tool_historical_analysis', {'date_from': '2024-01-01', 'date_to': '2024-12-31'}),
                ('tool_model_metrics', {}),
            ]
            for tool_name, kwargs in calls:
                result = self.app.chatbot_tools[tool_name](**kwargs)
                serialized = json.dumps(result, default=str)
                for forbidden in forbidden_strings:
                    self.assertNotIn(
                        forbidden, serialized,
                        msg=f'{tool_name} leaked patient-identifying info: {forbidden!r}',
                    )

    def test_staff_recommendation_flags_low_evidence(self):
        with self.app.test_request_context('/'):
            session['user_id'] = 1
            session['role'] = 'superadmin'
            result = self.app.chatbot_tools['tool_staff_recommendation'](diagnosis='Totally Unseen Diagnosis Xyzzy')
        self.assertEqual(result['source'], 'fallback')
        self.assertTrue(result['low_evidence'])

    def test_backtest_forecast_excludes_target_month_and_reports_actual(self):
        with self.app.app_context():
            branch = Branch.query.first()
            branch_id = branch.id
            # A handful of records is intentionally below the tool's
            # minimum-training-rows threshold, exercising the safe
            # "not enough data" path rather than a full (slow) RF retrain --
            # the 'actual' count and training-cutoff exclusion are checked
            # either way.
            for day in ('01', '02', '03'):
                db.session.add(ConsultationRecord(
                    branch_id=branch.id, consultation_date=f'2024-02-{day}', age_group='Adult', gender='Male',
                    diagnosis='Hypertension Monitoring', department='General Medicine', physician='Dr. Ada',
                    consultation_type='New',
                ))
            db.session.add(ConsultationRecord(
                branch_id=branch.id, consultation_date='2024-01-15', age_group='Adult', gender='Male',
                diagnosis='Hypertension Monitoring', department='General Medicine', physician='Dr. Ada',
                consultation_type='New',
            ))
            db.session.commit()

        with self.app.test_request_context('/'):
            session['user_id'] = 1
            session['role'] = 'superadmin'
            session['branch_id'] = branch_id
            result = self.app.chatbot_tools['tool_backtest_forecast'](
                diagnosis='Hypertension Monitoring', target_month=2, target_year=2024,
            )

        self.assertEqual(result['actual'], 3)
        self.assertEqual(result['training_cutoff'], '2024-02-01')

    def test_future_forecast_rejects_a_month_that_is_not_in_the_future(self):
        with self.app.app_context():
            branch = Branch.query.first()
            branch_id = branch.id
            db.session.add(ConsultationRecord(
                branch_id=branch.id, consultation_date='2024-01-15', age_group='Adult', gender='Male',
                diagnosis='Hypertension Monitoring', department='General Medicine', physician='Dr. Ada',
                consultation_type='New',
            ))
            db.session.commit()

        with self.app.test_request_context('/'):
            session['user_id'] = 1
            session['role'] = 'superadmin'
            session['branch_id'] = branch_id
            # A definitely-past month regardless of when this test runs.
            result = self.app.chatbot_tools['tool_future_forecast'](target_month=1, target_year=2000)

        self.assertIn('error', result)
        self.assertIn('tool_predictions', result['error'])

    def test_future_forecast_rejects_a_month_too_far_out(self):
        with self.app.app_context():
            branch = Branch.query.first()
            branch_id = branch.id
            db.session.add(ConsultationRecord(
                branch_id=branch.id, consultation_date='2024-01-15', age_group='Adult', gender='Male',
                diagnosis='Hypertension Monitoring', department='General Medicine', physician='Dr. Ada',
                consultation_type='New',
            ))
            db.session.commit()

        with self.app.test_request_context('/'):
            session['user_id'] = 1
            session['role'] = 'superadmin'
            session['branch_id'] = branch_id
            # Decades out regardless of when this test runs -- must exceed
            # MAX_FUTURE_FORECAST_MONTHS from "next month".
            result = self.app.chatbot_tools['tool_future_forecast'](target_month=1, target_year=2099)

        self.assertIn('error', result)
        self.assertIn('too far', result['error'])


class StaffingGapNotificationTests(unittest.TestCase):
    """The 'Staffing Gap This Week' dashboard banner and its per-role
    'Notify' button (POST /staff/notify-gap). Email sending is always
    mocked here -- never exercise the real SMTP path in tests."""

    def setUp(self):
        self.app = app
        self.app.config.update(TESTING=True, SECRET_KEY='test-secret')
        self.client = self.app.test_client()
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
                data={'_csrf_token': self.csrf_token, 'new_password': 'test-password-123', 'confirm_password': 'test-password-123'},
                follow_redirects=True,
            )
        return response

    def test_notify_requires_a_specific_branch(self):
        self._login()
        with self.client.session_transaction() as sess:
            sess['selected_branch_id'] = 'all'
        response = self.client.post(
            '/staff/notify-gap',
            data={'role': 'General Physicians', '_csrf_token': self.csrf_token},
            follow_redirects=True,
        )
        self.assertIn(b'Select a specific branch', response.data)

    def test_notify_with_no_matching_staff_flashes_error(self):
        self._login()
        response = self.client.post(
            '/staff/notify-gap',
            data={'role': 'Nonexistent Role', '_csrf_token': self.csrf_token},
            follow_redirects=True,
        )
        self.assertIn(b'No active Nonexistent Role staff found', response.data)

    def test_notify_staff_without_email_never_calls_send_and_flags_missing(self):
        with self.app.app_context():
            branch = Branch.query.first()
            db.session.add(StaffMember(branch_id=branch.id, name='Dr. No Email', role='General Physicians', is_active=True))
            db.session.commit()
        self._login()
        with patch.object(self.app, 'send_appointment_email') as mock_send:
            response = self.client.post(
                '/staff/notify-gap',
                data={'role': 'General Physicians', '_csrf_token': self.csrf_token},
                follow_redirects=True,
            )
        mock_send.assert_not_called()
        self.assertIn(b'none of them have an email on file', response.data)

    def test_notify_sends_to_staff_with_email_and_logs_audit(self):
        with self.app.app_context():
            branch = Branch.query.first()
            db.session.add(StaffMember(
                branch_id=branch.id, name='Dr. Ada', role='General Physicians',
                email='dr.ada@example.com', is_active=True,
            ))
            db.session.commit()
        self._login()
        with patch.object(self.app, 'send_appointment_email', return_value=True) as mock_send:
            response = self.client.post(
                '/staff/notify-gap',
                data={'role': 'General Physicians', '_csrf_token': self.csrf_token},
                follow_redirects=True,
            )
        mock_send.assert_called_once()
        recipient = mock_send.call_args[0][0]
        self.assertEqual(recipient, 'dr.ada@example.com')
        self.assertIn(b'1 General Physicians staff member(s) were notified', response.data)

        with self.app.app_context():
            audit_row = AuditLog.query.filter_by(action='notify_staffing_gap').order_by(AuditLog.id.desc()).first()
            self.assertIsNotNone(audit_row)
            self.assertIn('General Physicians', audit_row.details)

    def test_weekly_staffing_gaps_only_includes_needs_staff_rows(self):
        summary = {
            'staff_demand_forecast': {
                'daily_prediction': {
                    'current_week_rows': [
                        {'staff_role': 'General Physicians', 'required_staff': 3, 'available_staff': 1, 'status_class': 'high', 'date': '2026-09-01', 'day_name': 'Tue'},
                        {'staff_role': 'General Physicians', 'required_staff': 2, 'available_staff': 1, 'status_class': 'high', 'date': '2026-09-02', 'day_name': 'Wed'},
                        {'staff_role': 'Laboratory Technicians', 'required_staff': 2, 'available_staff': 3, 'status_class': 'healthy', 'date': '2026-09-01', 'day_name': 'Tue'},
                    ],
                },
            },
        }
        gaps = self.app.weekly_staffing_gaps(summary)
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]['role'], 'General Physicians')
        self.assertEqual(gaps[0]['gap'], 2)  # the worse of the two "high" days (3-1), not the milder one (2-1)

    def test_staff_page_renders_selection_checkboxes(self):
        with self.app.app_context():
            branch = Branch.query.first()
            db.session.add(StaffMember(branch_id=branch.id, name='Dr. Ada', role='General Physicians', email='dr.ada@example.com', is_active=True))
            db.session.commit()
        self._login()
        response = self.client.get('/staff')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'notifySelectedForm', response.data)
        self.assertIn(b'staff_ids', response.data)
        self.assertIn(b'dr.ada@example.com', response.data)

    def test_notify_selected_with_no_selection_flashes_error(self):
        self._login()
        response = self.client.post(
            '/staff/notify-selected',
            data={'_csrf_token': self.csrf_token},
            follow_redirects=True,
        )
        self.assertIn(b'Select at least one staff member', response.data)

    def test_notify_selected_sends_custom_message_to_chosen_staff_only(self):
        with self.app.app_context():
            branch = Branch.query.first()
            picked = StaffMember(branch_id=branch.id, name='Dr. Ada', role='General Physicians', email='dr.ada@example.com', is_active=True)
            not_picked = StaffMember(branch_id=branch.id, name='Dr. Ben', role='General Physicians', email='dr.ben@example.com', is_active=True)
            db.session.add_all([picked, not_picked])
            db.session.commit()
            picked_id = picked.id
        self._login()
        with patch.object(self.app, 'send_appointment_email', return_value=True) as mock_send:
            response = self.client.post(
                '/staff/notify-selected',
                data={'staff_ids': [str(picked_id)], 'message': 'Can you cover Friday?', '_csrf_token': self.csrf_token},
                follow_redirects=True,
            )
        mock_send.assert_called_once()
        args = mock_send.call_args[0]
        self.assertEqual(args[0], 'dr.ada@example.com')
        self.assertEqual(args[2], 'Can you cover Friday?')
        self.assertIn(b'1 staff member(s) were notified', response.data)
        self.assertIn(b'Dr. Ada', response.data)

        with self.app.app_context():
            audit_row = AuditLog.query.filter_by(action='notify_selected_staff').order_by(AuditLog.id.desc()).first()
            self.assertIsNotNone(audit_row)


if __name__ == '__main__':
    unittest.main()
