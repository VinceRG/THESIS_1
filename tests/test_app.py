import os
import tempfile
import unittest
from io import BytesIO

import pandas as pd

from app import app, ConsultationRecord, StaffMember, User, build_training_frame, db, init_db, prepare_monthly_data


class SmartClinicAppTests(unittest.TestCase):
    def setUp(self):
        self.app = app
        self.app.config.update(TESTING=True, SECRET_KEY='test-secret')
        self.client = self.app.test_client()
        with self.app.app_context():
            db.drop_all()
            db.create_all()
            init_db()

    def test_login_page_renders(self):
        response = self.client.get('/login')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Login', response.data)

    def test_dashboard_requires_login(self):
        response = self.client.get('/dashboard', follow_redirects=True)
        self.assertIn(b'Login', response.data)

    def test_prepare_monthly_data_with_no_records(self):
        with self.app.app_context():
            df = prepare_monthly_data()
        self.assertTrue(df.empty)

    def test_dashboard_shows_prediction_month_labels(self):
        self.client.post('/login', data={'username': 'admin', 'password': 'admin123'})
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

    def test_upload_trains_model_and_writes_metrics_report(self):
        self.client.post('/login', data={'username': 'admin', 'password': 'admin123'})
        csv_data = b'consultation_date,age_group,gender,diagnosis,department,physician,consultation_type\n2024-01-01,Adult,Male,Hypertension,General Medicine,Dr. Ada,Follow-up\n2024-01-02,Adult,Female,Diabetes,General Medicine,Dr. Ada,New\n2024-02-01,Adult,Male,Hypertension,General Medicine,Dr. Ada,Follow-up\n2024-02-02,Adult,Female,Diabetes,General Medicine,Dr. Ada,New\n2024-03-01,Adult,Male,Upper Respiratory Infection,General Medicine,Dr. Ada,New\n2024-03-02,Adult,Female,Hypertension,General Medicine,Dr. Ada,Follow-up\n'
        data = {'file': (BytesIO(csv_data), 'sample.csv')}
        response = self.client.post('/upload', data=data, content_type='multipart/form-data', follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        report_path = os.path.join(self.app.config['UPLOAD_FOLDER'], 'training_report.txt')
        self.assertTrue(os.path.exists(report_path))

    def test_predict_page_uses_metrics_from_report(self):
        self.client.post('/login', data={'username': 'admin', 'password': 'admin123'})
        csv_data = b'consultation_date,age_group,gender,diagnosis,department,physician,consultation_type\n2024-01-01,Adult,Male,Hypertension,General Medicine,Dr. Ada,Follow-up\n2024-01-02,Adult,Female,Diabetes,General Medicine,Dr. Ada,New\n2024-02-01,Adult,Male,Hypertension,General Medicine,Dr. Ada,Follow-up\n2024-02-02,Adult,Female,Diabetes,General Medicine,Dr. Ada,New\n2024-03-01,Adult,Male,Upper Respiratory Infection,General Medicine,Dr. Ada,New\n2024-03-02,Adult,Female,Hypertension,General Medicine,Dr. Ada,Follow-up\n'
        data = {'file': (BytesIO(csv_data), 'sample.csv')}
        self.client.post('/upload', data=data, content_type='multipart/form-data', follow_redirects=True)
        response = self.client.get('/predict')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'R2 Score', response.data)
        self.assertIn(b'MAE', response.data)
        self.assertIn(b'MSE', response.data)
        self.assertIn(b'RMSE', response.data)
        self.assertNotIn(b'0.0', response.data)

    def test_build_training_frame_augments_sparse_data(self):
        sparse_df = pd.DataFrame([
            {'consultation_date': '2024-01-01', 'age_group': 'Adult', 'gender': 'Male', 'diagnosis': 'Hypertension', 'department': 'General Medicine', 'physician': 'Dr. Ada', 'consultation_type': 'New'},
            {'consultation_date': '2024-02-01', 'age_group': 'Adult', 'gender': 'Female', 'diagnosis': 'Diabetes', 'department': 'General Medicine', 'physician': 'Dr. Ada', 'consultation_type': 'Follow-up'},
        ])
        augmented_df = build_training_frame(sparse_df)
        self.assertGreaterEqual(len(augmented_df), 20)


if __name__ == '__main__':
    unittest.main()
