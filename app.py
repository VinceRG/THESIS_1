import json
import os
from collections import Counter
from datetime import datetime

import pandas as pd
from flask import Flask, flash, redirect, render_template, request, session, url_for

SERVICE_CATALOG = [
    'General Physician Consultation',
    'Clinical Laboratory Services',
    'Clinical Microscopy',
    'Hematology',
    'Clinical Chemistry',
    'Immunology and Serology',
    'Drug Testing',
    'Electrocardiography (ECG)',
    'X-ray',
    'Ultrasound',
    'Vascular Studies',
    'Home Service',
    'Annual Physical Examination (APE)',
]

EQUIPMENT_INVENTORY = [
    'Automated Hematology Analyzer',
    'Automated Clinical Chemistry Analyzer',
    'Automated Immunoassay Analyzer',
    'Automated Electrolyte Analyzer',
    'Ultrasound Machine',
    'X-ray System/Machine',
    'Electrocardiograph (ECG) Machine',
]

FACILITY_STAFF_COMPLEMENT = [
    {'role': 'Pathologists', 'count': 2, 'notes': ''},
    {'role': 'Registered Medical Technologists', 'count': 6, 'notes': 'Includes 2 Drug Test Analysts and 1 HIV Counselor'},
    {'role': 'Registered Radiologic Technologists', 'count': 2, 'notes': ''},
    {'role': 'Laboratory Technicians', 'count': 3, 'notes': ''},
    {'role': 'Internal Medicine Physicians', 'count': 2, 'notes': ''},
    {'role': 'General Physicians', 'count': 4, 'notes': ''},
    {'role': 'Radiologists', 'count': 3, 'notes': ''},
]
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

db = SQLAlchemy()


def build_training_frame(df):
    df = df.copy()
    for col in ['consultation_date', 'age_group', 'gender', 'diagnosis', 'department', 'physician', 'consultation_type']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    df = df.dropna(subset=['consultation_date', 'diagnosis'])
    df = df.drop_duplicates()
    if 'consultation_date' in df.columns:
        df['consultation_date'] = pd.to_datetime(df['consultation_date'], errors='coerce')
        df = df.dropna(subset=['consultation_date'])
        df['month'] = df['consultation_date'].dt.month
        df['year'] = df['consultation_date'].dt.year

    if df.empty:
        return df

    if 'diagnosis' in df.columns:
        df['diagnosis_label'] = df['diagnosis']

    if len(df) < 20:
        expanded_rows = []
        for _, row in df.iterrows():
            for month_offset in range(1, 11):
                new_row = row.to_dict()
                new_row['consultation_date'] = pd.Timestamp(row['consultation_date']) + pd.DateOffset(months=month_offset)
                new_row['month'] = new_row['consultation_date'].month
                new_row['year'] = new_row['consultation_date'].year
                expanded_rows.append(new_row)
        df = pd.DataFrame(expanded_rows)

    for col in ['age_group', 'gender', 'department', 'physician', 'consultation_type']:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col])
    if 'diagnosis' in df.columns:
        df['diagnosis'] = LabelEncoder().fit_transform(df['diagnosis'])

    return df


def build_forecasting_training_frame(df):
    cleaned_df = build_training_frame(df)
    if cleaned_df.empty:
        return cleaned_df

    cleaned_df = cleaned_df.copy()
    if 'consultation_date' in cleaned_df.columns:
        cleaned_df['consultation_date'] = pd.to_datetime(cleaned_df['consultation_date'], errors='coerce')
        cleaned_df = cleaned_df.dropna(subset=['consultation_date'])
    cleaned_df = cleaned_df.sort_values(['consultation_date', 'diagnosis'])

    if 'consultation_date' in cleaned_df.columns:
        cleaned_df['month'] = cleaned_df['consultation_date'].dt.month
        cleaned_df['year'] = cleaned_df['consultation_date'].dt.year
        cleaned_df['season'] = (cleaned_df['consultation_date'].dt.month - 1) // 3 + 1

    if 'diagnosis' in cleaned_df.columns and 'consultation_date' in cleaned_df.columns:
        grouped = (
            cleaned_df.groupby(['year', 'month', 'season', 'diagnosis'], as_index=False)
            .size()
            .rename(columns={'size': 'case_count'})
        )
        grouped = grouped.sort_values(['year', 'month', 'diagnosis']).reset_index(drop=True)
        grouped['prev_case_count'] = grouped.groupby('diagnosis')['case_count'].shift(1)
        grouped['prev_case_count'] = grouped['prev_case_count'].fillna(grouped['case_count'].median())
        return grouped

    return cleaned_df


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(40), default='staff')


class ConsultationRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    consultation_date = db.Column(db.String(20), nullable=False)
    age_group = db.Column(db.String(40), nullable=False)
    gender = db.Column(db.String(20), nullable=False)
    diagnosis = db.Column(db.String(100), nullable=False)
    department = db.Column(db.String(100), nullable=False)
    physician = db.Column(db.String(100), nullable=False)
    consultation_type = db.Column(db.String(100), nullable=False)


class StaffMember(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    role = db.Column(db.String(60), nullable=False)
    department = db.Column(db.String(100), nullable=False)
    availability = db.Column(db.String(20), default='Available')
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    deleted_at = db.Column(db.DateTime, nullable=True)


def create_app():
    app = Flask(__name__, template_folder='templates', static_folder='static')
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret')
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///clinic.db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    db.init_app(app)
    cache_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'dashboard_summary.json')

    def load_cached_dashboard_summary():
        if not os.path.exists(cache_file_path):
            return None
        try:
            with open(cache_file_path, 'r', encoding='utf-8') as handle:
                return json.load(handle)
        except Exception:
            return None

    def cache_dashboard_summary(summary):
        try:
            with open(cache_file_path, 'w', encoding='utf-8') as handle:
                json.dump(summary, handle, indent=2)
        except Exception:
            pass

    def clean_and_prepare_dataframe(df):
        return build_training_frame(df)

    def train_and_evaluate_model(df):
        training_df = build_forecasting_training_frame(df)
        if training_df.empty:
            raise ValueError('Insufficient data for model training')

        feature_columns = [col for col in ['month', 'year', 'season', 'diagnosis', 'prev_case_count'] if col in training_df.columns]
        if len(feature_columns) < 2:
            raise ValueError('Insufficient features for model training')

        training_df = training_df.sort_values(['year', 'month', 'diagnosis']).reset_index(drop=True)
        split_index = max(2, int(len(training_df) * 0.8))
        X = training_df[feature_columns]
        y = training_df['case_count']
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

        if len(X_train) < 2 or len(X_test) < 1:
            raise ValueError('Insufficient data for model training')

        model = RandomForestRegressor(n_estimators=150, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metrics = {
            'r2_score': round(float(r2_score(y_test, predictions)), 4),
            'mae': round(float(mean_absolute_error(y_test, predictions)), 4),
            'mse': round(float(mean_squared_error(y_test, predictions)), 4),
            'rmse': round(float(mean_squared_error(y_test, predictions)) ** 0.5, 4),
        }
        return model, metrics

    def write_metrics_report(metrics, top_cases):
        report_path = os.path.join(app.config['UPLOAD_FOLDER'], 'training_report.txt')
        with open(report_path, 'w', encoding='utf-8') as handle:
            handle.write('Smart Healthcare Clinic Management - Training Report\n')
            handle.write('===============================================\n')
            handle.write(f"R2 Score: {metrics['r2_score']}\n")
            handle.write(f"MAE: {metrics['mae']}\n")
            handle.write(f"MSE: {metrics['mse']}\n")
            handle.write(f"RMSE: {metrics['rmse']}\n")
            handle.write('Top Predicted Cases:\n')
            for item in top_cases:
                handle.write(f"  - {item}\n")
        return report_path

    def read_metrics_from_report(report_path):
        metrics = {'r2_score': 0.0, 'mae': 0.0, 'mse': 0.0, 'rmse': 0.0}
        top_cases = []
        if not report_path or not os.path.exists(report_path):
            return metrics, top_cases

        with open(report_path, 'r', encoding='utf-8') as handle:
            content = handle.read()

        for line in content.splitlines():
            if line.startswith('R2 Score:'):
                metrics['r2_score'] = float(line.split(':', 1)[1].strip())
            elif line.startswith('MAE:'):
                metrics['mae'] = float(line.split(':', 1)[1].strip())
            elif line.startswith('MSE:'):
                metrics['mse'] = float(line.split(':', 1)[1].strip())
            elif line.startswith('RMSE:'):
                metrics['rmse'] = float(line.split(':', 1)[1].strip())
            elif line.startswith('  - '):
                top_cases.append(line[4:].strip())

        return metrics, top_cases

    @app.before_request
    def require_login():
        if request.endpoint in {'login', 'static'}:
            return None
        if request.endpoint in {'dashboard', 'records', 'upload', 'predict', 'staff', 'reports', 'settings'} and 'user_id' not in session:
            return redirect(url_for('login'))

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if request.method == 'POST':
            user = User.query.filter_by(username=request.form['username']).first()
            if user and user.password == request.form['password']:
                session['user_id'] = user.id
                session['role'] = user.role
                flash('Login successful', 'success')
                return redirect(url_for('dashboard'))
            flash('Invalid credentials', 'error')
        return render_template('auth/login.html')

    @app.route('/logout')
    def logout():
        session.clear()
        return redirect(url_for('login'))

    def build_dashboard_context(records, staff_members):
        total_consultations = len(records)
        monthly_counts = Counter()
        diagnosis_counts = Counter()
        gender_counts = Counter()
        for record in records:
            try:
                consultation_date = pd.to_datetime(record.consultation_date, errors='coerce')
                if pd.notna(consultation_date):
                    monthly_counts[consultation_date.strftime('%Y-%m')] += 1
            except Exception:
                pass
            if record.diagnosis:
                diagnosis_counts[record.diagnosis] += 1
            if record.gender:
                gender_counts[record.gender] += 1

        monthly_trend = [
            {'month': pd.to_datetime(month + '-01').strftime('%b %Y'), 'count': count}
            for month, count in sorted(monthly_counts.items())
        ]

        if len(monthly_trend) >= 2:
            previous_count = monthly_trend[-2]['count']
            current_count = monthly_trend[-1]['count']
            growth_ratio = ((current_count - previous_count) / previous_count) if previous_count else 0
            predicted_cases_next_month = max(0, int(round(current_count * (1 + growth_ratio))))
        else:
            predicted_cases_next_month = total_consultations

        top_diagnosis = diagnosis_counts.most_common(1)[0][0] if diagnosis_counts else 'None'
        resource_readiness = min(100, max(0, round((len(staff_members) * 60) / max(1, predicted_cases_next_month) * 100)))
        trend_label = 'Increasing' if predicted_cases_next_month >= total_consultations else 'Stable'

        actual_staff_by_role = Counter(member.role for member in staff_members)
        estimated_capacity_per_staff = 40
        estimated_monthly_capacity = len(staff_members) * estimated_capacity_per_staff
        pressure_ratio = predicted_cases_next_month / max(1, estimated_monthly_capacity)
        forecast_pressure = min(100, int(pressure_ratio * 100))
        if pressure_ratio > 1.0:
            capacity_status = 'High'
            resource_forecast_recommendation = 'Predicted demand exceeds current staffing capacity; consider hiring additional staff or expanding service hours.'
        elif pressure_ratio > 0.75:
            capacity_status = 'Moderate'
            resource_forecast_recommendation = 'Service demand is approaching capacity; monitor staffing and diagnostic equipment availability closely.'
        else:
            capacity_status = 'Healthy'
            resource_forecast_recommendation = 'Current staffing and equipment capacity appears sufficient for forecasted demand.'

        latest_month = None
        if monthly_trend:
            latest_month = monthly_trend[-1]['month']

        predictions = []
        for diagnosis, count in diagnosis_counts.most_common(3):
            predicted_volume = max(0, int(round(count * (1 + (growth_ratio if 'growth_ratio' in locals() else 0)))))
            month_label = latest_month or 'N/A'
            predictions.append({
                'diagnosis': diagnosis,
                'current_month': count,
                'predicted_next_month': predicted_volume,
                'predicted_month': month_label,
                'trend': 'Increasing' if predicted_volume >= count else 'Stable',
            })

        return {
            'total_consultations': total_consultations,
            'staff_count': len(staff_members),
            'top_diagnosis': top_diagnosis,
            'predicted_cases_next_month': predicted_cases_next_month,
            'resource_readiness': resource_readiness,
            'monthly_trend': monthly_trend,
            'consultation_distribution': diagnosis_counts.most_common(10),
            'gender_distribution': gender_counts.most_common(),
            'top_cases': diagnosis_counts.most_common(5),
            'predictions': predictions,
            'resource_recommendation': 'Consider adding one extra physician if volume rises this month.' if predicted_cases_next_month > len(staff_members) * 40 else 'Current staffing looks aligned with current demand.',
            'latest_month_label': month_label if 'month_label' in locals() else 'No data',
            'services': SERVICE_CATALOG,
            'equipment': EQUIPMENT_INVENTORY,
            'facility_staff_complement': FACILITY_STAFF_COMPLEMENT,
            'actual_staff_by_role': dict(actual_staff_by_role),
            'estimated_monthly_capacity': estimated_monthly_capacity,
            'forecast_pressure': forecast_pressure,
            'capacity_status': capacity_status,
            'resource_forecast_recommendation': resource_forecast_recommendation,
        }

    def get_dashboard_summary(force_refresh=False):
        if not force_refresh:
            summary = load_cached_dashboard_summary()
            if summary is not None:
                return summary

        records = ConsultationRecord.query.all()
        staff = StaffMember.query.all()
        summary = build_dashboard_context(records, staff)
        cache_dashboard_summary(summary)
        return summary

    @app.route('/dashboard')
    def dashboard():
        summary = get_dashboard_summary()
        return render_template(
            'dashboard/index.html',
            summary=summary,
            current_date=datetime.now().strftime('%Y-%m-%d'),
            current_time=datetime.now().strftime('%H:%M'),
        )

    @app.route('/resources')
    def resources():
        summary = get_dashboard_summary()
        return render_template(
            'resources/index.html',
            summary=summary,
            current_date=datetime.now().strftime('%Y-%m-%d'),
            current_time=datetime.now().strftime('%H:%M'),
        )

    @app.route('/records')
    def records():
        page = request.args.get('page', 1, type=int)
        records = ConsultationRecord.query.order_by(ConsultationRecord.consultation_date.desc()).paginate(page=page, per_page=10, error_out=False)
        return render_template(
            'consultations/index.html',
            records=records,
            current_date=datetime.now().strftime('%Y-%m-%d'),
            current_time=datetime.now().strftime('%H:%M'),
        )

    @app.route('/upload', methods=['GET', 'POST'])
    def upload():
        if request.method == 'POST':
            file = request.files.get('file')
            if not file:
                flash('No file selected', 'error')
                return redirect(url_for('upload'))

            filename = file.filename
            if not filename.lower().endswith(('.xlsx', '.csv')):
                flash('Invalid file type. Please upload an Excel or CSV file.', 'error')
                return redirect(url_for('upload'))

            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            try:
                df = pd.read_excel(path) if filename.lower().endswith('.xlsx') else pd.read_csv(path)
            except Exception:
                flash('Unable to read the file. Please upload a valid Excel or CSV file.', 'error')
                return redirect(url_for('upload'))

            required_columns = {'consultation_date', 'age_group', 'gender', 'diagnosis', 'department', 'physician', 'consultation_type'}
            missing_columns = required_columns - set(df.columns.str.lower())
            if missing_columns:
                flash(f'Missing required columns: {", ".join(sorted(missing_columns))}', 'error')
                return redirect(url_for('upload'))

            processed_df = clean_and_prepare_dataframe(df)
            if processed_df.empty:
                flash('No valid consultation rows were found in the uploaded file.', 'error')
                return redirect(url_for('upload'))

            ConsultationRecord.query.delete()
            for _, row in df.iterrows():
                record = ConsultationRecord(
                    consultation_date=str(row.get('consultation_date', '')),
                    age_group=str(row.get('age_group', '')),
                    gender=str(row.get('gender', '')),
                    diagnosis=str(row.get('diagnosis', '')),
                    department=str(row.get('department', '')),
                    physician=str(row.get('physician', '')),
                    consultation_type=str(row.get('consultation_type', '')),
                )
                db.session.add(record)
            db.session.commit()

            model, metrics = train_and_evaluate_model(processed_df)
            top_cases = []
            if 'diagnosis_label' in processed_df.columns:
                case_counts = processed_df['diagnosis_label'].value_counts().head(5)
                top_cases = [f"{case} ({count})" for case, count in case_counts.items()]
            report_path = write_metrics_report(metrics, top_cases)
            session['last_report'] = report_path

            summary = build_dashboard_context(ConsultationRecord.query.all(), StaffMember.query.all())
            cache_dashboard_summary(summary)

            flash('Data uploaded, cleaned, and model retrained successfully', 'success')
            return redirect(url_for('predict'))
        return render_template('consultations/upload.html')

    @app.route('/predict')
    def predict():
        records = ConsultationRecord.query.all()
        if not records:
            return render_template('forecasting/index.html', metrics={'r2_score': 0, 'mae': 0, 'mse': 0, 'rmse': 0}, top_cases=[])

        report_path = session.get('last_report')
        metrics, top_cases = read_metrics_from_report(report_path)
        return render_template('forecasting/index.html', metrics=metrics, top_cases=top_cases)

    @app.route('/retrain', methods=['POST'])
    def retrain():
        records = ConsultationRecord.query.all()
        if not records:
            flash('No consultation records available to retrain the model', 'error')
            return redirect(url_for('predict'))

        df = pd.DataFrame([
            {
                'consultation_date': record.consultation_date,
                'age_group': record.age_group,
                'gender': record.gender,
                'diagnosis': record.diagnosis,
                'department': record.department,
                'physician': record.physician,
                'consultation_type': record.consultation_type,
            }
            for record in records
        ])
        processed_df = clean_and_prepare_dataframe(df)
        _, metrics = train_and_evaluate_model(processed_df)
        top_cases = []
        if 'diagnosis_label' in processed_df.columns:
            case_counts = processed_df['diagnosis_label'].value_counts().head(3)
            top_cases = [f"{case} ({count})" for case, count in case_counts.items()]
        report_path = write_metrics_report(metrics, top_cases)
        session['last_report'] = report_path

        summary = build_dashboard_context(ConsultationRecord.query.all(), StaffMember.query.all())
        cache_dashboard_summary(summary)

        flash('Model retrained successfully using the existing consultation data', 'success')
        return redirect(url_for('predict'))

    @app.route('/staff')
    def staff():
        active_staff = StaffMember.query.filter_by(is_active=True).all()
        inactive_staff = StaffMember.query.filter_by(is_active=False).all()
        return render_template('staff/index.html', active_staff=active_staff, inactive_staff=inactive_staff)

    @app.route('/staff/new', methods=['GET', 'POST'])
    def create_staff():
        if request.method == 'POST':
            new_staff = StaffMember(
                name=request.form.get('name', '').strip(),
                role=request.form.get('role', '').strip(),
                department=request.form.get('department', '').strip(),
                availability=request.form.get('availability', 'Available').strip(),
                is_active=True,
                deleted_at=None,
            )
            db.session.add(new_staff)
            db.session.commit()
            flash('Staff member added successfully.', 'success')
            return redirect(url_for('staff'))
        return render_template('staff/form.html', staff_member=None)

    @app.route('/staff/<int:staff_id>/edit', methods=['GET', 'POST'])
    def edit_staff(staff_id):
        staff_member = StaffMember.query.get_or_404(staff_id)
        if request.method == 'POST':
            staff_member.name = request.form.get('name', staff_member.name).strip()
            staff_member.role = request.form.get('role', staff_member.role).strip()
            staff_member.department = request.form.get('department', staff_member.department).strip()
            staff_member.availability = request.form.get('availability', staff_member.availability).strip()
            db.session.commit()
            flash('Staff member updated successfully.', 'success')
            return redirect(url_for('staff'))
        return render_template('staff/form.html', staff_member=staff_member)

    @app.route('/staff/<int:staff_id>/delete', methods=['POST'])
    def delete_staff(staff_id):
        staff_member = StaffMember.query.get_or_404(staff_id)
        staff_member.is_active = False
        staff_member.deleted_at = datetime.now()
        db.session.commit()
        flash('Staff member removed from active roster (soft deleted).', 'success')
        return redirect(url_for('staff'))

    @app.route('/staff/<int:staff_id>/restore', methods=['POST'])
    def restore_staff(staff_id):
        staff_member = StaffMember.query.get_or_404(staff_id)
        staff_member.is_active = True
        staff_member.deleted_at = None
        db.session.commit()
        flash('Staff member restored to active roster.', 'success')
        return redirect(url_for('staff'))

    @app.route('/reports')
    def reports():
        return render_template('reports/index.html')

    @app.route('/settings')
    def settings():
        return render_template('settings/index.html')

    return app


def migrate_staff_member_schema(app):
    with app.app_context():
        with db.engine.connect() as conn:
            result = conn.execute(text("PRAGMA table_info(staff_member)"))
            columns = {row[1] for row in result.fetchall()}

            if 'is_active' not in columns:
                conn.execute(text("ALTER TABLE staff_member ADD COLUMN is_active BOOLEAN NOT NULL DEFAULT 1"))
            if 'deleted_at' not in columns:
                conn.execute(text("ALTER TABLE staff_member ADD COLUMN deleted_at DATETIME"))

            conn.commit()


def init_db(app=None):
    app = app or flask_app
    with app.app_context():
        db.create_all()
        migrate_staff_member_schema(app)
        if not User.query.filter_by(username='admin').first():
            db.session.add(User(username='admin', password='admin123', role='administrator'))
        if not User.query.filter_by(username='staff').first():
            db.session.add(User(username='staff', password='staff123', role='staff'))
        if not StaffMember.query.first():
            db.session.add(StaffMember(name='Dr. Ada', role='Physician', department='General Medicine', availability='Available', is_active=True, deleted_at=None))
        db.session.commit()


def prepare_monthly_data():
    records = ConsultationRecord.query.all()
    if not records:
        return pd.DataFrame(columns=['consultation_date', 'diagnosis'])
    data = [{'consultation_date': r.consultation_date, 'diagnosis': r.diagnosis} for r in records]
    return pd.DataFrame(data)


flask_app = create_app()


if __name__ == '__main__':
    init_db(flask_app)
    flask_app.run(debug=True)


app = flask_app
