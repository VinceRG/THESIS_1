import json
import os
import traceback
from collections import Counter
from datetime import datetime

import pandas as pd
import numpy as np
from flask import Flask, flash, redirect, render_template, request, session, url_for
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder

# -------------------------------------------------------------
# Constants
# -------------------------------------------------------------
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

# Resource planning constants
ROOM_COUNT = 5
AVG_CONSULTATION_MINUTES = 20
WORKING_HOURS_PER_DAY = 8
DAYS_PER_MONTH = 22
STAFF_CAPACITY_PER_MONTH = 40

# -------------------------------------------------------------
# Database setup
# -------------------------------------------------------------
db = SQLAlchemy()

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

# -------------------------------------------------------------
# Enhanced Data processing functions
# -------------------------------------------------------------
def build_training_frame(df):
    """Clean and encode the raw DataFrame."""
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

    # Encode categorical columns
    for col in ['age_group', 'gender', 'department', 'physician', 'consultation_type']:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    if 'diagnosis' in df.columns:
        df['diagnosis'] = LabelEncoder().fit_transform(df['diagnosis'].astype(str))

    return df

def build_forecasting_training_frame(df):
    """
    Build a rich time‑series frame with:
    - Multiple lags (1, 3, 6, 12 months)
    - Rolling averages (3-month)
    - Demographic proportions (using global averages as fallback)
    """
    raw = df.copy()
    for col in ['consultation_date', 'age_group', 'gender', 'diagnosis']:
        if col in raw.columns:
            raw[col] = raw[col].astype(str).str.strip()
    raw = raw.dropna(subset=['consultation_date', 'diagnosis'])
    raw['consultation_date'] = pd.to_datetime(raw['consultation_date'], errors='coerce')
    raw = raw.dropna(subset=['consultation_date'])
    raw['month'] = raw['consultation_date'].dt.month
    raw['year'] = raw['consultation_date'].dt.year

    # Group by (year, month, diagnosis) to get total case count
    grouped = (raw.groupby(['year', 'month', 'diagnosis'], as_index=False)
               .size()
               .rename(columns={'size': 'case_count'}))

    # Compute demographic proportions per (year, month, diagnosis)
    demo = raw.groupby(['year', 'month', 'diagnosis'], group_keys=False).apply(
        lambda g: pd.Series({
            'pct_adult': (g['age_group'] == 'Adult').mean(),
            'pct_child': (g['age_group'] == 'Child').mean(),
            'pct_senior': (g['age_group'] == 'Senior').mean(),
            'pct_male': (g['gender'] == 'Male').mean(),
            'pct_female': (g['gender'] == 'Female').mean()
        })
    ).reset_index()

    merged = grouped.merge(demo, on=['year', 'month', 'diagnosis'], how='left')
    
    # Instead of hardcoding 0.5, use global average for each column if missing
    for col in ['pct_adult', 'pct_child', 'pct_senior', 'pct_male', 'pct_female']:
        global_mean = merged[col].mean()
        merged[col] = merged[col].fillna(global_mean if not np.isnan(global_mean) else 0.5)

    # Sort to compute lags correctly
    merged = merged.sort_values(['diagnosis', 'year', 'month']).reset_index(drop=True)

    # Add season
    merged['season'] = (merged['month'] - 1) // 3 + 1

    # ---- Multiple Lags ----
    merged['lag_1'] = merged.groupby('diagnosis')['case_count'].shift(1)
    merged['lag_3'] = merged.groupby('diagnosis')['case_count'].shift(3)
    merged['lag_6'] = merged.groupby('diagnosis')['case_count'].shift(6)
    merged['lag_12'] = merged.groupby('diagnosis')['case_count'].shift(12)
    
    # Rolling averages (using lag_1 and lag_3)
    merged['rolling_mean_3'] = merged.groupby('diagnosis')['case_count'].transform(
        lambda x: x.rolling(3, min_periods=1).mean().shift(1)
    )

    # For rows where lag_1 is NaN (first month of a diagnosis), fill with overall median
    overall_median = merged['case_count'].median()
    for col in ['lag_1', 'lag_3', 'lag_6', 'lag_12']:
        merged[col] = merged[col].fillna(merged.groupby('diagnosis')[col].transform('median'))
        merged[col] = merged[col].fillna(overall_median)  # ultimate fallback
        
    merged['rolling_mean_3'] = merged['rolling_mean_3'].fillna(merged['lag_1'])

    # Drop rows where we still don't have a lag (shouldn't happen after fill)
    merged = merged.dropna(subset=['lag_1', 'lag_3', 'lag_6', 'lag_12', 'rolling_mean_3'])

    if merged.empty:
        return merged

    # Encode diagnosis
    le = LabelEncoder()
    merged['diagnosis'] = le.fit_transform(merged['diagnosis'])
    merged.attrs['diagnosis_encoder'] = le

    return merged

# -------------------------------------------------------------
# Enhanced Model training with Cross-Validation & Hyperparameter Tuning
# -------------------------------------------------------------
def train_and_evaluate_model(df):
    training_df = build_forecasting_training_frame(df)
    if training_df.empty:
        raise ValueError('Insufficient data for model training')

    diagnosis_encoder = training_df.attrs.get('diagnosis_encoder')
    if diagnosis_encoder is None:
        diagnosis_encoder = LabelEncoder()
        raw_diag = df['diagnosis'].astype(str).str.strip()
        diagnosis_encoder.fit(raw_diag)
    label_mapping = {i: name for i, name in enumerate(diagnosis_encoder.classes_)}

    # Expanded feature set
    feature_columns = [
        'month', 'year', 'season', 'diagnosis',
        'lag_1', 'lag_3', 'lag_6', 'lag_12', 'rolling_mean_3',
        'pct_adult', 'pct_child', 'pct_senior', 'pct_male', 'pct_female'
    ]
    missing = [col for col in feature_columns if col not in training_df.columns]
    if missing:
        raise ValueError(f'Missing columns in training frame: {missing}')

    X = training_df[feature_columns]
    y = training_df['case_count']

    # ---- Time Series Cross-Validation ----
    tscv = TimeSeriesSplit(n_splits=3)
    cv_scores = {'r2': [], 'mae': [], 'rmse': []}

    # ---- Hyperparameter Tuning with RandomizedSearchCV ----
    param_dist = {
        'n_estimators': [100, 150, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.5]
    }
    
    # We use a subset of the data for tuning to keep it fast
    tune_size = min(1000, len(X))
    X_tune = X.iloc[:tune_size]
    y_tune = y.iloc[:tune_size]
    
    rf = RandomForestRegressor(random_state=42)
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=20,
        cv=min(3, tscv.n_splits),
        scoring='r2',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    random_search.fit(X_tune, y_tune)
    best_model = random_search.best_estimator_

    # Evaluate on the full dataset using TimeSeriesSplit to get stability metrics
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        if len(X_train) < 5 or len(X_test) < 1:
            continue
            
        model = RandomForestRegressor(**random_search.best_params_, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        cv_scores['r2'].append(r2_score(y_test, preds))
        cv_scores['mae'].append(mean_absolute_error(y_test, preds))
        cv_scores['rmse'].append(np.sqrt(mean_squared_error(y_test, preds)))

    # Train the final best model on ALL data for production
    final_model = RandomForestRegressor(**random_search.best_params_, random_state=42)
    final_model.fit(X, y)
    
    # Get predictions on the full training set to compute in-sample metrics
    full_preds = final_model.predict(X)
    metrics = {
        'r2_score': round(r2_score(y, full_preds), 4),
        'mae': round(mean_absolute_error(y, full_preds), 4),
        'mse': round(mean_squared_error(y, full_preds), 4),
        'rmse': round(np.sqrt(mean_squared_error(y, full_preds)), 4),
        'cv_r2_mean': round(np.mean(cv_scores['r2']), 4) if cv_scores['r2'] else None,
        'cv_r2_std': round(np.std(cv_scores['r2']), 4) if cv_scores['r2'] else None,
        'best_params': random_search.best_params_
    }

    return final_model, metrics, feature_columns, label_mapping

# -------------------------------------------------------------
# Forecast generation with correct lag retrieval
# -------------------------------------------------------------
def generate_forecast_for_month(model, feature_columns, label_mapping, df, target_month, target_year):
    """
    Generate predictions for a specific month by correctly looking up
    lag_1, lag_3, lag_6, lag_12 from the historical data.
    """
    training_df = build_forecasting_training_frame(df)
    if training_df.empty:
        return []

    # For each diagnosis, get the row for the month just before the target
    # We need the state of the system at target_month - 1 to compute lags.
    prev_month = target_month - 1
    prev_year = target_year
    if prev_month == 0:
        prev_month = 12
        prev_year -= 1

    # We need lag_12: target_month - 12
    lag12_month = target_month - 12
    lag12_year = target_year
    if lag12_month <= 0:
        lag12_month += 12
        lag12_year -= 1

    # Get historical data for each diagnosis at the required lag periods
    # We'll build a lookup table for each diagnosis
    historical = training_df.copy()
    historical['year_month'] = historical['year'] * 100 + historical['month']

    prev_key = prev_year * 100 + prev_month
    lag3_key = prev_year * 100 + (prev_month - 2) if prev_month >= 3 else (prev_year - 1) * 100 + (prev_month + 9)
    lag6_key = prev_year * 100 + (prev_month - 5) if prev_month >= 6 else (prev_year - 1) * 100 + (prev_month + 6)
    lag12_key = lag12_year * 100 + lag12_month

    predictions = []
    diagnoses = training_df['diagnosis'].unique()

    for diag in diagnoses:
        diag_data = historical[historical['diagnosis'] == diag]
        
        # Get the most recent row for this diagnosis (to get demographics and basic info)
        latest_row = diag_data.sort_values(['year', 'month']).iloc[-1]
        
        # Get specific lag values
        lag_1_val = diag_data[diag_data['year_month'] == prev_key]['case_count'].values
        lag_3_val = diag_data[diag_data['year_month'] == lag3_key]['case_count'].values
        lag_6_val = diag_data[diag_data['year_month'] == lag6_key]['case_count'].values
        lag_12_val = diag_data[diag_data['year_month'] == lag12_key]['case_count'].values
        
        # Fallback: if specific lag not found, use the most recent available
        lag_1_val = lag_1_val[0] if len(lag_1_val) > 0 else latest_row['lag_1']
        lag_3_val = lag_3_val[0] if len(lag_3_val) > 0 else latest_row['lag_3']
        lag_6_val = lag_6_val[0] if len(lag_6_val) > 0 else latest_row['lag_6']
        lag_12_val = lag_12_val[0] if len(lag_12_val) > 0 else latest_row['lag_12']
        rolling_mean_3_val = (lag_1_val + lag_3_val) / 2  # approximation

        pred_row = {
            'month': target_month,
            'year': target_year,
            'season': (target_month - 1) // 3 + 1,
            'diagnosis': latest_row['diagnosis'],
            'lag_1': lag_1_val,
            'lag_3': lag_3_val,
            'lag_6': lag_6_val,
            'lag_12': lag_12_val,
            'rolling_mean_3': rolling_mean_3_val,
            'pct_adult': latest_row['pct_adult'],
            'pct_child': latest_row['pct_child'],
            'pct_senior': latest_row['pct_senior'],
            'pct_male': latest_row['pct_male'],
            'pct_female': latest_row['pct_female'],
        }
        predictions.append(pred_row)

    pred_df = pd.DataFrame(predictions)
    X_pred = pred_df[feature_columns]
    preds = model.predict(X_pred)

    results = []
    for i, diag_enc in enumerate(pred_df['diagnosis']):
        diag_name = label_mapping.get(diag_enc, f"Unknown_{diag_enc}")
        results.append((diag_name, max(0, round(preds[i]))))

    results.sort(key=lambda x: x[1], reverse=True)
    return results

def generate_forecast_for_specific_month(df, target_month, target_year):
    """Wrapper to generate forecast for a specific month."""
    if df.empty:
        return None, None, None
    try:
        model, metrics, feature_cols, label_mapping = train_and_evaluate_model(df)
        forecast = generate_forecast_for_month(model, feature_cols, label_mapping, df, target_month, target_year)
        total_pred = sum(count for _, count in forecast)
        return total_pred, forecast, metrics
    except Exception:
        traceback.print_exc()
        return None, None, None

# -------------------------------------------------------------
# Flask application
# -------------------------------------------------------------
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

    def build_dashboard_context(records, staff_members):
        total_consultations = len(records)

        monthly_counts = Counter()
        diagnosis_counts = Counter()
        gender_counts = Counter()
        for record in records:
            try:
                d = pd.to_datetime(record.consultation_date, errors='coerce')
                if pd.notna(d):
                    monthly_counts[d.strftime('%Y-%m')] += 1
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

        # Dynamic months
        now = datetime.now()
        predicted_month = now.month
        predicted_year = now.year
        reference_month = predicted_month
        reference_year = predicted_year - 1

        predicted_month_label = datetime(predicted_year, predicted_month, 1).strftime('%b %Y')
        reference_month_label = datetime(reference_year, reference_month, 1).strftime('%b %Y')

        # Count actual consultations for the reference month (last year)
        reference_month_counts = Counter()
        for record in records:
            d = pd.to_datetime(record.consultation_date, errors='coerce')
            if pd.notna(d) and d.year == reference_year and d.month == reference_month:
                reference_month_counts[record.diagnosis] += 1

        # Generate forecast
        if records:
            df = pd.DataFrame([{
                'consultation_date': r.consultation_date,
                'age_group': r.age_group,
                'gender': r.gender,
                'diagnosis': r.diagnosis,
                'department': r.department,
                'physician': r.physician,
                'consultation_type': r.consultation_type,
            } for r in records])
            total_pred, forecast, rf_metrics = generate_forecast_for_specific_month(df, predicted_month, predicted_year)
        else:
            total_pred, forecast, rf_metrics = None, None, None

        if total_pred is not None and total_pred > 0:
            predicted_cases_next_month = total_pred
            predictions = []
            for diag, count in forecast[:5]:
                ref_count = reference_month_counts.get(diag, 0)
                predictions.append({
                    'diagnosis': diag,
                    'current_month': ref_count,
                    'predicted_next_month': count,
                    'predicted_month': predicted_month_label,
                    'trend': 'Increasing' if count > ref_count else 'Stable'
                })
            if not predictions:
                predictions = [{
                    'diagnosis': 'No data',
                    'current_month': 0,
                    'predicted_next_month': 0,
                    'predicted_month': predicted_month_label,
                    'trend': 'Stable'
                }]
            resource_recommendation = 'Based on enhanced Random Forest with CV tuning and seasonal lags.'
        else:
            # Naive fallback
            if len(monthly_trend) >= 2:
                prev_count = monthly_trend[-2]['count']
                curr_count = monthly_trend[-1]['count']
                growth = ((curr_count - prev_count) / prev_count) if prev_count else 0
                predicted_cases_next_month = max(0, int(round(curr_count * (1 + growth))))
            else:
                predicted_cases_next_month = total_consultations

            predictions = []
            for diag, cnt in diagnosis_counts.most_common(5):
                ref_count = reference_month_counts.get(diag, 0)
                if len(monthly_trend) >= 2:
                    prev = monthly_trend[-2]['count']
                    curr = monthly_trend[-1]['count']
                    gr = ((curr - prev) / prev) if prev else 0
                    pred_cnt = max(0, int(round(cnt * (1 + gr))))
                else:
                    pred_cnt = cnt
                predictions.append({
                    'diagnosis': diag,
                    'current_month': ref_count,
                    'predicted_next_month': pred_cnt,
                    'predicted_month': predicted_month_label,
                    'trend': 'Increasing' if pred_cnt > ref_count else 'Stable'
                })
            resource_recommendation = 'Naive fallback (RF unavailable).'
            rf_metrics = None

        top_diagnosis = diagnosis_counts.most_common(1)[0][0] if diagnosis_counts else 'None'
        resource_readiness = min(100, max(0, round((len(staff_members) * 60) / max(1, predicted_cases_next_month) * 100)))

        estimated_monthly_capacity = len(staff_members) * STAFF_CAPACITY_PER_MONTH
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

        available_room_minutes = ROOM_COUNT * WORKING_HOURS_PER_DAY * 60 * DAYS_PER_MONTH
        required_room_minutes = predicted_cases_next_month * AVG_CONSULTATION_MINUTES
        room_utilization = min(100, int((required_room_minutes / max(1, available_room_minutes)) * 100))
        if room_utilization > 90:
            room_recommendation = 'Consider extending hours or adding consultation rooms.'
        elif room_utilization > 70:
            room_recommendation = 'Room utilisation is high; monitor scheduling closely.'
        else:
            room_recommendation = 'Room capacity appears adequate.'

        actual_staff_by_role = Counter(member.role for member in staff_members)

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
            'resource_recommendation': resource_recommendation,
            'reference_month': reference_month_label,
            'predicted_month_label': predicted_month_label,
            'services': SERVICE_CATALOG,
            'equipment': EQUIPMENT_INVENTORY,
            'facility_staff_complement': FACILITY_STAFF_COMPLEMENT,
            'actual_staff_by_role': dict(actual_staff_by_role),
            'estimated_monthly_capacity': estimated_monthly_capacity,
            'forecast_pressure': forecast_pressure,
            'capacity_status': capacity_status,
            'resource_forecast_recommendation': resource_forecast_recommendation,
            'room_utilization': room_utilization,
            'room_recommendation': room_recommendation,
            'rf_metrics': rf_metrics,
        }

    def get_dashboard_summary(force_refresh=False):
        if not force_refresh:
            cached = load_cached_dashboard_summary()
            if cached is not None:
                return cached

        records = ConsultationRecord.query.all()
        staff = StaffMember.query.filter_by(is_active=True).all()
        summary = build_dashboard_context(records, staff)
        cache_dashboard_summary(summary)
        return summary

    # -------------------------------------------------------------
    # Routes (unchanged except for dashboard/upload which use the new logic)
    # -------------------------------------------------------------
    @app.before_request
    def require_login():
        if request.endpoint in {'login', 'static'}:
            return None
        if request.endpoint in {'dashboard', 'records', 'upload', 'predict', 'staff', 'reports', 'settings', 'resources'} and 'user_id' not in session:
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

            records = ConsultationRecord.query.all()
            if records:
                df = pd.DataFrame([{
                    'consultation_date': r.consultation_date,
                    'age_group': r.age_group,
                    'gender': r.gender,
                    'diagnosis': r.diagnosis,
                    'department': r.department,
                    'physician': r.physician,
                    'consultation_type': r.consultation_type,
                } for r in records])
                try:
                    _, metrics, _, _ = train_and_evaluate_model(df)
                    report_path = os.path.join(app.config['UPLOAD_FOLDER'], 'training_report.txt')
                    with open(report_path, 'w', encoding='utf-8') as handle:
                        handle.write('Smart Healthcare Clinic Management - Enhanced Training Report\n')
                        handle.write('==========================================================\n')
                        handle.write(f"R2 Score: {metrics['r2_score']}\n")
                        handle.write(f"MAE: {metrics['mae']}\n")
                        handle.write(f"MSE: {metrics['mse']}\n")
                        handle.write(f"RMSE: {metrics['rmse']}\n")
                        if metrics.get('cv_r2_mean'):
                            handle.write(f"CV R2 Mean: {metrics['cv_r2_mean']} (±{metrics['cv_r2_std']})\n")
                        handle.write(f"Best Params: {metrics['best_params']}\n")
                    session['last_report'] = report_path
                except Exception as e:
                    traceback.print_exc()
                    flash(f'Model retraining failed: {str(e)}', 'error')
            else:
                flash('No records to train model.', 'warning')

            get_dashboard_summary(force_refresh=True)
            flash('Data uploaded and model retrained successfully', 'success')
            return redirect(url_for('predict'))

        return render_template('consultations/upload.html')

    @app.route('/predict')
    def predict():
        records = ConsultationRecord.query.all()
        if not records:
            flash('No consultation records found. Please upload data first.', 'warning')
            return render_template('forecasting/index.html',
                                   metrics={'r2_score': 0, 'mae': 0, 'mse': 0, 'rmse': 0},
                                   top_cases=[],
                                   forecast=[])

        df = pd.DataFrame([{
            'consultation_date': r.consultation_date,
            'age_group': r.age_group,
            'gender': r.gender,
            'diagnosis': r.diagnosis,
            'department': r.department,
            'physician': r.physician,
            'consultation_type': r.consultation_type,
        } for r in records])

        try:
            model, metrics, feature_cols, label_mapping = train_and_evaluate_model(df)
            # Predict next month after the latest data
            dates = pd.to_datetime(df['consultation_date'], errors='coerce')
            latest = dates.max()
            next_month = latest.month + 1
            next_year = latest.year
            if next_month > 12:
                next_month = 1
                next_year += 1
            forecast = generate_forecast_for_month(model, feature_cols, label_mapping, df, next_month, next_year)
            diagnosis_counts = Counter(df['diagnosis'])
            top_cases = [f"{diag} ({count})" for diag, count in diagnosis_counts.most_common(5)]
        except Exception as e:
            traceback.print_exc()
            flash(f'Could not generate forecast: {str(e)}', 'error')
            metrics = {'r2_score': 0, 'mae': 0, 'mse': 0, 'rmse': 0}
            top_cases = []
            forecast = []

        return render_template('forecasting/index.html',
                               metrics=metrics,
                               top_cases=top_cases,
                               forecast=forecast)

    @app.route('/retrain', methods=['POST'])
    def retrain():
        records = ConsultationRecord.query.all()
        if not records:
            flash('No consultation records available to retrain the model', 'error')
            return redirect(url_for('predict'))

        df = pd.DataFrame([{
            'consultation_date': r.consultation_date,
            'age_group': r.age_group,
            'gender': r.gender,
            'diagnosis': r.diagnosis,
            'department': r.department,
            'physician': r.physician,
            'consultation_type': r.consultation_type,
        } for r in records])

        try:
            _, metrics, _, _ = train_and_evaluate_model(df)
            report_path = os.path.join(app.config['UPLOAD_FOLDER'], 'training_report.txt')
            with open(report_path, 'w', encoding='utf-8') as handle:
                handle.write('Smart Healthcare Clinic Management - Enhanced Training Report\n')
                handle.write('==========================================================\n')
                handle.write(f"R2 Score: {metrics['r2_score']}\n")
                handle.write(f"MAE: {metrics['mae']}\n")
                handle.write(f"MSE: {metrics['mse']}\n")
                handle.write(f"RMSE: {metrics['rmse']}\n")
                if metrics.get('cv_r2_mean'):
                    handle.write(f"CV R2 Mean: {metrics['cv_r2_mean']} (±{metrics['cv_r2_std']})\n")
                handle.write(f"Best Params: {metrics['best_params']}\n")
            session['last_report'] = report_path
            get_dashboard_summary(force_refresh=True)
            flash('Model retrained successfully using the existing consultation data', 'success')
        except Exception as e:
            traceback.print_exc()
            flash(f'Retraining failed: {str(e)}', 'error')

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

# -------------------------------------------------------------
# Database initialisation and schema migration
# -------------------------------------------------------------
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
            db.session.add(StaffMember(name='Dr. Ada', role='Physician', department='General Medicine',
                                       availability='Available', is_active=True, deleted_at=None))
        db.session.commit()

flask_app = create_app()

if __name__ == '__main__':
    init_db(flask_app)
    flask_app.run(debug=True)

app = flask_app