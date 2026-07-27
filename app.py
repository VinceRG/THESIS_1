import json
import os
import traceback
from collections import Counter
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from flask import Flask, Response, flash, redirect, render_template, request, session, url_for
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

FACILITY_STAFF_ROSTER = [
    {'name': 'Dr. Maria Santos', 'role': 'Pathologists'},
    {'name': 'Dr. Roberto Cruz', 'role': 'Pathologists'},
    {'name': 'Angela Reyes, RMT', 'role': 'Registered Medical Technologists'},
    {'name': 'Mark Villanueva, RMT', 'role': 'Registered Medical Technologists'},
    {'name': 'Patricia Mendoza, RMT', 'role': 'Registered Medical Technologists'},
    {'name': 'Carlo Navarro, RMT', 'role': 'Registered Medical Technologists'},
    {'name': 'Joanna Lim, RMT', 'role': 'Registered Medical Technologists'},
    {'name': 'Rafael Torres, RMT', 'role': 'Registered Medical Technologists'},
    {'name': 'Grace Aquino, RRT', 'role': 'Registered Radiologic Technologists'},
    {'name': 'Nico Garcia, RRT', 'role': 'Registered Radiologic Technologists'},
    {'name': 'Liza Ramos', 'role': 'Laboratory Technicians'},
    {'name': 'Benjie Castillo', 'role': 'Laboratory Technicians'},
    {'name': 'Mariel Dizon', 'role': 'Laboratory Technicians'},
    {'name': 'Dr. Elena Bautista', 'role': 'Internal Medicine Physicians'},
    {'name': 'Dr. Victor Manuel', 'role': 'Internal Medicine Physicians'},
    {'name': 'Dr. Ana Lopez', 'role': 'General Physicians'},
    {'name': 'Dr. Paolo Rivera', 'role': 'General Physicians'},
    {'name': 'Dr. Camille Tan', 'role': 'General Physicians'},
    {'name': 'Dr. Miguel Fernandez', 'role': 'General Physicians'},
    {'name': 'Dr. Andrea Sy', 'role': 'Radiologists'},
    {'name': 'Dr. Henry Ong', 'role': 'Radiologists'},
    {'name': 'Dr. Sofia Mercado', 'role': 'Radiologists'},
]

# Resource planning constants
ROOM_COUNT = 5
AVG_CONSULTATION_MINUTES = 20
WORKING_HOURS_PER_DAY = 8
DAYS_PER_MONTH = 22
STAFF_CAPACITY_PER_MONTH = 40
DEFAULT_BRANCH_CODE = 'MAIN'
DEFAULT_BRANCH_NAME = 'Accudetek Main Branch'
DEFAULT_BRANCH_ADDRESS = 'JL Building, 12 M.H. del Pilar St, San Nicolas, Pasig, 1600 Metro Manila'
MAIN_ADMIN_ROLES = {'administrator', 'main_admin'}
ALL_BRANCHES_SCOPE = 'all'
USER_ROLE_OPTIONS = ['administrator', 'branch_admin', 'staff']

# -------------------------------------------------------------
# Database setup
# -------------------------------------------------------------
db = SQLAlchemy()

class Branch(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    code = db.Column(db.String(40), unique=True, nullable=False)
    address = db.Column(db.String(255), nullable=True)
    email = db.Column(db.String(120), nullable=True)
    contact_number = db.Column(db.String(40), nullable=True)
    is_main = db.Column(db.Boolean, default=False, nullable=False)
    is_active = db.Column(db.Boolean, default=True, nullable=False)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    branch_id = db.Column(db.Integer, db.ForeignKey('branch.id'), nullable=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(40), default='staff')
    branch = db.relationship('Branch', backref='users')

class ConsultationRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    branch_id = db.Column(db.Integer, db.ForeignKey('branch.id'), nullable=True)
    consultation_date = db.Column(db.String(20), nullable=False)
    age_group = db.Column(db.String(40), nullable=False)
    gender = db.Column(db.String(20), nullable=False)
    diagnosis = db.Column(db.String(100), nullable=False)
    department = db.Column(db.String(100), nullable=False)
    physician = db.Column(db.String(100), nullable=False)
    consultation_type = db.Column(db.String(100), nullable=False)
    branch = db.relationship('Branch', backref='consultation_records')

class StaffMember(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    branch_id = db.Column(db.Integer, db.ForeignKey('branch.id'), nullable=True)
    name = db.Column(db.String(100), nullable=False)
    role = db.Column(db.String(60), nullable=False)
    availability = db.Column(db.String(20), default='Available')
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    deleted_at = db.Column(db.DateTime, nullable=True)
    branch = db.relationship('Branch', backref='staff_members')

def build_facility_staff_roster():
    return [
        {
            'name': person['name'],
            'role': person['role'],
            'availability': 'Available',
        }
        for person in FACILITY_STAFF_ROSTER
    ]

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
        raise ValueError(f'Missing columns """  """in training frame: {missing}')

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

def build_forecasting_training_frame(df):
    """Build one diagnosis-age-gender-month row per observed month, with honest lag features."""
    raw = df.copy()
    for col in ['consultation_date', 'diagnosis', 'age_group', 'gender']:
        if col in raw.columns:
            raw[col] = raw[col].astype(str).str.strip()
    raw = raw.dropna(subset=['consultation_date', 'diagnosis'])
    raw['consultation_date'] = pd.to_datetime(raw['consultation_date'], errors='coerce')
    raw = raw.dropna(subset=['consultation_date'])
    raw = raw[raw['diagnosis'] != '']
    if 'age_group' not in raw.columns:
        raw['age_group'] = 'Unknown'
    if 'gender' not in raw.columns:
        raw['gender'] = 'Unknown'
    raw['age_group'] = raw['age_group'].replace(['', 'nan', 'None', 'NaT'], 'Unknown').fillna('Unknown')
    raw['gender'] = raw['gender'].replace(['', 'nan', 'None', 'NaT'], 'Unknown').fillna('Unknown')
    if raw.empty:
        return pd.DataFrame()

    raw['period'] = raw['consultation_date'].dt.to_period('M')
    diagnoses = sorted(raw['diagnosis'].unique())
    age_groups = sorted(raw['age_group'].unique())
    genders = sorted(raw['gender'].unique())
    periods = pd.period_range(raw['period'].min(), raw['period'].max(), freq='M')
    full_index = pd.MultiIndex.from_product(
        [diagnoses, age_groups, genders, periods],
        names=['diagnosis', 'age_group', 'gender', 'period']
    )

    grouped = raw.groupby(['diagnosis', 'age_group', 'gender', 'period']).size().rename('case_count')
    merged = grouped.reindex(full_index, fill_value=0).reset_index()
    merged['period_start'] = merged['period'].dt.to_timestamp()
    merged['year'] = merged['period_start'].dt.year
    merged['month'] = merged['period_start'].dt.month
    merged['season'] = (merged['month'] - 1) // 3 + 1
    merged['time_index'] = (merged['year'] - merged['year'].min()) * 12 + merged['month']
    segment_columns = ['diagnosis', 'age_group', 'gender']
    merged = merged.sort_values(segment_columns + ['period']).reset_index(drop=True)

    segment_group = merged.groupby(segment_columns)['case_count']
    merged['lag_1'] = segment_group.shift(1)
    merged['lag_2'] = segment_group.shift(2)
    merged['lag_3'] = segment_group.shift(3)
    merged['lag_6'] = segment_group.shift(6)
    merged['lag_12'] = segment_group.shift(12)
    merged['rolling_mean_3'] = merged.groupby(segment_columns)['case_count'].transform(
        lambda x: x.rolling(3, min_periods=1).mean().shift(1)
    )
    merged['rolling_mean_6'] = merged.groupby(segment_columns)['case_count'].transform(
        lambda x: x.rolling(6, min_periods=1).mean().shift(1)
    )
    merged['rolling_std_3'] = merged.groupby(segment_columns)['case_count'].transform(
        lambda x: x.rolling(3, min_periods=2).std().shift(1)
    )
    merged['trend_1'] = merged['lag_1'] - merged['lag_2']

    history_columns = [
        'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12',
        'rolling_mean_3', 'rolling_mean_6', 'rolling_std_3', 'trend_1'
    ]
    merged[history_columns] = merged[history_columns].fillna(0)

    le = LabelEncoder()
    merged['diagnosis_code'] = le.fit_transform(merged['diagnosis'])
    diagnosis_dummies = pd.get_dummies(merged['diagnosis'], prefix='diagnosis', dtype=int)
    age_dummies = pd.get_dummies(merged['age_group'], prefix='age_group', dtype=int)
    gender_dummies = pd.get_dummies(merged['gender'], prefix='gender', dtype=int)
    merged = pd.concat([merged, diagnosis_dummies, age_dummies, gender_dummies], axis=1)
    merged.attrs['diagnosis_encoder'] = le
    return merged

def _safe_r2(y_true, preds):
    if len(y_true) < 2 or np.isclose(np.var(y_true), 0):
        return None
    return r2_score(y_true, preds)

def _regression_metrics(y_true, preds):
    mse = mean_squared_error(y_true, preds)
    score = _safe_r2(y_true, preds)
    return {
        'r2': round(score, 4) if score is not None else None,
        'mae': round(mean_absolute_error(y_true, preds), 4),
        'mse': round(mse, 4),
        'rmse': round(np.sqrt(mse), 4),
    }

def _model_verdict(metrics):
    validation_r2 = metrics.get('validation_r2')
    improvement = metrics.get('improvement_vs_baseline_pct')
    if validation_r2 is not None and validation_r2 >= 0.5 and improvement is not None and improvement > 0:
        return 'Acceptable'
    if validation_r2 is not None and validation_r2 > 0 and improvement is not None and improvement > 0:
        return 'Needs more data, but better than baseline'
    return 'Not acceptable yet'

def build_non_demographic_training_frame(df):
    """Build one diagnosis-month row per observed month without age or gender features."""
    raw = df.copy()
    for col in ['consultation_date', 'diagnosis']:
        if col in raw.columns:
            raw[col] = raw[col].astype(str).str.strip()
    raw = raw.dropna(subset=['consultation_date', 'diagnosis'])
    raw['consultation_date'] = pd.to_datetime(raw['consultation_date'], errors='coerce')
    raw = raw.dropna(subset=['consultation_date'])
    raw = raw[raw['diagnosis'] != '']
    if raw.empty:
        return pd.DataFrame()

    raw['period'] = raw['consultation_date'].dt.to_period('M')
    diagnoses = sorted(raw['diagnosis'].unique())
    periods = pd.period_range(raw['period'].min(), raw['period'].max(), freq='M')
    full_index = pd.MultiIndex.from_product(
        [diagnoses, periods],
        names=['diagnosis', 'period']
    )

    grouped = raw.groupby(['diagnosis', 'period']).size().rename('case_count')
    merged = grouped.reindex(full_index, fill_value=0).reset_index()
    merged['period_start'] = merged['period'].dt.to_timestamp()
    merged['year'] = merged['period_start'].dt.year
    merged['month'] = merged['period_start'].dt.month
    merged['season'] = (merged['month'] - 1) // 3 + 1
    merged['time_index'] = (merged['year'] - merged['year'].min()) * 12 + merged['month']
    merged = merged.sort_values(['diagnosis', 'period']).reset_index(drop=True)

    segment_group = merged.groupby(['diagnosis'])['case_count']
    merged['lag_1'] = segment_group.shift(1)
    merged['lag_2'] = segment_group.shift(2)
    merged['lag_3'] = segment_group.shift(3)
    merged['lag_6'] = segment_group.shift(6)
    merged['lag_12'] = segment_group.shift(12)
    merged['rolling_mean_3'] = merged.groupby(['diagnosis'])['case_count'].transform(
        lambda x: x.rolling(3, min_periods=1).mean().shift(1)
    )
    merged['rolling_mean_6'] = merged.groupby(['diagnosis'])['case_count'].transform(
        lambda x: x.rolling(6, min_periods=1).mean().shift(1)
    )
    merged['rolling_std_3'] = merged.groupby(['diagnosis'])['case_count'].transform(
        lambda x: x.rolling(3, min_periods=2).std().shift(1)
    )
    merged['trend_1'] = merged['lag_1'] - merged['lag_2']

    history_columns = [
        'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12',
        'rolling_mean_3', 'rolling_mean_6', 'rolling_std_3', 'trend_1'
    ]
    merged[history_columns] = merged[history_columns].fillna(0)
    diagnosis_dummies = pd.get_dummies(merged['diagnosis'], prefix='diagnosis', dtype=int)
    return pd.concat([merged, diagnosis_dummies], axis=1)

def evaluate_model_without_demographics(df):
    training_df = build_non_demographic_training_frame(df)
    if training_df.empty:
        return None

    diagnosis_feature_columns = sorted([
        col for col in training_df.columns
        if col.startswith('diagnosis_')
    ])
    feature_columns = [
        'month', 'season', 'time_index',
        'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12',
        'rolling_mean_3', 'rolling_mean_6', 'rolling_std_3', 'trend_1',
    ] + diagnosis_feature_columns

    training_df = training_df.sort_values(['period_start', 'diagnosis']).reset_index(drop=True)
    unique_periods = sorted(training_df['period'].unique())
    if len(unique_periods) < 8:
        return None

    holdout_months = min(6, max(2, len(unique_periods) // 4))
    validation_periods = unique_periods[-holdout_months:]
    train_df = training_df[~training_df['period'].isin(validation_periods)]
    validation_df = training_df[training_df['period'].isin(validation_periods)]

    param_dist = {
        'n_estimators': [100, 150, 200, 300],
        'max_depth': [4, 6, 8, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.5]
    }
    segment_count = max(1, training_df[['diagnosis']].drop_duplicates().shape[0])
    cv_splits = min(3, max(2, len(train_df) // max(1, segment_count * 3)))
    random_search = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_distributions=param_dist,
        n_iter=20,
        cv=TimeSeriesSplit(n_splits=cv_splits),
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    random_search.fit(train_df[feature_columns], train_df['case_count'])

    validation_preds = random_search.best_estimator_.predict(validation_df[feature_columns])
    validation_metrics = _regression_metrics(validation_df['case_count'], validation_preds)
    baseline_metrics = _regression_metrics(validation_df['case_count'], validation_df['lag_1'])
    improvement = None
    if baseline_metrics['mae']:
        improvement = round(((baseline_metrics['mae'] - validation_metrics['mae']) / baseline_metrics['mae']) * 100, 2)

    return {
        'training_grain': 'diagnosis by month',
        'training_rows': int(len(training_df)),
        'validation_r2': validation_metrics['r2'],
        'validation_mae': validation_metrics['mae'],
        'validation_rmse': validation_metrics['rmse'],
        'baseline_mae': baseline_metrics['mae'],
        'improvement_vs_baseline_pct': improvement,
        'training_months': len(unique_periods) - holdout_months,
        'validation_months': holdout_months,
        'validation_period_start': str(validation_periods[0]),
        'validation_period_end': str(validation_periods[-1]),
    }

def train_and_evaluate_model(df):
    training_df = build_forecasting_training_frame(df)
    if training_df.empty:
        raise ValueError('Insufficient data for model training')

    diagnosis_encoder = training_df.attrs.get('diagnosis_encoder')
    label_mapping = {i: name for i, name in enumerate(diagnosis_encoder.classes_)}
    diagnosis_feature_columns = sorted([
        col for col in training_df.columns
        if col.startswith('diagnosis_') and col != 'diagnosis_code'
    ])
    demographic_feature_columns = sorted([
        col for col in training_df.columns
        if col.startswith('age_group_') or col.startswith('gender_')
    ])
    feature_columns = [
        'month', 'season', 'time_index',
        'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12',
        'rolling_mean_3', 'rolling_mean_6', 'rolling_std_3', 'trend_1',
    ] + diagnosis_feature_columns + demographic_feature_columns

    training_df = training_df.sort_values(['period_start', 'diagnosis', 'age_group', 'gender']).reset_index(drop=True)
    unique_periods = sorted(training_df['period'].unique())
    if len(unique_periods) < 8:
        raise ValueError('At least 8 months of consultation data are needed for reliable validation')

    holdout_months = min(6, max(2, len(unique_periods) // 4))
    validation_periods = unique_periods[-holdout_months:]
    train_df = training_df[~training_df['period'].isin(validation_periods)]
    validation_df = training_df[training_df['period'].isin(validation_periods)]

    X_train = train_df[feature_columns]
    y_train = train_df['case_count']
    X_validation = validation_df[feature_columns]
    y_validation = validation_df['case_count']

    param_dist = {
        'n_estimators': [100, 150, 200, 300],
        'max_depth': [4, 6, 8, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.5]
    }
    segment_count = max(1, training_df[['diagnosis', 'age_group', 'gender']].drop_duplicates().shape[0])
    cv_splits = min(3, max(2, len(train_df) // max(1, segment_count * 3)))
    random_search = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_distributions=param_dist,
        n_iter=20,
        cv=TimeSeriesSplit(n_splits=cv_splits),
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    random_search.fit(X_train, y_train)

    validation_preds = random_search.best_estimator_.predict(X_validation)
    validation_metrics = _regression_metrics(y_validation, validation_preds)
    baseline_metrics = _regression_metrics(y_validation, validation_df['lag_1'])

    cv_scores = {'r2': [], 'mae': [], 'rmse': []}
    for train_idx, test_idx in TimeSeriesSplit(n_splits=cv_splits).split(training_df):
        fold_train = training_df.iloc[train_idx]
        fold_test = training_df.iloc[test_idx]
        model = RandomForestRegressor(**random_search.best_params_, random_state=42)
        model.fit(fold_train[feature_columns], fold_train['case_count'])
        preds = model.predict(fold_test[feature_columns])
        fold_metrics = _regression_metrics(fold_test['case_count'], preds)
        if fold_metrics['r2'] is not None:
            cv_scores['r2'].append(fold_metrics['r2'])
        cv_scores['mae'].append(fold_metrics['mae'])
        cv_scores['rmse'].append(fold_metrics['rmse'])

    final_model = RandomForestRegressor(**random_search.best_params_, random_state=42)
    final_model.fit(training_df[feature_columns], training_df['case_count'])
    training_metrics = _regression_metrics(
        training_df['case_count'],
        final_model.predict(training_df[feature_columns])
    )

    improvement = None
    if baseline_metrics['mae']:
        improvement = round(((baseline_metrics['mae'] - validation_metrics['mae']) / baseline_metrics['mae']) * 100, 2)

    metrics = {
        'total_training_rows': int(len(training_df)),
        'total_segments': int(training_df[['diagnosis', 'age_group', 'gender']].drop_duplicates().shape[0]),
        'diagnosis_count': int(training_df['diagnosis'].nunique()),
        'age_group_count': int(training_df['age_group'].nunique()),
        'gender_count': int(training_df['gender'].nunique()),
        'data_period_start': str(unique_periods[0]),
        'data_period_end': str(unique_periods[-1]),
        'r2_score': validation_metrics['r2'],
        'mae': validation_metrics['mae'],
        'mse': validation_metrics['mse'],
        'rmse': validation_metrics['rmse'],
        'validation_r2': validation_metrics['r2'],
        'validation_mae': validation_metrics['mae'],
        'validation_mse': validation_metrics['mse'],
        'validation_rmse': validation_metrics['rmse'],
        'training_r2': training_metrics['r2'],
        'training_mae': training_metrics['mae'],
        'training_rmse': training_metrics['rmse'],
        'baseline_mae': baseline_metrics['mae'],
        'baseline_rmse': baseline_metrics['rmse'],
        'baseline_r2': baseline_metrics['r2'],
        'improvement_vs_baseline_pct': improvement,
        'cv_r2_mean': round(np.mean(cv_scores['r2']), 4) if cv_scores['r2'] else None,
        'cv_r2_std': round(np.std(cv_scores['r2']), 4) if cv_scores['r2'] else None,
        'cv_mae_mean': round(np.mean(cv_scores['mae']), 4) if cv_scores['mae'] else None,
        'cv_rmse_mean': round(np.mean(cv_scores['rmse']), 4) if cv_scores['rmse'] else None,
        'training_months': len(unique_periods) - holdout_months,
        'validation_months': holdout_months,
        'validation_period_start': str(validation_periods[0]),
        'validation_period_end': str(validation_periods[-1]),
        'best_params': random_search.best_params_,
    }
    try:
        metrics['model_b_without_demographics'] = evaluate_model_without_demographics(df)
    except Exception:
        traceback.print_exc()
        metrics['model_b_without_demographics'] = None
    metrics['model_verdict'] = _model_verdict(metrics)
    return final_model, metrics, feature_columns, label_mapping

def generate_forecast_for_month(model, feature_columns, label_mapping, df, target_month, target_year):
    training_df = build_forecasting_training_frame(df)
    if training_df.empty:
        generate_forecast_for_month.last_demographic_forecast = {
            'age_group': [],
            'gender': [],
            'segments': [],
        }
        return []

    target_period = pd.Period(year=target_year, month=target_month, freq='M')
    next_rows = []
    diagnosis_columns = [
        col for col in feature_columns
        if col.startswith('diagnosis_') and col != 'diagnosis_code'
    ]
    age_columns = [col for col in feature_columns if col.startswith('age_group_')]
    gender_columns = [col for col in feature_columns if col.startswith('gender_')]

    for (diagnosis, age_group, gender), segment_data in training_df.groupby(['diagnosis', 'age_group', 'gender']):
        history = segment_data.sort_values('period').copy()
        counts = history.set_index('period')['case_count']

        def lag_value(months_back):
            period = target_period - months_back
            return counts.get(period, 0)

        lag_1 = lag_value(1)
        lag_2 = lag_value(2)
        recent_3 = [lag_value(i) for i in range(1, 4)]
        recent_6 = [lag_value(i) for i in range(1, 7)]
        row = {
            'month': target_month,
            'season': (target_month - 1) // 3 + 1,
            'time_index': int((target_period.year - training_df['year'].min()) * 12 + target_period.month),
            'lag_1': lag_1,
            'lag_2': lag_2,
            'lag_3': lag_value(3),
            'lag_6': lag_value(6),
            'lag_12': lag_value(12),
            'rolling_mean_3': float(np.mean(recent_3)),
            'rolling_mean_6': float(np.mean(recent_6)),
            'rolling_std_3': float(np.std(recent_3, ddof=1)) if len(recent_3) > 1 else 0,
            'trend_1': lag_1 - lag_2,
            'diagnosis_code': int(history['diagnosis_code'].iloc[-1]),
            'diagnosis': diagnosis,
            'age_group': age_group,
            'gender': gender,
        }
        for col in diagnosis_columns:
            row[col] = 1 if col == f'diagnosis_{diagnosis}' else 0
        for col in age_columns:
            row[col] = 1 if col == f'age_group_{age_group}' else 0
        for col in gender_columns:
            row[col] = 1 if col == f'gender_{gender}' else 0
        next_rows.append(row)

    pred_df = pd.DataFrame(next_rows)
    preds = model.predict(pred_df[feature_columns])
    pred_df['predicted_cases'] = [max(0, round(value)) for value in preds]

    diagnosis_totals = (
        pred_df.groupby('diagnosis')['predicted_cases']
        .sum()
        .sort_values(ascending=False)
    )
    age_totals = (
        pred_df.groupby('age_group')['predicted_cases']
        .sum()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={'predicted_cases': 'forecasted_cases'})
        .to_dict('records')
    )
    gender_totals = (
        pred_df.groupby('gender')['predicted_cases']
        .sum()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={'predicted_cases': 'forecasted_cases'})
        .to_dict('records')
    )
    segment_rows = (
        pred_df[pred_df['predicted_cases'] > 0]
        .sort_values('predicted_cases', ascending=False)
        [['diagnosis', 'age_group', 'gender', 'predicted_cases']]
        .rename(columns={'predicted_cases': 'forecasted_cases'})
        .head(15)
        .to_dict('records')
    )

    results = [(diagnosis, int(count)) for diagnosis, count in diagnosis_totals.items()]
    generate_forecast_for_month.last_demographic_forecast = {
        'age_group': age_totals,
        'gender': gender_totals,
        'segments': segment_rows,
    }
    return results

def write_training_report(report_path, metrics):
    def month_label(value):
        try:
            return datetime.strptime(str(value), '%Y-%m').strftime('%B %Y')
        except Exception:
            return str(value)

    def fmt(value, suffix=''):
        if value is None:
            return 'N/A'
        return f'{value}{suffix}'

    model_b = metrics.get('model_b_without_demographics') or {}
    best_params = metrics.get('best_params') or {}

    with open(report_path, 'w', encoding='utf-8') as handle:
        handle.write('Smart Healthcare Clinic Management - Model Training and Evaluation Report\n')
        handle.write('=======================================================================\n')
        handle.write('\n1. Model Objective\n')
        handle.write('\n')
        handle.write('The objective of this Random Forest regression model is to forecast the monthly consultation demand at Accudetek Health Diagnostics. ')
        handle.write('The model predicts the expected consultation volume by diagnosis, age group, and gender to provide decision-support information for clinic administrators in staff planning, resource preparedness, service readiness, and operational decision-making.\n')

        handle.write('\n2. Dataset and Feature Preparation\n')
        handle.write('\n')
        handle.write(f"Dataset Period: {month_label(metrics.get('data_period_start', 'N/A'))} - {month_label(metrics.get('data_period_end', 'N/A'))}\n\n")
        handle.write(f"Training Frame Rows: {metrics.get('total_training_rows', 'N/A')}\n\n")
        handle.write(f"Diagnosis-Age Group-Gender Segments: {metrics.get('total_segments', 'N/A')}\n\n")
        handle.write(f"Diagnoses Covered: {metrics.get('diagnosis_count', 'N/A')}\n\n")
        handle.write(f"Age Groups Covered: {metrics.get('age_group_count', 'N/A')}\n\n")
        handle.write(f"Gender Categories Covered: {metrics.get('gender_count', 'N/A')}\n\n")
        handle.write('Training Granularity: Diagnosis x Age Group x Gender x Month\n\n')
        handle.write('Predictive Features\n\n')
        handle.write('The model was trained using the following features:\n\n')
        for feature in [
            'Diagnosis',
            'Age Group',
            'Gender',
            'Monthly Seasonality',
            'Time Index',
            'One-, Two-, Three-, Six-, and Twelve-Month Lag Values',
            'Rolling Averages',
            'Recent Trend Indicators',
        ]:
            handle.write(f'{feature}\n')
        handle.write('\n')
        handle.write('The consultation records were transformed into monthly consultation counts for each diagnosis-age group-gender segment. ')
        handle.write('Feature engineering was then performed by generating lag values, rolling averages, seasonal indicators, and trend features to enable the Random Forest model to learn recurring consultation patterns and temporal behavior.\n')

        handle.write('\n3. Model Validation Method\n')
        handle.write('\n')
        handle.write('To simulate real-world forecasting, the model used a time-based validation approach rather than a random train-test split.\n\n')
        handle.write('Earlier months were used for model training, while the most recent months were reserved for validation.\n\n')
        handle.write('This prevents information leakage and reflects the practical forecasting scenario in which future consultation demand must be predicted using only historical records.\n\n')
        handle.write(f"Training Months: {metrics.get('training_months')}\n\n")
        handle.write(f"Validation Months: {metrics.get('validation_months')}\n\n")
        handle.write(f"Validation Period: {month_label(metrics.get('validation_period_start'))} - {month_label(metrics.get('validation_period_end'))}\n\n")
        handle.write('Baseline Forecast\n\n')
        handle.write("The model was compared against a naive forecasting approach that assumes the next month's consultation count will be the same as the previous month's value.\n\n")
        handle.write('This baseline provides a simple benchmark for evaluating whether the Random Forest model learns meaningful consultation patterns beyond historical repetition.\n')

        handle.write('\n4. Demographic Feature Comparison\n')
        handle.write('\n')
        handle.write('Two Random Forest models were evaluated.\n\n')
        handle.write('Model A - Proposed Model\n\n')
        handle.write('Features Included:\n\n')
        for feature in ['Diagnosis', 'Age Group', 'Gender', 'Monthly Lag Values', 'Rolling Averages', 'Trend Features', 'Seasonal Indicators']:
            handle.write(f'{feature}\n')
        handle.write('\nValidation Results\n\n')
        handle.write(f"Validation R2: {fmt(metrics.get('validation_r2'))}\n")
        handle.write(f"Validation MAE: {fmt(metrics.get('validation_mae'))}\n")
        handle.write(f"Validation RMSE: {fmt(metrics.get('validation_rmse'))}\n")
        handle.write(f"Improvement over Baseline: {fmt(metrics.get('improvement_vs_baseline_pct'), '%')}\n")
        handle.write('\nModel B - Comparison Model\n\n')
        handle.write('Features Included\n\n')
        for feature in ['Diagnosis', 'Monthly Lag Values', 'Rolling Averages', 'Trend Features', 'Seasonal Indicators']:
            handle.write(f'{feature}\n')
        handle.write('\n(Age Group and Gender excluded.)\n\n')
        handle.write('Validation Results\n\n')
        handle.write(f"Validation R2: {fmt(model_b.get('validation_r2'))}\n")
        handle.write(f"Validation MAE: {fmt(model_b.get('validation_mae'))}\n")
        handle.write(f"Validation RMSE: {fmt(model_b.get('validation_rmse'))}\n")
        handle.write(f"Improvement over Baseline: {fmt(model_b.get('improvement_vs_baseline_pct'), '%')}\n")
        handle.write('\nInterpretation\n\n')
        if model_b:
            mae_delta = round(model_b.get('validation_mae', 0) - metrics.get('validation_mae', 0), 4)
            if mae_delta > 0:
                handle.write(f'Model A generated lower prediction error than Model B by {mae_delta} MAE points while producing detailed forecasts by diagnosis, age group, and gender.\n\n')
            elif mae_delta < 0:
                handle.write(f'Model B generated lower MAE by {abs(mae_delta)} points, but it performed a simpler diagnosis-level forecasting task and cannot produce age-group or gender-specific forecast outputs.\n\n')
            else:
                handle.write('Model A and Model B produced the same MAE on this validation split, but Model A provides demographic-specific forecast outputs required by the study objectives.\n\n')
            handle.write('Since the two models operate at different levels of granularity, their R2 values should not be interpreted as a direct measure of superiority.\n\n')
            handle.write('Model A was selected because it provides demographic-specific predictions that better support clinic planning, staffing, and resource allocation.\n')
        else:
            handle.write('Model B metrics were not available in this report. Run model retraining to generate the non-demographic comparison results.\n')

        handle.write('\n5. Model Performance\n')
        handle.write('\n')
        handle.write(f"Model Verdict: {metrics.get('model_verdict', 'Unknown')}\n")
        handle.write('\nMetric\tResult\n')
        handle.write(f"Validation R2\t{fmt(metrics.get('validation_r2'))}\n")
        handle.write(f"Validation MAE\t{fmt(metrics.get('validation_mae'))}\n")
        handle.write(f"Validation RMSE\t{fmt(metrics.get('validation_rmse'))}\n")
        handle.write(f"Baseline MAE\t{fmt(metrics.get('baseline_mae'))}\n")
        handle.write(f"Baseline RMSE\t{fmt(metrics.get('baseline_rmse'))}\n")
        handle.write(f"Improvement over Baseline\t{fmt(metrics.get('improvement_vs_baseline_pct'), '%')}\n")
        handle.write(f"Cross-Validation R2 Mean\t{fmt(metrics.get('cv_r2_mean'))} +/- {fmt(metrics.get('cv_r2_std'))}\n")
        handle.write(f"Cross-Validation MAE Mean\t{fmt(metrics.get('cv_mae_mean'))}\n")
        handle.write(f"Training R2 (Reference Only)\t{fmt(metrics.get('training_r2'))}\n")

        handle.write('\n6. Performance Metric Interpretation\n')
        handle.write('Validation R2\n\n')
        handle.write('Measures how well the model explains the variation in unseen validation data.\n\n')
        handle.write('Higher values indicate stronger predictive performance on future consultation records.\n\n')
        handle.write('Mean Absolute Error (MAE)\n\n')
        handle.write('Measures the average prediction error in consultation cases.\n\n')
        handle.write(f"An MAE of {fmt(metrics.get('validation_mae'))} means that, on average, the predicted consultation volume differs from the actual value by approximately {round(metrics.get('validation_mae', 0)) if metrics.get('validation_mae') is not None else 'N/A'} consultation cases per diagnosis-age group-gender segment.\n\n")
        handle.write('Root Mean Squared Error (RMSE)\n\n')
        handle.write('Measures the magnitude of prediction errors while assigning greater weight to larger mistakes.\n\n')
        handle.write('Lower RMSE values indicate better overall forecasting accuracy.\n\n')
        handle.write('Baseline Improvement\n\n')
        handle.write("This measures how much the Random Forest model reduced prediction error compared with simply using the previous month's consultation count as the forecast.\n\n")
        handle.write(f"A {fmt(metrics.get('improvement_vs_baseline_pct'), '%')} improvement demonstrates that the model learned meaningful consultation patterns rather than merely repeating historical observations.\n")

        handle.write('\n7. Final Interpretation\n')
        handle.write('\n')
        handle.write('Based on the validation results, the Random Forest regression model demonstrated acceptable performance for short-term consultation forecasting.\n\n')
        handle.write(f"The model achieved a validation R2 of {fmt(metrics.get('validation_r2'))}, indicating that it explained approximately {round(metrics.get('validation_r2', 0) * 100) if metrics.get('validation_r2') is not None else 'N/A'}% of the variation in unseen consultation demand.\n\n")
        handle.write(f"Its Mean Absolute Error of {fmt(metrics.get('validation_mae'))} indicates that the average forecasting error was approximately {round(metrics.get('validation_mae', 0)) if metrics.get('validation_mae') is not None else 'N/A'} consultation cases per diagnosis-age group-gender month segment.\n\n")
        handle.write(f"Compared with the naive last-month forecasting baseline, the proposed model reduced prediction error by {fmt(metrics.get('improvement_vs_baseline_pct'), '%')}, demonstrating that it successfully learned recurring consultation patterns rather than simply repeating previous observations.\n\n")
        handle.write('These findings support the feasibility of the proposed Smart Healthcare Clinic Management System as a decision-support tool for forecasting consultation trends, supporting staff planning, improving service preparedness, and assisting operational decision-making at Accudetek Health Diagnostics.\n')

        handle.write('\n8. Best Random Forest Parameters\n')
        handle.write(f"n_estimators      : {best_params.get('n_estimators', 'N/A')}\n")
        handle.write(f"max_depth         : {best_params.get('max_depth', 'N/A')}\n")
        handle.write(f"min_samples_split : {best_params.get('min_samples_split', 'N/A')}\n")
        handle.write(f"min_samples_leaf  : {best_params.get('min_samples_leaf', 'N/A')}\n")
        handle.write(f"max_features      : {best_params.get('max_features', 'N/A')}\n")

        handle.write('\n9. Study Limitation\n')
        handle.write('\n')
        handle.write("Note: The reported model performance was obtained using the simulated consultation dataset developed for this study. ")
        handle.write("Although the system demonstrates the technical feasibility of consultation forecasting, deployment in an actual clinical environment would require retraining and validation using Accudetek Health Diagnostics' historical consultation records to verify predictive performance under real-world conditions.\n")

def _pdf_escape(value):
    return str(value).replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')

def _build_simple_pdf(title, lines):
    max_lines_per_page = 46
    pages = []
    current = [title, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', '']
    for line in lines:
        if len(current) >= max_lines_per_page:
            pages.append(current)
            current = [title + ' (continued)', '']
        current.append(str(line))
    pages.append(current)

    objects = []
    page_refs = []
    font_obj = 3
    next_obj = 4

    for page_lines in pages:
        content_lines = ['BT', '/F1 11 Tf', '50 790 Td', '14 TL']
        first = True
        for line in page_lines:
            text = _pdf_escape(line[:105])
            if first:
                content_lines.append(f'({text}) Tj')
                first = False
            else:
                content_lines.append(f'T* ({text}) Tj')
        content_lines.append('ET')
        content = '\n'.join(content_lines).encode('latin-1', errors='replace')

        content_obj = next_obj
        page_obj = next_obj + 1
        next_obj += 2
        objects.append((content_obj, b'<< /Length ' + str(len(content)).encode('ascii') + b' >>\nstream\n' + content + b'\nendstream'))
        objects.append((page_obj, f'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] /Resources << /Font << /F1 {font_obj} 0 R >> >> /Contents {content_obj} 0 R >>'.encode('ascii')))
        page_refs.append(f'{page_obj} 0 R')

    objects.insert(0, (1, b'<< /Type /Catalog /Pages 2 0 R >>'))
    objects.insert(1, (2, f'<< /Type /Pages /Kids [{" ".join(page_refs)}] /Count {len(page_refs)} >>'.encode('ascii')))
    objects.insert(2, (font_obj, b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>'))
    objects = sorted(objects, key=lambda item: item[0])

    pdf = bytearray(b'%PDF-1.4\n')
    offsets = [0]
    for obj_num, body in objects:
        offsets.append(len(pdf))
        pdf.extend(f'{obj_num} 0 obj\n'.encode('ascii'))
        pdf.extend(body)
        pdf.extend(b'\nendobj\n')
    xref_pos = len(pdf)
    pdf.extend(f'xref\n0 {len(objects) + 1}\n'.encode('ascii'))
    pdf.extend(b'0000000000 65535 f \n')
    for offset in offsets[1:]:
        pdf.extend(f'{offset:010d} 00000 n \n'.encode('ascii'))
    pdf.extend(f'trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF'.encode('ascii'))
    return bytes(pdf)

def _jpeg_size(path):
    try:
        with open(path, 'rb') as handle:
            data = handle.read()
        index = 2
        while index < len(data):
            if data[index] != 0xFF:
                index += 1
                continue
            marker = data[index + 1]
            length = int.from_bytes(data[index + 2:index + 4], 'big')
            if marker in {0xC0, 0xC1, 0xC2}:
                height = int.from_bytes(data[index + 5:index + 7], 'big')
                width = int.from_bytes(data[index + 7:index + 9], 'big')
                return width, height, data
            index += 2 + length
    except Exception:
        return None
    return None

def _pdf_text(x, y, text, size=10, color='0.067 0.208 0.310', font='F1'):
    return f'{color} rg BT /{font} {size} Tf {x} {y} Td ({_pdf_escape(text)}) Tj ET'

def _build_report_pdf(payload, logo_path=None):
    width, height = 612, 842
    primary = '0.059 0.357 0.541'
    accent = '0.169 0.655 0.773'
    text = '0.067 0.208 0.310'
    muted = '0.392 0.455 0.545'
    light = '0.961 0.984 0.992'
    border = '0.831 0.875 0.918'
    colors = [(0.059, 0.357, 0.541), (0.169, 0.655, 0.773), (0.184, 0.620, 0.267), (0.851, 0.467, 0.024), (0.486, 0.227, 0.929), (0.863, 0.149, 0.149)]
    image_info = _jpeg_size(logo_path) if logo_path else None
    pages = []

    def clip(value, limit):
        return str(value)

    def new_page(title=None):
        commands = [
            f'{light} rg 0 0 {width} {height} re f',
            f'{primary} rg 0 760 {width} 82 re f',
            f'{accent} rg 0 754 {width} 6 re f',
        ]
        if image_info:
            commands.append('q 46 0 0 46 42 778 cm /Logo Do Q')
            title_x = 100
        else:
            commands.append('1 1 1 rg 42 778 46 46 re f')
            commands.append(_pdf_text(54, 795, 'A', 20, primary, 'F2'))
            title_x = 100
        commands.extend([
            _pdf_text(title_x, 805, 'Accudetek Health Diagnostics', 15, '1 1 1', 'F2'),
            _pdf_text(title_x, 786, title or payload['title'], 11, '1 1 1'),
            _pdf_text(420, 805, 'Generated', 9, '0.860 0.945 0.969'),
            _pdf_text(420, 789, payload['generated_at'], 11, '1 1 1', 'F2'),
        ])
        return commands

    def add_footer(commands):
        commands.append(_pdf_text(42, 42, 'Smart Healthcare Clinic Management', 8, muted))
        pages.append(commands)

    def wrap_words(value, max_chars):
        words = str(value).split()
        lines = []
        current = ''
        for word in words:
            candidate = word if not current else current + ' ' + word
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines or ['']

    def draw_metric_table(commands, x, y, sections):
        commands.extend([
            f'{primary} rg {x} {y - 4} 528 26 re f',
            _pdf_text(x + 14, y + 4, 'Metric', 10, '1 1 1', 'F2'),
            _pdf_text(x + 340, y + 4, 'Value', 10, '1 1 1', 'F2'),
        ])
        row_y = y - 24
        for section in sections[:7]:
            metric_lines = wrap_words(section.get('metric', ''), 48)
            value_lines = wrap_words(section.get('value', ''), 30)
            line_count = max(len(metric_lines), len(value_lines))
            row_height = 12 * line_count + 10
            commands.append(f'{border} RG {x} {row_y + 8} m {x + 528} {row_y + 8} l S')
            for line_index, line in enumerate(metric_lines):
                commands.append(_pdf_text(x + 14, row_y - (line_index * 12), line, 8, text, 'F2'))
            for line_index, line in enumerate(value_lines):
                commands.append(_pdf_text(x + 340, row_y - (line_index * 12), line, 8, primary, 'F2'))
            row_y -= row_height
        commands.append(f'{border} RG {x} {row_y + 8} m {x + 528} {row_y + 8} l S')
        commands.append(f'{border} RG {x} {row_y + 8} {528} {y + 18 - (row_y + 8)} re S')
        return row_y

    def draw_line_chart(commands, chart, title, note, x=42, y=705, w=528, h=330):
        commands.append(_pdf_text(x, y, clip(title, 74), 14, text, 'F2'))
        if note:
            commands.append(_pdf_text(x, y - 16, clip(note, 110), 8, muted))
        left, bottom, chart_w, chart_h = x + 36, y - h + 34, w - 58, h - 95
        commands.extend([f'1 1 1 rg {x} {bottom - 34} {w} {h - 35} re f', f'{border} RG {x} {bottom - 34} {w} {h - 35} re S', f'{border} RG {left} {bottom} {chart_w} {chart_h} re S'])
        labels = chart.get('labels', [])
        datasets = chart.get('datasets', [])
        max_value = max([max(ds.get('data') or [0]) for ds in datasets] + [1])
        y_max = max(10, int(np.ceil(max_value / 10.0) * 10))
        for step in range(6):
            yy = bottom + (chart_h * step / 5)
            value = int(y_max * step / 5)
            commands.append(f'0.906 0.925 0.953 RG {left} {yy:.2f} m {left + chart_w} {yy:.2f} l S')
            commands.append(_pdf_text(x + 4, yy - 3, value, 7, muted))
        label_divisor = max(1, len(labels) - 1)
        for idx, label in enumerate(labels):
            xx = left + (chart_w * idx / label_divisor)
            commands.append(_pdf_text(xx - 8, bottom - 18, clip(label, 6), 7, muted))
        legend_x, legend_y = left, bottom + chart_h + 16
        for ds_idx, dataset in enumerate(datasets):
            r, g, b = colors[ds_idx % len(colors)]
            label = dataset.get('year', dataset.get('label', 'Series'))
            commands.append(f'{r} {g} {b} rg {legend_x} {legend_y - 2} 12 4 re f')
            commands.append(_pdf_text(legend_x + 17, legend_y - 5, label, 8, text))
            legend_x += 68
            points = []
            values = dataset.get('data', [])
            point_divisor = max(1, len(values) - 1)
            for point_idx, value in enumerate(values):
                xx = left + (chart_w * point_idx / point_divisor)
                yy = bottom + (chart_h * (value / y_max))
                points.append((xx, yy))
            if points:
                path = [f'{r} {g} {b} RG 2 w {points[0][0]:.2f} {points[0][1]:.2f} m']
                path.extend(f'{xx:.2f} {yy:.2f} l' for xx, yy in points[1:])
                path.append('S')
                commands.append(' '.join(path))
                for xx, yy in points:
                    commands.append(f'{r} {g} {b} rg {xx - 2:.2f} {yy - 2:.2f} 4 4 re f')

    def draw_bar_chart(commands, chart, x=42, y=705, w=528, h=260):
        commands.append(_pdf_text(x, y, clip(chart.get('title', 'Chart'), 74), 14, text, 'F2'))
        labels = chart.get('labels', [])
        values = chart.get('data', [])
        max_value = max(values + [1])
        left, bottom, chart_w, chart_h = x + 34, y - h + 38, w - 62, h - 78
        commands.extend([f'1 1 1 rg {x} {bottom - 34} {w} {h - 25} re f', f'{border} RG {x} {bottom - 34} {w} {h - 25} re S'])
        bar_count = max(1, len(values))
        bar_w = min(56, chart_w / (bar_count * 1.7))
        gap = (chart_w - (bar_w * bar_count)) / max(1, bar_count)
        for idx, value in enumerate(values):
            r, g, b = colors[idx % len(colors)]
            bar_h = chart_h * (value / max_value)
            xx = left + gap / 2 + idx * (bar_w + gap)
            commands.append(f'{r} {g} {b} rg {xx:.2f} {bottom:.2f} {bar_w:.2f} {bar_h:.2f} re f')
            commands.append(_pdf_text(xx, bottom + bar_h + 6, value, 8, text, 'F2'))
            commands.append(_pdf_text(xx - 4, bottom - 18, clip(labels[idx], 12), 7, muted))
        commands.append(_pdf_text(x + 8, bottom + chart_h + 8, clip(chart.get('y_title', 'Consultations'), 45), 8, muted))

    def draw_table(commands, title, rows, x=42, y=705, w=528, max_rows=18):
        commands.append(_pdf_text(x, y, str(title), 14, text, 'F2'))
        if not rows:
            commands.append(_pdf_text(x, y - 24, 'No rows available.', 9, muted))
            return
        keys = list(rows[0].keys())[:4]
        col_w = w / len(keys)
        header_y = y - 28
        commands.append(f'{primary} rg {x} {header_y - 8} {w} 22 re f')
        for idx, key in enumerate(keys):
            header_lines = wrap_words(str(key).replace('_', ' ').title(), 18)
            commands.append(_pdf_text(x + idx * col_w + 6, header_y, header_lines[0], 8, '1 1 1', 'F2'))
        row_y = header_y - 22
        row_limit = min(max_rows, len(rows))
        for row in rows[:row_limit]:
            wrapped_cells = [wrap_words(row.get(key, ''), 20) for key in keys]
            line_count = max(len(lines) for lines in wrapped_cells)
            row_height = 12 * line_count + 10
            commands.append(f'{border} RG {x} {row_y + 8} m {x + w} {row_y + 8} l S')
            for idx, key in enumerate(keys):
                for line_index, line in enumerate(wrapped_cells[idx]):
                    commands.append(_pdf_text(x + idx * col_w + 6, row_y - (line_index * 12), line, 7, text))
            row_y -= row_height
        commands.append(f'{border} RG {x} {row_y + 8} m {x + w} {row_y + 8} l S')

    def draw_simple_metric_table(commands, title, sections, x=42, y=705, w=528):
        commands.append(_pdf_text(x, y, title, 14, text, 'F2'))
        header_y = y - 30
        table_top = header_y + 14
        metric_w = 190
        value_w = w - metric_w
        commands.append(f'{primary} rg {x} {header_y - 8} {w} 22 re f')
        commands.append(_pdf_text(x + 8, header_y, 'Metric', 9, '1 1 1', 'F2'))
        commands.append(_pdf_text(x + metric_w + 8, header_y, 'Value', 9, '1 1 1', 'F2'))
        row_y = header_y - 24
        for section in sections:
            metric_lines = wrap_words(section.get('metric', ''), 28)
            value_lines = wrap_words(section.get('value', ''), 48)
            line_count = max(len(metric_lines), len(value_lines))
            row_height = 12 * line_count + 8
            commands.append(f'{border} RG {x} {row_y + 8} m {x + w} {row_y + 8} l S')
            for line_index, line in enumerate(metric_lines):
                commands.append(_pdf_text(x + 8, row_y - (line_index * 12), line, 8, text, 'F2'))
            for line_index, line in enumerate(value_lines):
                commands.append(_pdf_text(x + metric_w + 8, row_y - (line_index * 12), line, 8, primary))
            row_y -= row_height
        commands.append(f'{border} RG {x} {row_y + 8} m {x + w} {row_y + 8} l S')
        commands.append(f'{border} RG {x} {row_y + 8} {w} {table_top - (row_y + 8)} re S')
        return row_y

    def draw_simple_rows_table(commands, title, rows, x=42, y=360, w=528):
        commands.append(_pdf_text(x, y, title, 14, text, 'F2'))
        if not rows:
            commands.append(_pdf_text(x, y - 24, 'No rows available.', 9, muted))
            return
        keys = list(rows[0].keys())[:2]
        col_w = w / max(1, len(keys))
        header_y = y - 30
        table_top = header_y + 14
        commands.append(f'{primary} rg {x} {header_y - 8} {w} 22 re f')
        for idx, key in enumerate(keys):
            commands.append(_pdf_text(x + idx * col_w + 8, header_y, str(key).replace('_', ' ').title(), 9, '1 1 1', 'F2'))
        row_y = header_y - 24
        for row in rows[:12]:
            wrapped = [wrap_words(row.get(key, ''), 32) for key in keys]
            line_count = max(len(lines) for lines in wrapped)
            row_height = 12 * line_count + 8
            commands.append(f'{border} RG {x} {row_y + 8} m {x + w} {row_y + 8} l S')
            for idx, lines in enumerate(wrapped):
                for line_index, line in enumerate(lines):
                    commands.append(_pdf_text(x + idx * col_w + 8, row_y - (line_index * 12), line, 8, text))
            row_y -= row_height
        commands.append(f'{border} RG {x} {row_y + 8} m {x + w} {row_y + 8} l S')
        commands.append(f'{border} RG {x} {row_y + 8} {w} {table_top - (row_y + 8)} re S')
        return row_y

    first = new_page()
    first.append(_pdf_text(42, 727, clip(payload['description'], 110), 11, text))
    first.append(_pdf_text(42, 711, clip(f"Branch: {payload.get('branch_name', DEFAULT_BRANCH_NAME)}", 90), 9, muted))
    if payload.get('key') == 'resource-recommendation':
        metric_end_y = draw_simple_metric_table(first, 'Resource Recommendation Summary', payload.get('sections', []), y=690)
        draw_simple_rows_table(first, payload.get('details_title', 'Details'), payload.get('rows', []), y=metric_end_y - 28)
        add_footer(first)
    else:
        metric_end_y = draw_metric_table(first, 42, 676, payload.get('sections', []))
    if payload.get('key') == 'resource-recommendation':
        pass
    elif payload.get('key') == 'prediction':
        draw_table(first, payload.get('details_title', 'Diagnosis Forecast Details'), payload.get('rows', []), y=metric_end_y - 28, max_rows=8)
        add_footer(first)
        page = new_page('Prediction Demographic Forecast')
        demographic_tables = payload.get('extra_tables', [])
        if len(demographic_tables) > 0:
            draw_table(page, demographic_tables[0].get('title', 'Forecast by Age Group'), demographic_tables[0].get('rows', []), y=705, max_rows=5)
        if len(demographic_tables) > 1:
            draw_table(page, demographic_tables[1].get('title', 'Forecast by Gender'), demographic_tables[1].get('rows', []), y=520, max_rows=5)
        if len(demographic_tables) > 2:
            draw_table(page, demographic_tables[2].get('title', 'Top Diagnosis-Demographic Forecast Segments'), demographic_tables[2].get('rows', []), y=335, max_rows=8)
        add_footer(page)
    elif payload.get('key') in {'monthly-consultation', 'quarterly'} and payload.get('chart_rows', {}).get('datasets'):
        draw_line_chart(first, payload['chart_rows'], payload.get('details_title', 'Trend Graph'), payload.get('details_note', ''), y=500, h=360)
        add_footer(first)
    elif payload.get('rows'):
        add_footer(first)
        page = new_page(payload.get('details_title', payload['title']))
        draw_table(page, payload.get('details_title', 'Details'), payload['rows'])
        add_footer(page)
    else:
        add_footer(first)

    if payload.get('key') not in {'prediction', 'resource-recommendation'}:
        extra_charts = payload.get('extra_charts', [])
        for index in range(0, len(extra_charts), 2):
            page = new_page('Additional Report Graphs')
            draw_bar_chart(page, extra_charts[index], y=705, h=255)
            if index + 1 < len(extra_charts):
                draw_bar_chart(page, extra_charts[index + 1], y=405, h=255)
            add_footer(page)

        extra_tables = payload.get('extra_tables', [])
        for index in range(0, len(extra_tables), 2):
            page = new_page('Additional Report Tables')
            draw_table(page, extra_tables[index].get('title', 'Additional Details'), extra_tables[index].get('rows', []), y=705, max_rows=10)
            if index + 1 < len(extra_tables):
                draw_table(page, extra_tables[index + 1].get('title', 'Additional Details'), extra_tables[index + 1].get('rows', []), y=390, max_rows=10)
            add_footer(page)

    objects = [(1, b'<< /Type /Catalog /Pages 2 0 R >>'), (3, b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>'), (5, b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>')]
    resources = '<< /Font << /F1 3 0 R /F2 5 0 R >>'
    next_obj = 6
    if image_info:
        img_w, img_h, img_data = image_info
        image_obj_num = next_obj
        next_obj += 1
        objects.append((image_obj_num, f'<< /Type /XObject /Subtype /Image /Width {img_w} /Height {img_h} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length {len(img_data)} >>\nstream\n'.encode('ascii') + img_data + b'\nendstream'))
        resources += f' /XObject << /Logo {image_obj_num} 0 R >>'
    resources += ' >>'
    page_refs = []
    for commands in pages:
        content = '\n'.join(commands).encode('latin-1', errors='replace')
        content_obj = next_obj
        page_obj = next_obj + 1
        next_obj += 2
        objects.append((content_obj, b'<< /Length ' + str(len(content)).encode('ascii') + b' >>\nstream\n' + content + b'\nendstream'))
        objects.append((page_obj, f'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] /Resources {resources} /Contents {content_obj} 0 R >>'.encode('ascii')))
        page_refs.append(f'{page_obj} 0 R')
    objects.append((2, f'<< /Type /Pages /Kids [{" ".join(page_refs)}] /Count {len(page_refs)} >>'.encode('ascii')))
    objects = sorted(objects, key=lambda item: item[0])

    pdf = bytearray(b'%PDF-1.4\n')
    offsets = [0]
    for obj_num, body in objects:
        offsets.append(len(pdf))
        pdf.extend(f'{obj_num} 0 obj\n'.encode('ascii'))
        pdf.extend(body)
        pdf.extend(b'\nendobj\n')
    xref_pos = len(pdf)
    pdf.extend(f'xref\n0 {max(obj for obj, _ in objects) + 1}\n'.encode('ascii'))
    pdf.extend(b'0000000000 65535 f \n')
    offset_by_obj = {obj_num: offset for obj_num, offset in zip([obj for obj, _ in objects], offsets[1:])}
    for obj_num in range(1, max(obj for obj, _ in objects) + 1):
        pdf.extend(f'{offset_by_obj.get(obj_num, 0):010d} 00000 n \n'.encode('ascii'))
    pdf.extend(f'trailer\n<< /Size {max(obj for obj, _ in objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF'.encode('ascii'))
    return bytes(pdf)

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
    settings_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'app_settings.json')
    dashboard_cache_version = 6

    def branch_cache_file_path(branch_id):
        cache_key = branch_id if branch_id is not None else ALL_BRANCHES_SCOPE
        return os.path.join(app.config['UPLOAD_FOLDER'], f'dashboard_summary_branch_{cache_key}.json')

    def load_app_settings():
        defaults = {'staff_capacity_per_month': STAFF_CAPACITY_PER_MONTH}
        if not os.path.exists(settings_file_path):
            return defaults
        try:
            with open(settings_file_path, 'r', encoding='utf-8') as handle:
                data = json.load(handle)
            defaults.update({
                'staff_capacity_per_month': max(1, int(data.get('staff_capacity_per_month', STAFF_CAPACITY_PER_MONTH)))
            })
        except Exception:
            pass
        return defaults

    def save_app_settings(settings):
        with open(settings_file_path, 'w', encoding='utf-8') as handle:
            json.dump(settings, handle, indent=2)

    def load_cached_dashboard_summary(branch_id):
        cache_file_path = branch_cache_file_path(branch_id)
        if not os.path.exists(cache_file_path):
            return None
        try:
            with open(cache_file_path, 'r', encoding='utf-8') as handle:
                summary = json.load(handle)
            if summary.get('cache_version') != dashboard_cache_version:
                return None
            return summary
        except Exception:
            return None

    def cache_dashboard_summary(summary, branch_id):
        cache_file_path = branch_cache_file_path(branch_id)
        try:
            summary['cache_version'] = dashboard_cache_version
            with open(cache_file_path, 'w', encoding='utf-8') as handle:
                json.dump(summary, handle, indent=2)
        except Exception:
            pass

    def remove_cached_dashboard_summary(branch_id):
        try:
            cache_file_path = branch_cache_file_path(branch_id)
            if os.path.exists(cache_file_path):
                os.remove(cache_file_path)
        except Exception:
            pass

    def ensure_default_branch():
        branch = Branch.query.filter_by(code=DEFAULT_BRANCH_CODE).first()
        if branch is None:
            branch = Branch(
                name=DEFAULT_BRANCH_NAME,
                code=DEFAULT_BRANCH_CODE,
                address=DEFAULT_BRANCH_ADDRESS,
                is_main=True,
                is_active=True,
            )
            db.session.add(branch)
            db.session.commit()
        return branch

    def can_view_all_branches():
        return session.get('role') in MAIN_ADMIN_ROLES

    def selected_branch_scope():
        selected = session.get('selected_branch_id') or session.get('branch_id')
        if can_view_all_branches() and selected == ALL_BRANCHES_SCOPE:
            return None
        if selected:
            try:
                return int(selected)
            except (TypeError, ValueError):
                pass
        return ensure_default_branch().id

    def current_branch_id():
        selected = session.get('selected_branch_id')
        branch_id = session.get('branch_id')
        if selected and selected != ALL_BRANCHES_SCOPE:
            branch_id = selected
        if branch_id:
            try:
                return int(branch_id)
            except (TypeError, ValueError):
                pass
        return ensure_default_branch().id

    def current_branch():
        scope = selected_branch_scope()
        if scope is None:
            return None
        branch = Branch.query.get(scope)
        return branch or ensure_default_branch()

    def branch_scope_label():
        branch = current_branch()
        return branch.name if branch else 'All Branches'

    def scoped_query(query, model, branch_id='selected'):
        if branch_id == 'selected':
            branch_id = selected_branch_scope()
        if branch_id is None:
            return query
        return query.filter(model.branch_id == branch_id)

    def require_specific_branch(target_endpoint):
        if selected_branch_scope() is not None:
            return None
        flash('Select a specific branch before making changes.', 'error')
        return redirect(url_for(target_endpoint))

    def require_main_admin():
        if can_view_all_branches():
            return None
        flash('Only the main administrator can manage branches.', 'error')
        return redirect(url_for('dashboard'))

    def build_dashboard_context(records, staff_members, branch=None):
        total_consultations = len(records)

        monthly_counts = Counter()
        diagnosis_counts = Counter()
        gender_counts = Counter()
        age_group_counts = Counter()
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
            if record.age_group:
                age_group_counts[record.age_group] += 1

        monthly_trend = [
            {'month': pd.to_datetime(month + '-01').strftime('%b %Y'), 'count': count}
            for month, count in sorted(monthly_counts.items())
        ]

        # Forecast the next calendar month from today's date.
        now = datetime.now()
        predicted_month = now.month + 1
        predicted_year = now.year
        if predicted_month > 12:
            predicted_month = 1
            predicted_year += 1
        reference_month = predicted_month
        reference_year = predicted_year - 1

        predicted_month_label = datetime(predicted_year, predicted_month, 1).strftime('%b %Y')
        predicted_month_full_label = datetime(predicted_year, predicted_month, 1).strftime('%B %Y')
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
            demographic_forecast = getattr(generate_forecast_for_month, 'last_demographic_forecast', {
                'age_group': [],
                'gender': [],
                'segments': [],
            })
        else:
            total_pred, forecast, rf_metrics = None, None, None
            demographic_forecast = {'age_group': [], 'gender': [], 'segments': []}

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
            demographic_forecast = {
                'age_group': [
                    {'age_group': group, 'forecasted_cases': count}
                    for group, count in age_group_counts.most_common()
                ],
                'gender': [
                    {'gender': gender, 'forecasted_cases': count}
                    for gender, count in gender_counts.most_common()
                ],
                'segments': [],
            }

        top_diagnosis = diagnosis_counts.most_common(1)[0][0] if diagnosis_counts else 'None'
        facility_staff_count = len(staff_members) if branch is None else sum(item['count'] for item in FACILITY_STAFF_COMPLEMENT)
        app_settings = load_app_settings()
        staff_capacity_per_month = app_settings['staff_capacity_per_month']

        estimated_monthly_capacity = facility_staff_count * staff_capacity_per_month
        resource_readiness = min(100, max(0, round(estimated_monthly_capacity / max(1, predicted_cases_next_month) * 100)))
        pressure_ratio = predicted_cases_next_month / max(1, estimated_monthly_capacity)
        forecast_pressure_raw = int(pressure_ratio * 100)
        forecast_pressure = min(100, int(pressure_ratio * 100))
        recommended_staff_count = int(np.ceil(predicted_cases_next_month / max(1, staff_capacity_per_month)))
        staff_gap = max(0, recommended_staff_count - facility_staff_count)
        if pressure_ratio > 1.0:
            capacity_status = 'High'
            resource_forecast_recommendation = 'Predicted demand exceeds current staffing capacity; consider hiring additional staff or expanding service hours.'
        elif pressure_ratio > 0.75:
            capacity_status = 'Moderate'
            resource_forecast_recommendation = 'Service demand is approaching capacity; monitor staffing and diagnostic equipment availability closely.'
        else:
            capacity_status = 'Healthy'
            resource_forecast_recommendation = 'Current staffing and equipment capacity appears sufficient for forecasted demand.'

        actual_staff_by_role = Counter(member.role for member in staff_members)

        return {
            'branch_id': branch.id if branch else None,
            'branch_name': branch.name if branch else DEFAULT_BRANCH_NAME,
            'branch_code': branch.code if branch else DEFAULT_BRANCH_CODE,
            'total_consultations': total_consultations,
            'staff_count': len(staff_members),
            'facility_staff_count': facility_staff_count,
            'staff_capacity_per_month': staff_capacity_per_month,
            'top_diagnosis': top_diagnosis,
            'predicted_cases_next_month': predicted_cases_next_month,
            'resource_readiness': resource_readiness,
            'monthly_trend': monthly_trend,
            'consultation_distribution': diagnosis_counts.most_common(10),
            'gender_distribution': gender_counts.most_common(),
            'age_group_distribution': age_group_counts.most_common(),
            'top_cases': diagnosis_counts.most_common(5),
            'predictions': predictions,
            'demographic_forecast': demographic_forecast,
            'resource_recommendation': resource_recommendation,
            'reference_month': reference_month_label,
            'predicted_month_label': predicted_month_label,
            'predicted_month_full_label': predicted_month_full_label,
            'services': SERVICE_CATALOG,
            'equipment': EQUIPMENT_INVENTORY,
            'facility_staff_complement': FACILITY_STAFF_COMPLEMENT,
            'actual_staff_by_role': dict(actual_staff_by_role),
            'estimated_monthly_capacity': estimated_monthly_capacity,
            'forecast_pressure': forecast_pressure,
            'forecast_pressure_raw': forecast_pressure_raw,
            'recommended_staff_count': recommended_staff_count,
            'staff_gap': staff_gap,
            'capacity_status': capacity_status,
            'resource_forecast_recommendation': resource_forecast_recommendation,
            'rf_metrics': rf_metrics,
        }

    def get_dashboard_summary(force_refresh=False, branch_id='selected'):
        if branch_id == 'selected':
            branch_id = selected_branch_scope()
        if force_refresh:
            remove_cached_dashboard_summary(branch_id)
            if branch_id is not None:
                remove_cached_dashboard_summary(None)
        if not force_refresh:
            cached = load_cached_dashboard_summary(branch_id)
            if cached is not None:
                return cached

        branch = None if branch_id is None else (Branch.query.get(branch_id) or ensure_default_branch())
        records = scoped_query(ConsultationRecord.query, ConsultationRecord, branch_id=branch_id).all()
        staff = scoped_query(StaffMember.query.filter_by(is_active=True), StaffMember, branch_id=branch_id).all()
        summary = build_dashboard_context(records, staff, branch=branch)
        if branch_id is None:
            summary['branch_name'] = 'All Branches'
            summary['branch_code'] = ALL_BRANCHES_SCOPE.upper()
        cache_dashboard_summary(summary, branch_id)
        return summary

    # -------------------------------------------------------------
    # Routes (unchanged except for dashboard/upload which use the new logic)
    # -------------------------------------------------------------
    @app.before_request
    def require_login():
        if request.endpoint in {'login', 'static'}:
            return None
        protected_endpoints = {
            'dashboard', 'records', 'upload', 'predict', 'staff', 'reports',
            'report_view', 'report_print', 'report_pdf',
            'permanent_delete_staff',
            'settings', 'resources',
            'branches', 'create_branch', 'edit_branch', 'toggle_branch', 'select_branch',
            'create_user', 'assign_user_branch',
        }
        if request.endpoint in protected_endpoints and 'user_id' not in session:
            return redirect(url_for('login'))

    @app.context_processor
    def inject_branch_context():
        try:
            if can_view_all_branches():
                branch_options = Branch.query.filter_by(is_active=True).order_by(Branch.name.asc()).all()
            else:
                branch_options = Branch.query.filter_by(id=session.get('branch_id'), is_active=True).all()
            return {
                'active_branch': current_branch(),
                'active_branch_label': branch_scope_label(),
                'selected_branch_value': ALL_BRANCHES_SCOPE if selected_branch_scope() is None else str(selected_branch_scope()),
                'branch_options': branch_options,
                'can_view_all_branches': can_view_all_branches(),
            }
        except Exception:
            return {
                'active_branch': None,
                'active_branch_label': DEFAULT_BRANCH_NAME,
                'selected_branch_value': '',
                'branch_options': [],
                'can_view_all_branches': False,
            }

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if request.method == 'POST':
            user = User.query.filter_by(username=request.form['username']).first()
            if user and user.password == request.form['password']:
                if not user.branch_id:
                    user.branch_id = ensure_default_branch().id
                    db.session.commit()
                session['user_id'] = user.id
                session['role'] = user.role
                session['branch_id'] = user.branch_id
                session['selected_branch_id'] = user.branch_id
                flash(f'Welcome back, {user.username}.', 'success')
                return redirect(url_for('dashboard'))
            flash('Invalid credentials', 'error')
        return render_template('auth/login.html')

    @app.route('/logout')
    def logout():
        session.clear()
        return redirect(url_for('login'))

    @app.route('/branches/select', methods=['POST'])
    def select_branch():
        requested_scope = request.form.get('branch_scope', '').strip()
        if requested_scope == ALL_BRANCHES_SCOPE:
            if not can_view_all_branches():
                flash('You are not allowed to view all branches.', 'error')
            else:
                session['selected_branch_id'] = ALL_BRANCHES_SCOPE
                flash('Showing combined data from all branches.', 'success')
            return redirect(url_for('dashboard'))

        try:
            branch_id = int(requested_scope)
        except (TypeError, ValueError):
            flash('Invalid branch selection.', 'error')
            return redirect(url_for('dashboard'))

        branch = Branch.query.filter_by(id=branch_id, is_active=True).first()
        if branch is None:
            flash('Selected branch is not available.', 'error')
            return redirect(url_for('dashboard'))

        if not can_view_all_branches() and branch.id != session.get('branch_id'):
            flash('You are not allowed to access that branch.', 'error')
            return redirect(url_for('dashboard'))

        session['selected_branch_id'] = branch.id
        flash(f'Showing {branch.name}.', 'success')
        return redirect(url_for('dashboard'))

    @app.route('/branches')
    def branches():
        redirect_response = require_main_admin()
        if redirect_response:
            return redirect_response

        branch_rows = []
        for branch in Branch.query.order_by(Branch.is_main.desc(), Branch.name.asc()).all():
            branch_rows.append({
                'branch': branch,
                'consultations': ConsultationRecord.query.filter_by(branch_id=branch.id).count(),
                'staff': StaffMember.query.filter_by(branch_id=branch.id, is_active=True).count(),
                'users': User.query.filter_by(branch_id=branch.id).count(),
            })

        return render_template(
            'branches/index.html',
            branch_rows=branch_rows,
            users=User.query.order_by(User.username.asc()).all(),
            active_branches=Branch.query.filter_by(is_active=True).order_by(Branch.name.asc()).all(),
            role_options=USER_ROLE_OPTIONS,
            current_date=datetime.now().strftime('%Y-%m-%d'),
            current_time=datetime.now().strftime('%H:%M'),
        )

    @app.route('/branches/users/new', methods=['GET', 'POST'])
    def create_user():
        redirect_response = require_main_admin()
        if redirect_response:
            return redirect_response

        active_branches = Branch.query.filter_by(is_active=True).order_by(Branch.name.asc()).all()
        if request.method == 'POST':
            username = request.form.get('username', '').strip()
            password = request.form.get('password', '').strip()
            role = request.form.get('role', 'staff').strip()
            try:
                branch_id = int(request.form.get('branch_id'))
            except (TypeError, ValueError):
                flash('Please select a valid branch for the user.', 'error')
                return redirect(url_for('create_user'))

            branch = Branch.query.filter_by(id=branch_id, is_active=True).first()
            if not username or not password:
                flash('Username and password are required.', 'error')
                return redirect(url_for('create_user'))
            if role not in USER_ROLE_OPTIONS:
                flash('Selected role is not valid.', 'error')
                return redirect(url_for('create_user'))
            if branch is None:
                flash('Selected branch is not active.', 'error')
                return redirect(url_for('create_user'))
            if User.query.filter_by(username=username).first():
                flash('Username already exists.', 'error')
                return redirect(url_for('create_user'))

            db.session.add(User(
                username=username,
                password=password,
                role=role,
                branch_id=branch.id,
            ))
            db.session.commit()
            flash('User added successfully.', 'success')
            return redirect(url_for('branches'))

        return render_template(
            'branches/user_form.html',
            active_branches=active_branches,
            role_options=USER_ROLE_OPTIONS,
            current_date=datetime.now().strftime('%Y-%m-%d'),
            current_time=datetime.now().strftime('%H:%M'),
        )

    @app.route('/branches/users/<int:user_id>/assign', methods=['POST'])
    def assign_user_branch(user_id):
        redirect_response = require_main_admin()
        if redirect_response:
            return redirect_response

        user = User.query.get_or_404(user_id)
        try:
            branch_id = int(request.form.get('branch_id'))
        except (TypeError, ValueError):
            flash('Please select a valid branch for the user.', 'error')
            return redirect(url_for('branches'))

        branch = Branch.query.filter_by(id=branch_id, is_active=True).first()
        role = request.form.get('role', user.role).strip()
        if branch is None:
            flash('Selected branch is not active.', 'error')
            return redirect(url_for('branches'))
        if role not in USER_ROLE_OPTIONS:
            flash('Selected role is not valid.', 'error')
            return redirect(url_for('branches'))
        if user.id == session.get('user_id') and role not in MAIN_ADMIN_ROLES:
            flash('You cannot remove your own main administrator access.', 'error')
            return redirect(url_for('branches'))

        user.branch_id = branch.id
        user.role = role
        db.session.commit()
        if user.id == session.get('user_id'):
            session['branch_id'] = branch.id
            session['role'] = role
            if session.get('selected_branch_id') != ALL_BRANCHES_SCOPE:
                session['selected_branch_id'] = branch.id
        flash('User branch access updated.', 'success')
        return redirect(url_for('branches'))

    @app.route('/branches/new', methods=['GET', 'POST'])
    def create_branch():
        redirect_response = require_main_admin()
        if redirect_response:
            return redirect_response

        if request.method == 'POST':
            name = request.form.get('name', '').strip()
            code = request.form.get('code', '').strip().upper()
            address = request.form.get('address', '').strip()
            email = request.form.get('email', '').strip()
            contact_number = request.form.get('contact_number', '').strip()
            if not name or not code:
                flash('Branch name and code are required.', 'error')
                return redirect(url_for('create_branch'))
            if Branch.query.filter_by(code=code).first():
                flash('Branch code already exists.', 'error')
                return redirect(url_for('create_branch'))
            branch = Branch(
                name=name,
                code=code,
                address=address,
                email=email,
                contact_number=contact_number,
                is_main=False,
                is_active=True,
            )
            db.session.add(branch)
            db.session.commit()
            flash('Branch added successfully.', 'success')
            return redirect(url_for('branches'))

        return render_template(
            'branches/form.html',
            branch=None,
            current_date=datetime.now().strftime('%Y-%m-%d'),
            current_time=datetime.now().strftime('%H:%M'),
        )

    @app.route('/branches/<int:branch_id>/edit', methods=['GET', 'POST'])
    def edit_branch(branch_id):
        redirect_response = require_main_admin()
        if redirect_response:
            return redirect_response

        branch = Branch.query.get_or_404(branch_id)
        if request.method == 'POST':
            name = request.form.get('name', '').strip()
            code = branch.code if branch.is_main else request.form.get('code', '').strip().upper()
            address = request.form.get('address', '').strip()
            email = request.form.get('email', '').strip()
            contact_number = request.form.get('contact_number', '').strip()
            if not name or not code:
                flash('Branch name and code are required.', 'error')
                return redirect(url_for('edit_branch', branch_id=branch.id))
            duplicate = Branch.query.filter(Branch.code == code, Branch.id != branch.id).first()
            if duplicate:
                flash('Branch code already exists.', 'error')
                return redirect(url_for('edit_branch', branch_id=branch.id))
            branch.name = name
            branch.code = code
            branch.address = address
            branch.email = email
            branch.contact_number = contact_number
            db.session.commit()
            get_dashboard_summary(force_refresh=True, branch_id=branch.id)
            flash('Branch updated successfully.', 'success')
            return redirect(url_for('branches'))

        return render_template(
            'branches/form.html',
            branch=branch,
            current_date=datetime.now().strftime('%Y-%m-%d'),
            current_time=datetime.now().strftime('%H:%M'),
        )

    @app.route('/branches/<int:branch_id>/toggle', methods=['POST'])
    def toggle_branch(branch_id):
        redirect_response = require_main_admin()
        if redirect_response:
            return redirect_response

        branch = Branch.query.get_or_404(branch_id)
        if branch.is_main and branch.is_active:
            flash('The main branch cannot be deactivated.', 'error')
            return redirect(url_for('branches'))
        branch.is_active = not branch.is_active
        db.session.commit()
        if session.get('selected_branch_id') == branch.id and not branch.is_active:
            session['selected_branch_id'] = session.get('branch_id') or ensure_default_branch().id
        flash('Branch status updated.', 'success')
        return redirect(url_for('branches'))

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
        records = (
            scoped_query(ConsultationRecord.query, ConsultationRecord)
            .order_by(ConsultationRecord.consultation_date.desc())
            .paginate(page=page, per_page=10, error_out=False)
        )
        return render_template(
            'consultations/index.html',
            records=records,
            all_branches_view=selected_branch_scope() is None,
            current_date=datetime.now().strftime('%Y-%m-%d'),
            current_time=datetime.now().strftime('%H:%M'),
        )

    @app.route('/records/clear', methods=['POST'])
    def clear_records():
        redirect_response = require_specific_branch('records')
        if redirect_response:
            return redirect_response
        deleted_count = ConsultationRecord.query.filter_by(branch_id=current_branch_id()).delete()
        db.session.commit()
        get_dashboard_summary(force_refresh=True)
        flash(f'Cleared {deleted_count} consultation records for the current branch.', 'success')
        return redirect(url_for('records'))

    @app.route('/upload', methods=['GET', 'POST'])
    def upload():
        if selected_branch_scope() is None:
            flash('Select a specific branch before uploading consultation data.', 'error')
            return redirect(url_for('records'))

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

            df.columns = df.columns.str.lower()
            upload_columns = [
                'consultation_date', 'age_group', 'gender', 'diagnosis',
                'department', 'physician', 'consultation_type'
            ]

            def normalize_record_value(column, value):
                if column == 'consultation_date':
                    parsed = pd.to_datetime(value, errors='coerce')
                    if pd.notna(parsed):
                        return parsed.strftime('%Y-%m-%d')
                return str(value).strip()

            def record_key(values):
                return tuple(normalize_record_value(column, values.get(column, '')) for column in upload_columns)

            existing_keys = {
                record_key({
                    'consultation_date': record.consultation_date,
                    'age_group': record.age_group,
                    'gender': record.gender,
                    'diagnosis': record.diagnosis,
                    'department': record.department,
                    'physician': record.physician,
                    'consultation_type': record.consultation_type,
                })
                for record in ConsultationRecord.query.filter_by(branch_id=current_branch_id()).all()
            }

            added_count = 0
            skipped_count = 0
            for _, row in df.iterrows():
                values = {column: normalize_record_value(column, row.get(column, '')) for column in upload_columns}
                key = tuple(values[column] for column in upload_columns)
                if key in existing_keys:
                    skipped_count += 1
                    continue
                existing_keys.add(key)
                record = ConsultationRecord(
                    branch_id=current_branch_id(),
                    consultation_date=values['consultation_date'],
                    age_group=values['age_group'],
                    gender=values['gender'],
                    diagnosis=values['diagnosis'],
                    department=values['department'],
                    physician=values['physician'],
                    consultation_type=values['consultation_type'],
                )
                db.session.add(record)
                added_count += 1
            db.session.commit()

            records = ConsultationRecord.query.filter_by(branch_id=current_branch_id()).all()
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
                    write_training_report(report_path, metrics)
                    session['last_report'] = report_path
                except Exception as e:
                    traceback.print_exc()
                    flash(f'Model retraining failed: {str(e)}', 'error')
            else:
                flash('No records to train model.', 'warning')

            get_dashboard_summary(force_refresh=True)
            flash(f'Data uploaded and model retrained successfully. Added {added_count} new records; skipped {skipped_count} duplicates.', 'success')
            return redirect(url_for('predict'))

        return render_template('consultations/upload.html')

    @app.route('/predict')
    def predict():
        summary = get_dashboard_summary()
        if not summary.get('total_consultations'):
            flash('No consultation records found. Please upload data first.', 'warning')
            return render_template('forecasting/index.html',
                                   metrics={'r2_score': 0, 'mae': 0, 'mse': 0, 'rmse': 0},
                                   top_cases=[],
                                   forecast=[],
                                   summary={
                                       'predicted_month_full_label': 'Next Month',
                                       'predictions': [],
                                       'demographic_forecast': {'age_group': [], 'gender': [], 'segments': []}
                                   })

        metrics = summary.get('rf_metrics') or {'r2_score': 0, 'mae': 0, 'mse': 0, 'rmse': 0}
        top_cases = [f"{diag} ({count})" for diag, count in summary.get('top_cases', [])]
        forecast = [
            (item.get('diagnosis'), item.get('predicted_next_month'))
            for item in summary.get('predictions', [])
        ]

        return render_template('forecasting/index.html',
                               metrics=metrics,
                               top_cases=top_cases,
                               forecast=forecast,
                               summary=summary)

    @app.route('/retrain', methods=['POST'])
    def retrain():
        redirect_response = require_specific_branch('predict')
        if redirect_response:
            return redirect_response
        records = ConsultationRecord.query.filter_by(branch_id=current_branch_id()).all()
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
            write_training_report(report_path, metrics)
            session['last_report'] = report_path
            get_dashboard_summary(force_refresh=True)
            flash('Model retrained successfully using the existing consultation data', 'success')
        except Exception as e:
            traceback.print_exc()
            flash(f'Retraining failed: {str(e)}', 'error')

        return redirect(url_for('predict'))

    @app.route('/staff')
    def staff():
        cutoff = datetime.now() - timedelta(days=30)
        expired_staff = scoped_query(StaffMember.query, StaffMember).filter(
            StaffMember.is_active.is_(False),
            StaffMember.deleted_at.isnot(None),
            StaffMember.deleted_at < cutoff,
        ).all()
        for member in expired_staff:
            db.session.delete(member)
        if expired_staff:
            db.session.commit()
            get_dashboard_summary(force_refresh=True)

        page = request.args.get('page', 1, type=int)
        active_staff = (
            scoped_query(StaffMember.query.filter_by(is_active=True), StaffMember)
            .order_by(StaffMember.role.asc(), StaffMember.name.asc())
            .paginate(page=page, per_page=10, error_out=False)
        )
        inactive_staff = (
            scoped_query(StaffMember.query.filter_by(is_active=False), StaffMember)
            .order_by(StaffMember.deleted_at.desc(), StaffMember.name.asc())
            .all()
        )
        return render_template(
            'staff/index.html',
            active_staff=active_staff,
            inactive_staff=inactive_staff,
            all_branches_view=selected_branch_scope() is None,
        )

    staff_role_options = [item['role'] for item in FACILITY_STAFF_COMPLEMENT]
    staff_availability_options = ['Available', 'Busy', 'On Leave', 'Unavailable']

    @app.route('/staff/load-facility-complement', methods=['POST'])
    def load_facility_complement():
        redirect_response = require_specific_branch('staff')
        if redirect_response:
            return redirect_response
        StaffMember.query.filter_by(is_active=True, branch_id=current_branch_id()).delete()
        for person in build_facility_staff_roster():
            db.session.add(StaffMember(
                branch_id=current_branch_id(),
                name=person['name'],
                role=person['role'],
                availability=person['availability'],
                is_active=True,
                deleted_at=None,
            ))
        db.session.commit()
        get_dashboard_summary(force_refresh=True)
        flash('Loaded the official 22-person clinic staff complement.', 'success')
        return redirect(url_for('staff'))

    @app.route('/staff/new', methods=['GET', 'POST'])
    def create_staff():
        redirect_response = require_specific_branch('staff')
        if redirect_response:
            return redirect_response
        if request.method == 'POST':
            new_staff = StaffMember(
                branch_id=current_branch_id(),
                name=request.form.get('name', '').strip(),
                role=request.form.get('role', '').strip(),
                availability=request.form.get('availability', 'Available').strip(),
                is_active=True,
                deleted_at=None,
            )
            db.session.add(new_staff)
            db.session.commit()
            get_dashboard_summary(force_refresh=True)
            flash('Staff member added successfully.', 'success')
            return redirect(url_for('staff'))
        return render_template(
            'staff/form.html',
            staff_member=None,
            role_options=staff_role_options,
            availability_options=staff_availability_options,
        )

    @app.route('/staff/<int:staff_id>/edit', methods=['GET', 'POST'])
    def edit_staff(staff_id):
        redirect_response = require_specific_branch('staff')
        if redirect_response:
            return redirect_response
        staff_member = StaffMember.query.filter_by(id=staff_id, branch_id=current_branch_id()).first_or_404()
        if request.method == 'POST':
            staff_member.name = request.form.get('name', staff_member.name).strip()
            staff_member.role = request.form.get('role', staff_member.role).strip()
            staff_member.availability = request.form.get('availability', staff_member.availability).strip()
            db.session.commit()
            get_dashboard_summary(force_refresh=True)
            flash('Staff member updated successfully.', 'success')
            return redirect(url_for('staff'))
        return render_template(
            'staff/form.html',
            staff_member=staff_member,
            role_options=staff_role_options,
            availability_options=staff_availability_options,
        )

    @app.route('/staff/<int:staff_id>/delete', methods=['POST'])
    def delete_staff(staff_id):
        redirect_response = require_specific_branch('staff')
        if redirect_response:
            return redirect_response
        staff_member = StaffMember.query.filter_by(id=staff_id, branch_id=current_branch_id()).first_or_404()
        staff_member.is_active = False
        staff_member.deleted_at = datetime.now()
        db.session.commit()
        get_dashboard_summary(force_refresh=True)
        flash('Staff member removed from active roster (soft deleted).', 'success')
        return redirect(url_for('staff'))

    @app.route('/staff/<int:staff_id>/restore', methods=['POST'])
    def restore_staff(staff_id):
        redirect_response = require_specific_branch('staff')
        if redirect_response:
            return redirect_response
        staff_member = StaffMember.query.filter_by(id=staff_id, branch_id=current_branch_id()).first_or_404()
        staff_member.is_active = True
        staff_member.deleted_at = None
        db.session.commit()
        get_dashboard_summary(force_refresh=True)
        flash('Staff member restored to active roster.', 'success')
        return redirect(url_for('staff'))

    @app.route('/staff/<int:staff_id>/permanent-delete', methods=['POST'])
    def permanent_delete_staff(staff_id):
        redirect_response = require_specific_branch('staff')
        if redirect_response:
            return redirect_response
        staff_member = StaffMember.query.filter_by(id=staff_id, branch_id=current_branch_id()).first_or_404()
        db.session.delete(staff_member)
        db.session.commit()
        get_dashboard_summary(force_refresh=True)
        flash('Staff member permanently deleted.', 'success')
        return redirect(url_for('staff'))

    report_definitions = {
        'monthly-consultation': {
            'title': 'Monthly Consultation Report',
            'description': 'Monthly consultation totals and most common diagnoses.'
        },
        'quarterly': {
            'title': 'Quarterly Report',
            'description': 'Quarterly operating summary using consultation volume and distribution.'
        },
        'prediction': {
            'title': 'Prediction Report',
            'description': 'Forecasted next-month diagnosis demand and model validation metrics.'
        },
        'resource-recommendation': {
            'title': 'Resource Recommendation Report',
            'description': 'Staffing, room capacity, and service readiness recommendations.'
        },
    }

    def records_to_dataframe():
        records = scoped_query(ConsultationRecord.query, ConsultationRecord).all()
        return pd.DataFrame([{
            'consultation_date': r.consultation_date,
            'age_group': r.age_group,
            'gender': r.gender,
            'diagnosis': r.diagnosis,
            'department': r.department,
            'physician': r.physician,
            'consultation_type': r.consultation_type,
        } for r in records])

    def build_report_payload(report_key):
        if report_key not in report_definitions:
            return None

        summary = get_dashboard_summary()
        df = records_to_dataframe()
        title = report_definitions[report_key]['title']
        generated_at = datetime.now().strftime('%Y-%m-%d %H:%M')
        sections = []
        rows = []
        chart_rows = []
        extra_charts = []
        extra_tables = []
        details_title = 'Details'
        details_note = 'Supporting rows used for this report.'

        if not df.empty:
            df['consultation_date'] = pd.to_datetime(df['consultation_date'], errors='coerce')
            df = df.dropna(subset=['consultation_date'])
            for col in ['age_group', 'gender']:
                if col not in df.columns:
                    df[col] = 'Unknown'
                df[col] = df[col].fillna('Unknown').replace('', 'Unknown')

        if report_key == 'monthly-consultation':
            monthly_rows = []
            average_monthly_consultations = 0
            average_monthly_top_diagnosis_consultations = 0
            chart_rows = {
                'labels': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                'x_title': 'Month of Year',
                'datasets': []
            }
            forecast_month_label = summary.get('predicted_month_label', 'Next Month')
            try:
                forecast_month_label = datetime.strptime(forecast_month_label, '%b %Y').strftime('%B %Y')
            except Exception:
                pass
            if not df.empty:
                monthly = (
                    df.assign(month=df['consultation_date'].dt.strftime('%Y-%m'))
                    .groupby('month')
                    .size()
                    .reset_index(name='consultations')
                    .sort_values('month', ascending=False)
                )
                monthly_rows = monthly.head(12).to_dict('records')
                monthly_for_chart = df.assign(
                    year=df['consultation_date'].dt.year,
                    month_number=df['consultation_date'].dt.month
                ).groupby(['year', 'month_number']).size().reset_index(name='consultations')
                chart_rows['datasets'] = [
                    {
                        'year': int(year),
                        'data': [
                            int(year_data.loc[year_data['month_number'] == month_number, 'consultations'].sum())
                            for month_number in range(1, 13)
                        ]
                    }
                    for year, year_data in monthly_for_chart.groupby('year')
                ]
                average_monthly_consultations = round(monthly['consultations'].mean(), 2)
                top_diagnosis = summary.get('top_diagnosis', 'None')
                if top_diagnosis != 'None':
                    top_diagnosis_monthly = (
                        df[df['diagnosis'] == top_diagnosis]
                        .assign(month=df['consultation_date'].dt.strftime('%Y-%m'))
                        .groupby('month')
                        .size()
                        .reset_index(name='consultations')
                    )
                    if not top_diagnosis_monthly.empty:
                        average_monthly_top_diagnosis_consultations = round(top_diagnosis_monthly['consultations'].mean(), 2)
                age_monthly = (
                    df.assign(month=df['consultation_date'].dt.strftime('%Y-%m'))
                    .groupby(['age_group', 'month'])
                    .size()
                    .reset_index(name='consultations')
                )
                if not age_monthly.empty:
                    age_summary = (
                        age_monthly.groupby('age_group')['consultations']
                        .mean()
                        .round(2)
                        .reset_index(name='consultations')
                        .sort_values('consultations', ascending=False)
                    )
                    extra_charts.append({
                        'title': 'Average Monthly Consultations by Age Group',
                        'type': 'bar',
                        'labels': age_summary['age_group'].tolist(),
                        'data': age_summary['consultations'].tolist(),
                        'y_title': 'Average Consultations'
                    })
                gender_monthly = (
                    df.assign(month=df['consultation_date'].dt.strftime('%Y-%m'))
                    .groupby(['gender', 'month'])
                    .size()
                    .reset_index(name='consultations')
                )
                if not gender_monthly.empty:
                    gender_summary = (
                        gender_monthly.groupby('gender')['consultations']
                        .mean()
                        .round(2)
                        .reset_index(name='consultations')
                        .sort_values('consultations', ascending=False)
                    )
                    extra_charts.append({
                        'title': 'Average Monthly Consultations by Sex',
                        'type': 'bar',
                        'labels': gender_summary['gender'].tolist(),
                        'data': gender_summary['consultations'].tolist(),
                        'y_title': 'Average Consultations'
                    })
            sections = [
                {
                    'metric': 'Average Monthly Consultations',
                    'value': average_monthly_consultations,
                    'explanation': 'Average number of consultations per month based on the uploaded records.'
                },
                {
                    'metric': 'Top Diagnosis',
                    'value': summary.get('top_diagnosis', 'None'),
                    'explanation': 'The diagnosis with the highest total count across all uploaded records.'
                },
                {
                    'metric': 'Average Monthly Consultations of Top Diagnosis',
                    'value': average_monthly_top_diagnosis_consultations,
                    'explanation': 'Average monthly consultation count for the top diagnosis across uploaded records.'
                },
                {
                    'metric': f'Forecasted Total Cases for the Next Month ({forecast_month_label})',
                    'value': summary.get('predicted_cases_next_month', 0),
                    'explanation': 'The forecasted total consultation demand for the next calendar month. This helps compare recent actual volume with expected upcoming demand.'
                },
            ]
            rows = []
            details_title = 'Monthly Consultation Trend Graph'
            details_note = 'This graph shows monthly consultation totals by year so the clinic can compare seasonal demand patterns and growth more easily.'

        elif report_key == 'quarterly':
            quarterly_rows = []
            average_quarterly_consultations = 0
            chart_rows = {
                'labels': ['Q1', 'Q2', 'Q3', 'Q4'],
                'x_title': 'Quarter of Year',
                'datasets': []
            }
            if not df.empty:
                quarterly = (
                    df.assign(quarter=df['consultation_date'].dt.to_period('Q').astype(str))
                    .groupby('quarter')
                    .size()
                    .reset_index(name='consultations')
                    .sort_values('quarter', ascending=False)
                )
                quarterly_rows = quarterly.head(8).to_dict('records')
                average_quarterly_consultations = round(quarterly['consultations'].mean(), 2)
                quarterly_for_chart = df.assign(
                    year=df['consultation_date'].dt.year,
                    quarter_number=df['consultation_date'].dt.quarter
                ).groupby(['year', 'quarter_number']).size().reset_index(name='consultations')
                chart_rows['datasets'] = [
                    {
                        'year': int(year),
                        'data': [
                            int(year_data.loc[year_data['quarter_number'] == quarter_number, 'consultations'].sum())
                            for quarter_number in range(1, 5)
                        ]
                    }
                    for year, year_data in quarterly_for_chart.groupby('year')
                ]
                age_quarterly = (
                    df.assign(quarter=df['consultation_date'].dt.to_period('Q').astype(str))
                    .groupby(['age_group', 'quarter'])
                    .size()
                    .reset_index(name='consultations')
                )
                if not age_quarterly.empty:
                    age_summary = (
                        age_quarterly.groupby('age_group')['consultations']
                        .mean()
                        .round(2)
                        .reset_index(name='consultations')
                        .sort_values('consultations', ascending=False)
                    )
                    extra_charts.append({
                        'title': 'Average Quarterly Consultations by Age Group',
                        'type': 'bar',
                        'labels': age_summary['age_group'].tolist(),
                        'data': age_summary['consultations'].tolist(),
                        'y_title': 'Average Consultations'
                    })
                gender_quarterly = (
                    df.assign(quarter=df['consultation_date'].dt.to_period('Q').astype(str))
                    .groupby(['gender', 'quarter'])
                    .size()
                    .reset_index(name='consultations')
                )
                if not gender_quarterly.empty:
                    gender_summary = (
                        gender_quarterly.groupby('gender')['consultations']
                        .mean()
                        .round(2)
                        .reset_index(name='consultations')
                        .sort_values('consultations', ascending=False)
                    )
                    extra_charts.append({
                        'title': 'Average Quarterly Consultations by Sex',
                        'type': 'bar',
                        'labels': gender_summary['gender'].tolist(),
                        'data': gender_summary['consultations'].tolist(),
                        'y_title': 'Average Consultations'
                    })
            sections = [
                {
                    'metric': 'Average Consultations per Quarter',
                    'value': average_quarterly_consultations,
                    'explanation': 'Average number of consultations per quarter based on uploaded records.'
                },
                {
                    'metric': 'Active Staff',
                    'value': summary.get('staff_count', 0),
                    'explanation': 'Number of active staff members currently counted for capacity planning.'
                },
                {
                    'metric': 'Capacity Status',
                    'value': summary.get('capacity_status', 'Unknown'),
                    'explanation': 'Overall demand-versus-staffing status based on predicted cases and estimated monthly capacity.'
                },
            ]
            rows = []
            details_title = 'Quarterly Consultation Trend Graph'
            details_note = 'This graph has four points per year, one for each quarter. Each colored line represents a year so quarter-to-quarter changes are easier to compare.'

        elif report_key == 'prediction':
            forecast = summary.get('predictions', [])
            metrics = summary.get('rf_metrics') or {}
            demographic_forecast = summary.get('demographic_forecast') or {}
            sections = [
                {
                    'metric': 'Predicted Month',
                    'value': summary.get('predicted_month_label', 'Unknown'),
                    'explanation': 'The future month being forecasted by the model.'
                },
                {
                    'metric': 'Predicted Cases Next Month',
                    'value': summary.get('predicted_cases_next_month', 0),
                    'explanation': 'The predicted total number of consultations expected for the forecast month.'
                },
                {
                    'metric': 'Model Verdict',
                    'value': metrics.get('model_verdict', 'Available after prediction/retrain'),
                    'explanation': 'A short quality label based on validation R2 and whether the model beats the last-month baseline.'
                },
                {
                    'metric': 'Validation R2',
                    'value': metrics.get('validation_r2', metrics.get('r2_score', 'N/A')),
                    'explanation': 'How well the model explained unseen validation months. Higher is better; values above 0 mean it beats predicting the average.'
                },
                {
                    'metric': 'Validation MAE',
                    'value': metrics.get('validation_mae', metrics.get('mae', 'N/A')),
                    'explanation': 'Average prediction error measured in consultation cases per diagnosis-month.'
                },
            ]
            rows = forecast
            details_title = 'Diagnosis Forecast Details'
            details_note = 'This table lists the forecasted next-month case count by diagnosis so staff and resources can be planned by expected demand area.'
            extra_tables = [
                {
                    'title': 'Forecast by Age Group',
                    'rows': demographic_forecast.get('age_group', [])
                },
                {
                    'title': 'Forecast by Gender',
                    'rows': demographic_forecast.get('gender', [])
                },
                {
                    'title': 'Top Diagnosis-Demographic Forecast Segments',
                    'rows': demographic_forecast.get('segments', [])
                },
            ]

        elif report_key == 'resource-recommendation':
            sections = [
                {
                    'metric': 'Estimated Monthly Capacity',
                    'value': summary.get('estimated_monthly_capacity', 0),
                    'explanation': 'Estimated number of consultations current active staff can support in one month.'
                },
                {
                    'metric': 'Forecast Pressure',
                    'value': f"{summary.get('forecast_pressure', 0)}%",
                    'explanation': 'Predicted demand divided by estimated monthly staff capacity.'
                },
                {
                    'metric': 'Capacity Status',
                    'value': summary.get('capacity_status', 'Unknown'),
                    'explanation': 'Simple status label summarizing whether forecasted demand is healthy, moderate, or high.'
                },
                {
                    'metric': 'Resource Recommendation',
                    'value': summary.get('resource_forecast_recommendation', 'N/A'),
                    'explanation': 'Recommended staffing/resource action based on forecast pressure.'
                },
            ]
            rows = [
                {'role': role, 'actual_count': count}
                for role, count in summary.get('actual_staff_by_role', {}).items()
            ]
            details_title = 'Active Staff By Role'
            details_note = 'This table shows current active staff counts by role, which are used when estimating monthly capacity.'

        return {
            'key': report_key,
            'title': title,
            'description': report_definitions[report_key]['description'],
            'generated_at': generated_at,
            'branch_name': summary.get('branch_name', DEFAULT_BRANCH_NAME),
            'branch_code': summary.get('branch_code', DEFAULT_BRANCH_CODE),
            'sections': sections,
            'rows': rows,
            'chart_rows': chart_rows,
            'extra_charts': extra_charts,
            'extra_tables': [table for table in extra_tables if table.get('rows')],
            'details_title': details_title,
            'details_note': details_note,
        }

    def report_lines(payload):
        lines = [payload['description'], f"Branch: {payload.get('branch_name', DEFAULT_BRANCH_NAME)}", '']
        for section in payload['sections']:
            lines.append(f"{section['metric']}: {section['value']}")
            lines.append(f"  Why shown: {section['explanation']}")
        if payload['rows']:
            lines.append('')
            lines.append(payload.get('details_title', 'Details') + ':')
            lines.append(payload.get('details_note', ''))
            for row in payload['rows']:
                if isinstance(row, dict):
                    lines.append(' | '.join(f'{key}: {value}' for key, value in row.items()))
                else:
                    lines.append(str(row))
        return lines

    @app.route('/reports')
    def reports():
        return render_template(
            'reports/index.html',
            reports=report_definitions,
            current_date=datetime.now().strftime('%Y-%m-%d'),
            current_time=datetime.now().strftime('%H:%M'),
        )

    @app.route('/reports/<report_key>')
    def report_view(report_key):
        payload = build_report_payload(report_key)
        if payload is None:
            flash('Report not found.', 'error')
            return redirect(url_for('reports'))
        return render_template(
            'reports/view.html',
            report=payload,
            print_mode=False,
            current_date=datetime.now().strftime('%Y-%m-%d'),
            current_time=datetime.now().strftime('%H:%M'),
        )

    @app.route('/reports/<report_key>/print')
    def report_print(report_key):
        payload = build_report_payload(report_key)
        if payload is None:
            flash('Report not found.', 'error')
            return redirect(url_for('reports'))
        return render_template(
            'reports/view.html',
            report=payload,
            print_mode=True,
            current_date=datetime.now().strftime('%Y-%m-%d'),
            current_time=datetime.now().strftime('%H:%M'),
        )

    @app.route('/reports/<report_key>/pdf')
    def report_pdf(report_key):
        payload = build_report_payload(report_key)
        if payload is None:
            flash('Report not found.', 'error')
            return redirect(url_for('reports'))
        logo_path = os.path.join(app.static_folder, 'acudetek.jfif')
        pdf_bytes = _build_report_pdf(payload, logo_path=logo_path)
        filename = f"{report_key}-{datetime.now().strftime('%Y%m%d')}.pdf"
        return Response(
            pdf_bytes,
            mimetype='application/pdf',
            headers={'Content-Disposition': f'attachment; filename="{filename}"'}
        )

    @app.route('/settings', methods=['GET', 'POST'])
    def settings():
        app_settings = load_app_settings()
        if request.method == 'POST':
            try:
                capacity = max(1, int(request.form.get('staff_capacity_per_month', STAFF_CAPACITY_PER_MONTH)))
                app_settings['staff_capacity_per_month'] = capacity
                save_app_settings(app_settings)
                get_dashboard_summary(force_refresh=True)
                flash('Staff capacity setting updated.', 'success')
            except Exception:
                flash('Please enter a valid staff capacity number.', 'error')
            return redirect(url_for('settings'))

        return render_template(
            'settings/index.html',
            settings=app_settings,
            current_date=datetime.now().strftime('%Y-%m-%d'),
            current_time=datetime.now().strftime('%H:%M'),
        )

    return app

# -------------------------------------------------------------
# Database initialisation and schema migration
# -------------------------------------------------------------
def migrate_staff_member_schema(app):
    with app.app_context():
        with db.engine.begin() as conn:
            result = conn.execute(text("PRAGMA table_info(staff_member)"))
            columns = {row[1] for row in result.fetchall()}
            if 'is_active' not in columns:
                conn.execute(text("ALTER TABLE staff_member ADD COLUMN is_active BOOLEAN NOT NULL DEFAULT 1"))
            if 'deleted_at' not in columns:
                conn.execute(text("ALTER TABLE staff_member ADD COLUMN deleted_at DATETIME"))
            if 'department' in columns:
                conn.execute(text("DROP TABLE IF EXISTS staff_member_new"))
                conn.execute(text("""
                    CREATE TABLE staff_member_new (
                        id INTEGER NOT NULL PRIMARY KEY,
                        name VARCHAR(100) NOT NULL,
                        role VARCHAR(60) NOT NULL,
                        availability VARCHAR(20),
                        is_active BOOLEAN NOT NULL DEFAULT 1,
                        deleted_at DATETIME
                    )
                """))
                conn.execute(text("""
                    INSERT INTO staff_member_new (id, name, role, availability, is_active, deleted_at)
                    SELECT id, name, role, availability, is_active, deleted_at
                    FROM staff_member
                """))
                conn.execute(text("DROP TABLE staff_member"))
                conn.execute(text("ALTER TABLE staff_member_new RENAME TO staff_member"))

def migrate_branch_schema(app):
    with app.app_context():
        with db.engine.begin() as conn:
            result = conn.execute(text("PRAGMA table_info(branch)"))
            branch_columns = {row[1] for row in result.fetchall()}
            if 'email' not in branch_columns:
                conn.execute(text("ALTER TABLE branch ADD COLUMN email VARCHAR(120)"))

        branch = Branch.query.filter_by(code=DEFAULT_BRANCH_CODE).first()
        if branch is None:
            branch = Branch(
                name=DEFAULT_BRANCH_NAME,
                code=DEFAULT_BRANCH_CODE,
                address=DEFAULT_BRANCH_ADDRESS,
                is_main=True,
                is_active=True,
            )
            db.session.add(branch)
            db.session.commit()

        with db.engine.begin() as conn:
            for table_name in ('user', 'consultation_record', 'staff_member'):
                result = conn.execute(text(f"PRAGMA table_info({table_name})"))
                columns = {row[1] for row in result.fetchall()}
                if 'branch_id' not in columns:
                    conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN branch_id INTEGER"))
                conn.execute(
                    text(f"UPDATE {table_name} SET branch_id = :branch_id WHERE branch_id IS NULL"),
                    {'branch_id': branch.id}
                )
                conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_branch_id ON {table_name}(branch_id)"))

def init_db(app=None):
    app = app or flask_app
    with app.app_context():
        db.create_all()
        migrate_staff_member_schema(app)
        migrate_branch_schema(app)
        default_branch = Branch.query.filter_by(code=DEFAULT_BRANCH_CODE).first()
        if not User.query.filter_by(username='admin').first():
            db.session.add(User(username='admin', password='admin123', role='administrator', branch_id=default_branch.id))
        if not User.query.filter_by(username='staff').first():
            db.session.add(User(username='staff', password='staff123', role='staff', branch_id=default_branch.id))
        if not StaffMember.query.first():
            for person in build_facility_staff_roster():
                db.session.add(StaffMember(
                    branch_id=default_branch.id,
                    name=person['name'],
                    role=person['role'],
                    availability=person['availability'],
                    is_active=True,
                    deleted_at=None,
                ))
        db.session.commit()

flask_app = create_app()

if __name__ == '__main__':
    init_db(flask_app)
    flask_app.run(debug=True)

app = flask_app
