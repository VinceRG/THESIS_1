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
    availability = db.Column(db.String(20), default='Available')
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    deleted_at = db.Column(db.DateTime, nullable=True)

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

def build_forecasting_training_frame(df):
    """Build one diagnosis-month row per observed month, with honest lag features."""
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
    full_index = pd.MultiIndex.from_product([diagnoses, periods], names=['diagnosis', 'period'])

    grouped = raw.groupby(['diagnosis', 'period']).size().rename('case_count')
    merged = grouped.reindex(full_index, fill_value=0).reset_index()
    merged['period_start'] = merged['period'].dt.to_timestamp()
    merged['year'] = merged['period_start'].dt.year
    merged['month'] = merged['period_start'].dt.month
    merged['season'] = (merged['month'] - 1) // 3 + 1
    merged['time_index'] = (merged['year'] - merged['year'].min()) * 12 + merged['month']
    merged = merged.sort_values(['diagnosis', 'period']).reset_index(drop=True)

    merged['lag_1'] = merged.groupby('diagnosis')['case_count'].shift(1)
    merged['lag_2'] = merged.groupby('diagnosis')['case_count'].shift(2)
    merged['lag_3'] = merged.groupby('diagnosis')['case_count'].shift(3)
    merged['lag_6'] = merged.groupby('diagnosis')['case_count'].shift(6)
    merged['lag_12'] = merged.groupby('diagnosis')['case_count'].shift(12)
    merged['rolling_mean_3'] = merged.groupby('diagnosis')['case_count'].transform(
        lambda x: x.rolling(3, min_periods=1).mean().shift(1)
    )
    merged['rolling_mean_6'] = merged.groupby('diagnosis')['case_count'].transform(
        lambda x: x.rolling(6, min_periods=1).mean().shift(1)
    )
    merged['rolling_std_3'] = merged.groupby('diagnosis')['case_count'].transform(
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
    merged = pd.concat([merged, diagnosis_dummies], axis=1)
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
    feature_columns = [
        'month', 'season', 'time_index',
        'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12',
        'rolling_mean_3', 'rolling_mean_6', 'rolling_std_3', 'trend_1',
    ] + diagnosis_feature_columns

    training_df = training_df.sort_values(['period_start', 'diagnosis']).reset_index(drop=True)
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
    cv_splits = min(3, max(2, len(train_df) // max(1, len(diagnosis_encoder.classes_) * 3)))
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
    metrics['model_verdict'] = _model_verdict(metrics)
    return final_model, metrics, feature_columns, label_mapping

def generate_forecast_for_month(model, feature_columns, label_mapping, df, target_month, target_year):
    training_df = build_forecasting_training_frame(df)
    if training_df.empty:
        return []

    target_period = pd.Period(year=target_year, month=target_month, freq='M')
    next_rows = []
    diagnosis_columns = [
        col for col in feature_columns
        if col.startswith('diagnosis_') and col != 'diagnosis_code'
    ]

    for diagnosis, diag_data in training_df.groupby('diagnosis'):
        history = diag_data.sort_values('period').copy()
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
        }
        for col in diagnosis_columns:
            row[col] = 1 if col == f'diagnosis_{diagnosis}' else 0
        next_rows.append(row)

    pred_df = pd.DataFrame(next_rows)
    preds = model.predict(pred_df[feature_columns])
    results = []
    for i, diag_enc in enumerate(pred_df['diagnosis_code']):
        diag_name = label_mapping.get(diag_enc, f"Unknown_{diag_enc}")
        results.append((diag_name, max(0, round(preds[i]))))
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def write_training_report(report_path, metrics):
    with open(report_path, 'w', encoding='utf-8') as handle:
        handle.write('Smart Healthcare Clinic Management - Training Report\n')
        handle.write('===================================================\n')
        handle.write(f"Model Verdict: {metrics.get('model_verdict', 'Unknown')}\n")
        handle.write(f"Validation Period: {metrics.get('validation_period_start')} to {metrics.get('validation_period_end')}\n")
        handle.write(f"Training Months: {metrics.get('training_months')}\n")
        handle.write(f"Validation Months: {metrics.get('validation_months')}\n")
        handle.write(f"Validation R2: {metrics.get('validation_r2')}\n")
        handle.write(f"Validation MAE: {metrics.get('validation_mae')}\n")
        handle.write(f"Validation RMSE: {metrics.get('validation_rmse')}\n")
        handle.write(f"Baseline MAE (last month): {metrics.get('baseline_mae')}\n")
        handle.write(f"Baseline RMSE (last month): {metrics.get('baseline_rmse')}\n")
        handle.write(f"Improvement vs Baseline: {metrics.get('improvement_vs_baseline_pct')}%\n")
        handle.write(f"CV R2 Mean: {metrics.get('cv_r2_mean')} (+/-{metrics.get('cv_r2_std')})\n")
        handle.write(f"CV MAE Mean: {metrics.get('cv_mae_mean')}\n")
        handle.write(f"Training R2 (in-sample only): {metrics.get('training_r2')}\n")
        handle.write(f"Best Params: {metrics.get('best_params')}\n")

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
    colors = [
        (0.059, 0.357, 0.541),
        (0.169, 0.655, 0.773),
        (0.184, 0.620, 0.267),
        (0.851, 0.467, 0.024),
        (0.486, 0.227, 0.929),
        (0.863, 0.149, 0.149),
    ]

    commands = [
        f'{light} rg 0 0 {width} {height} re f',
        f'{primary} rg 0 760 {width} 82 re f',
        f'{accent} rg 0 754 {width} 6 re f',
    ]

    image_info = _jpeg_size(logo_path) if logo_path else None
    image_obj_num = None
    if image_info:
        commands.append('q 46 0 0 46 42 778 cm /Logo Do Q')
        title_x = 100
    else:
        commands.append('1 1 1 rg 42 778 46 46 re f')
        commands.append(_pdf_text(54, 795, 'A', 20, primary, 'F2'))
        title_x = 100

    commands.extend([
        _pdf_text(title_x, 805, 'Accudetek Health Diagnostics', 15, '1 1 1', 'F2'),
        _pdf_text(title_x, 786, payload['title'], 11, '1 1 1'),
        _pdf_text(420, 805, 'Generated', 9, '0.860 0.945 0.969'),
        _pdf_text(420, 789, payload['generated_at'], 11, '1 1 1', 'F2'),
        _pdf_text(42, 727, payload['description'], 11, text),
        f'1 1 1 rg 42 552 528 150 re f',
        f'{border} RG 42 552 528 150 re S',
        f'{primary} rg 42 676 528 26 re f',
        _pdf_text(56, 684, 'Metric', 10, '1 1 1', 'F2'),
        _pdf_text(390, 684, 'Value', 10, '1 1 1', 'F2'),
    ])

    row_y = 652
    for section in payload['sections'][:6]:
        metric = str(section.get('metric', ''))
        if len(metric) > 54:
            metric = metric[:51] + '...'
        commands.append(f'{border} RG 42 {row_y - 8} 528 1 re f')
        commands.append(_pdf_text(56, row_y, metric, 9, text, 'F2'))
        commands.append(_pdf_text(390, row_y, section.get('value', ''), 10, primary, 'F2'))
        row_y -= 22

    chart_top = 505
    commands.extend([
        _pdf_text(42, 524, payload.get('details_title', 'Details'), 14, text, 'F2'),
        _pdf_text(42, 508, payload.get('details_note', '')[:105], 8, muted),
    ])

    if payload.get('key') in {'monthly-consultation', 'quarterly'} and payload.get('chart_rows', {}).get('datasets'):
        left, bottom, chart_w, chart_h = 64, 222, 480, 235
        commands.extend([
            f'1 1 1 rg 42 188 528 300 re f',
            f'{border} RG 42 188 528 300 re S',
            f'{border} RG {left} {bottom} {chart_w} {chart_h} re S',
        ])
        labels = payload['chart_rows']['labels']
        datasets = payload['chart_rows']['datasets']
        max_value = max([max(item['data']) for item in datasets] + [1])
        y_max = max(10, int(np.ceil(max_value / 10.0) * 10))

        for step in range(0, 6):
            y = bottom + (chart_h * step / 5)
            value = int(y_max * step / 5)
            commands.append(f'0.906 0.925 0.953 RG {left} {y:.2f} m {left + chart_w} {y:.2f} l S')
            commands.append(_pdf_text(42, y - 3, value, 7, muted))

        label_divisor = max(1, len(labels) - 1)
        for index, label in enumerate(labels):
            x = left + (chart_w * index / label_divisor)
            commands.append(f'{border} RG {x:.2f} {bottom} m {x:.2f} {bottom - 4} l S')
            commands.append(_pdf_text(x - 8, bottom - 18, label, 7, muted))

        legend_x = left
        legend_y = bottom + chart_h + 18
        for ds_index, dataset in enumerate(datasets):
            r, g, b = colors[ds_index % len(colors)]
            commands.append(f'{r} {g} {b} rg {legend_x} {legend_y - 2} 14 4 re f')
            commands.append(_pdf_text(legend_x + 19, legend_y - 5, dataset['year'], 8, text))
            legend_x += 70

            points = []
            point_divisor = max(1, len(dataset['data']) - 1)
            for point_index, value in enumerate(dataset['data']):
                x = left + (chart_w * point_index / point_divisor)
                y = bottom + (chart_h * (value / y_max))
                points.append((x, y))
            if points:
                path = [f'{r} {g} {b} RG 2 w {points[0][0]:.2f} {points[0][1]:.2f} m']
                path.extend(f'{x:.2f} {y:.2f} l' for x, y in points[1:])
                path.append('S')
                commands.append(' '.join(path))
                for x, y in points:
                    commands.append(f'{r} {g} {b} rg {x - 2:.2f} {y - 2:.2f} 4 4 re f')
    elif payload.get('rows'):
        y = 470
        for row in payload['rows'][:10]:
            commands.append(_pdf_text(52, y, ' | '.join(f'{k}: {v}' for k, v in row.items())[:110], 8, text))
            y -= 16

    commands.append(_pdf_text(42, 42, 'Smart Healthcare Clinic Management', 8, muted))
    content = '\n'.join(commands).encode('latin-1', errors='replace')

    objects = [
        (1, b'<< /Type /Catalog /Pages 2 0 R >>'),
        (2, b'<< /Type /Pages /Kids [4 0 R] /Count 1 >>'),
        (3, b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>'),
        (5, b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>'),
    ]
    resources = '<< /Font << /F1 3 0 R /F2 5 0 R >>'
    next_obj = 6
    if image_info:
        img_w, img_h, img_data = image_info
        image_obj_num = next_obj
        next_obj += 1
        objects.append((
            image_obj_num,
            f'<< /Type /XObject /Subtype /Image /Width {img_w} /Height {img_h} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length {len(img_data)} >>\nstream\n'.encode('ascii') + img_data + b'\nendstream'
        ))
        resources += f' /XObject << /Logo {image_obj_num} 0 R >>'
    resources += ' >>'
    objects.append((next_obj, b'<< /Length ' + str(len(content)).encode('ascii') + b' >>\nstream\n' + content + b'\nendstream'))
    content_obj = next_obj
    objects.append((4, f'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] /Resources {resources} /Contents {content_obj} 0 R >>'.encode('ascii')))
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
    cache_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'dashboard_summary.json')
    settings_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'app_settings.json')
    dashboard_cache_version = 3

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

    def load_cached_dashboard_summary():
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

    def cache_dashboard_summary(summary):
        try:
            summary['cache_version'] = dashboard_cache_version
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
        facility_staff_count = sum(item['count'] for item in FACILITY_STAFF_COMPLEMENT)
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
            'forecast_pressure_raw': forecast_pressure_raw,
            'recommended_staff_count': recommended_staff_count,
            'staff_gap': staff_gap,
            'capacity_status': capacity_status,
            'resource_forecast_recommendation': resource_forecast_recommendation,
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
        protected_endpoints = {
            'dashboard', 'records', 'upload', 'predict', 'staff', 'reports',
            'report_view', 'report_print', 'report_pdf',
            'permanent_delete_staff',
            'settings', 'resources'
        }
        if request.endpoint in protected_endpoints and 'user_id' not in session:
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

    @app.route('/records/clear', methods=['POST'])
    def clear_records():
        deleted_count = ConsultationRecord.query.delete()
        db.session.commit()
        get_dashboard_summary(force_refresh=True)
        flash(f'Cleared {deleted_count} consultation records.', 'success')
        return redirect(url_for('records'))

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
                for record in ConsultationRecord.query.all()
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
        expired_staff = StaffMember.query.filter(
            StaffMember.is_active.is_(False),
            StaffMember.deleted_at.isnot(None),
            StaffMember.deleted_at < cutoff,
        ).all()
        for member in expired_staff:
            db.session.delete(member)
        if expired_staff:
            db.session.commit()

        active_staff = StaffMember.query.filter_by(is_active=True).all()
        inactive_staff = StaffMember.query.filter_by(is_active=False).all()
        return render_template('staff/index.html', active_staff=active_staff, inactive_staff=inactive_staff)

    staff_role_options = [item['role'] for item in FACILITY_STAFF_COMPLEMENT]
    staff_availability_options = ['Available', 'Busy', 'On Leave', 'Unavailable']

    @app.route('/staff/load-facility-complement', methods=['POST'])
    def load_facility_complement():
        StaffMember.query.filter_by(is_active=True).delete()
        for person in build_facility_staff_roster():
            db.session.add(StaffMember(
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
        if request.method == 'POST':
            new_staff = StaffMember(
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
        staff_member = StaffMember.query.get_or_404(staff_id)
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
        staff_member = StaffMember.query.get_or_404(staff_id)
        staff_member.is_active = False
        staff_member.deleted_at = datetime.now()
        db.session.commit()
        get_dashboard_summary(force_refresh=True)
        flash('Staff member removed from active roster (soft deleted).', 'success')
        return redirect(url_for('staff'))

    @app.route('/staff/<int:staff_id>/restore', methods=['POST'])
    def restore_staff(staff_id):
        staff_member = StaffMember.query.get_or_404(staff_id)
        staff_member.is_active = True
        staff_member.deleted_at = None
        db.session.commit()
        get_dashboard_summary(force_refresh=True)
        flash('Staff member restored to active roster.', 'success')
        return redirect(url_for('staff'))

    @app.route('/staff/<int:staff_id>/permanent-delete', methods=['POST'])
    def permanent_delete_staff(staff_id):
        staff_member = StaffMember.query.get_or_404(staff_id)
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
        records = ConsultationRecord.query.all()
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
        details_title = 'Details'
        details_note = 'Supporting rows used for this report.'

        if not df.empty:
            df['consultation_date'] = pd.to_datetime(df['consultation_date'], errors='coerce')
            df = df.dropna(subset=['consultation_date'])

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
            rows = monthly_rows
            details_title = 'Monthly Consultation History'
            details_note = 'This table shows actual historical consultation totals per month. It is shown so the clinic can see recent demand patterns and compare them with the latest forecast.'

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
            rows = quarterly_rows
            details_title = 'Quarterly Consultation History'
            details_note = 'This graph has four points per year, one for each quarter. Each colored line represents a year so quarter-to-quarter changes are easier to compare.'

        elif report_key == 'prediction':
            forecast = summary.get('predictions', [])
            metrics = summary.get('rf_metrics') or {}
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
            'sections': sections,
            'rows': rows,
            'chart_rows': chart_rows,
            'details_title': details_title,
            'details_note': details_note,
        }

    def report_lines(payload):
        lines = [payload['description'], '']
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
            for person in build_facility_staff_roster():
                db.session.add(StaffMember(
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
