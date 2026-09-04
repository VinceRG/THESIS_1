"""Compare the production Random Forest forecasting model against a Decision
Tree baseline, trained on the identical feature set and time-based
train/validation split, for the thesis model-comparison table.

Usage:
    .venv/Scripts/python.exe scripts/compare_forecasting_models.py [branch_id]

Reuses app.build_forecasting_training_frame() so the feature engineering
(lags, rolling stats, one-hot diagnosis/age/gender, complete-year filtering,
rare-diagnosis grouping) exactly matches what the live Random Forest model
trains on -- only the estimator changes between rows.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

import app as appmod


def _regression_metrics(y_true, preds):
    return appmod._regression_metrics(y_true, preds)


def main():
    branch_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    flask_app = appmod.app
    with flask_app.app_context():
        ConsultationRecord = appmod.ConsultationRecord
        records = ConsultationRecord.query.filter_by(branch_id=branch_id).all()
        if not records:
            print(f'No consultation records found for branch_id={branch_id}')
            return

        df = pd.DataFrame([{
            'consultation_date': r.consultation_date,
            'age_group': r.age_group,
            'gender': r.gender,
            'diagnosis': r.diagnosis,
            'department': r.department,
            'physician': r.physician,
            'consultation_type': r.consultation_type,
        } for r in records])

        rf_model, rf_metrics, feature_columns, _ = appmod.train_and_evaluate_model(
            df, fast=False, compute_model_b=False
        )

        training_df = appmod.build_forecasting_training_frame(df)
        training_df = training_df.sort_values(
            ['period_start', 'diagnosis', 'age_group', 'gender']
        ).reset_index(drop=True)
        unique_periods = sorted(training_df['period'].unique())
        holdout_months = min(6, max(2, len(unique_periods) // 4))
        validation_periods = unique_periods[-holdout_months:]
        train_df = training_df[~training_df['period'].isin(validation_periods)]
        validation_df = training_df[training_df['period'].isin(validation_periods)]

        X_train = train_df[feature_columns]
        y_train = train_df['case_count']
        X_validation = validation_df[feature_columns]
        y_validation = validation_df['case_count']

        results = [('Random Forest', rf_metrics['validation_r2'], rf_metrics['validation_mae'], rf_metrics['validation_rmse'])]

        best_params = rf_metrics['best_params']
        tree = DecisionTreeRegressor(
            max_depth=best_params.get('max_depth'),
            min_samples_split=best_params.get('min_samples_split'),
            min_samples_leaf=best_params.get('min_samples_leaf'),
            max_features=best_params.get('max_features'),
            random_state=42,
        )
        tree.fit(X_train, y_train)
        tree_preds = np.maximum(0, tree.predict(X_validation))
        tree_metrics = _regression_metrics(y_validation, tree_preds)
        results.append(('Decision Tree', tree_metrics['r2'], tree_metrics['mae'], tree_metrics['rmse']))

        print(f"Validation period: {rf_metrics['validation_period_start']} - {rf_metrics['validation_period_end']}")
        print(f"Training rows: {len(train_df)}  Validation rows: {len(validation_df)}\n")
        print(f"{'Model':<20}{'R2':>10}{'MAE':>10}{'RMSE':>10}")
        for name, r2, mae, rmse in results:
            r2_str = f'{r2:.4f}' if r2 is not None else 'N/A'
            print(f'{name:<20}{r2_str:>10}{mae:>10.4f}{rmse:>10.4f}')


if __name__ == '__main__':
    main()
