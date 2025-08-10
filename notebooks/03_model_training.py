from xgboost import XGBRegressor
from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')
import numpy as np


def run_backtesting(df, feature_cols, target_col, 
                   train_days=365, test_days=1, step_days=1,
                   retrain_freq=7, 
                   xgb_params=None):
    """
    Backtesting for day-ahead 
    
    Args:
        df: DataFrame with processed features
        feature_cols: list of features
        target_col: target column
        train_days: training days
        test_days: test day
        step_days: step between consecutive tests
        retrain_freq: retrain frequency for efficiency
        xgb_params: parameters for XGBoost
    """
    
    if xgb_params is None:
        xgb_params = {
            'n_estimators': 200,
            'learning_rate': 0.1,
            'max_depth': 6,
            'random_state': 42,
            'n_jobs': -1
        }
    
    results = []
    predictions = []
    model = None
    
    start_idx = train_days * 24  
    step_hours = step_days * 24
    test_hours = test_days * 24
    
    print(f"Start backtesting: {len(df)} ore totali")
    print(f"Training: {train_days} giorni ({train_days * 24} ore)")
    print(f"Test: {test_days} giorni per volta")
    print(f"Retraining each {retrain_freq} step")
    
    step_count = 0
    
    for i in range(start_idx, len(df) - test_hours + 1, step_hours):
        
        # rolling windows
        train_start = i - train_days * 24
        train_end = i
        test_start = i  
        test_end = i + test_hours
        
        # data extraction 
        train = df.iloc[train_start:train_end]
        test = df.iloc[test_start:test_end]
        
        # drop nan values
        train_clean = train.dropna()
        test_clean = test.dropna()
        
        if len(train_clean) < 24 or len(test_clean) == 0:
            print(f"Skip step {step_count}: not enough data available")
            continue
            
        X_train = train_clean[feature_cols]
        y_train = train_clean[target_col]
        X_test = test_clean[feature_cols]
        y_test = test_clean[target_col]
        
        # Retrain to improve efficiency
        if model is None or step_count % retrain_freq == 0:
            print(f"Retrainig at step {step_count} ({test.index[0].strftime('%Y-%m-%d')})")
            
            model = XGBRegressor(**xgb_params)
            model.fit(X_train, y_train)
        
        # Predictions
        preds = model.predict(X_test)
        
        # Error metrics
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        
        
        # Save results
        result = {
            'step': step_count,
            'date': test.index[0].date(),
            'start_time': test.index[0],
            'end_time': test.index[-1], 
            'mae': mae,
            'rmse': rmse,
            'mean_actual': y_test.mean(),
            'mean_predicted': preds.mean(),
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }
        results.append(result)
        
        # Save predictions
        pred_detail = pd.DataFrame({
            'datetime': test_clean.index,
            'actual': y_test.values,
            'predicted': preds,
            'error': y_test.values - preds,
            'abs_error': np.abs(y_test.values - preds)
        })
        predictions.append(pred_detail)
        
        step_count += 1
        
        # Progress each 30 step
        if step_count % 30 == 0:
            print(f"Completed step {step_count}, last date: {test.index[0].strftime('%Y-%m-%d')}")
    
    results_df = pd.DataFrame(results)
    predictions_df = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    
    print(f"\n=== BACKTESTING COMPLETED ===")
    print(f"Total steps: {len(results_df)}")
    print(f"Period: {results_df['date'].min()} - {results_df['date'].max()}")
    
    return results_df, predictions_df



cur_path=Path(__file__).resolve().parent

project_root = cur_path.parent

df = pd.read_pickle(project_root / "data" / "processed_data" / "processed_features.pkl")


feature_cols = ['year', 'month', 'season', 'day', 'weekday', 'is_weekend', 
                'hour', 'is_holiday', 'lag_24', 'lag_48', 'lag_72', 'lag_168',
                '1_day_mean', '1_day_std', '7_day_mean', '7_day_std', 
                '1_day_trend', '7_day_trend', 'trend_ratio']

target_col = 'PUN'

results, predictions = run_backtesting(
    df, feature_cols, target_col,
    train_days=365,
    test_days=1,
    step_days=1,
    retrain_freq=7  
)

def analyze_results(results_df):
    """Analysis of the results"""
    
    print("\n=== PERFORMANCE OVERVIEW ===")
    print(f"MAE: {results_df['mae'].mean():.2f} ± {results_df['mae'].std():.2f}")
    print(f"RMSE: {results_df['rmse'].mean():.2f} ± {results_df['rmse'].std():.2f}")
    
    # Seasonal Performance
    if len(results_df) > 50:
        results_df['month'] = pd.to_datetime(results_df['date']).dt.month
        results_df['season'] = results_df['month'] % 12 // 3 + 1
        
        seasonal_perf = results_df.groupby('season')['mae'].agg(['mean', 'std', 'count'])
        print("\n=== SEASONAL PERFORMANCE ===")
        print("Season: 1=Winter, 2=Spring, 3=Summer, 4=Autumn")
        print(seasonal_perf.round(2))
    
    # Worst/Best days
    print(f"\n=== WORST DAYS (MAE) ===")
    worst_days = results_df.nlargest(5, 'mae')[['date', 'mae', 'mape']]
    print(worst_days.round(2))
    
    print(f"\n=== BEST DAYS (MAE) ===") 
    best_days = results_df.nsmallest(5, 'mae')[['date', 'mae', 'mape']]
    print(best_days.round(2))
    
    return results_df.describe()

summary = analyze_results(results)

results.to_pickle(project_root / "results" / "training_summary.pkl")

predictions.to_pickle(project_root / "results" / "actual_vs_forecast.pkl")

