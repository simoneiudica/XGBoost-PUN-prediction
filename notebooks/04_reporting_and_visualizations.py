import pandas as pd 
from pathlib import Path
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cur_path=Path(__file__).resolve().parent

project_root = cur_path.parent

df_predictions = pd.read_pickle(project_root / "results" / "actual_vs_forecast.pkl")

df_results = pd.read_pickle(project_root / "results" / "training_summary.pkl")


def create_professional_plots(results_df, predictions_df, model_name="XGBoost"):
 
    
    # Set style professionale
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Performance over time
    ax1 = plt.subplot(3, 3, 1)
    results_df['date'] = pd.to_datetime(results_df['date'],format='%Y-%m-%d')
    plt.plot(results_df['date'], results_df['mae'], alpha=0.7, linewidth=1)
    plt.plot(results_df['date'], results_df['mae'].rolling(30).mean(), 
             color='red', linewidth=2, label='30-day MA')
    plt.title('MAE Performance Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('MAE (€/MWh)')
    plt.legend()
    plt.xticks(rotation=45)
    
    # 2. Errors distribution
    ax2 = plt.subplot(3, 3, 2)
    plt.hist(results_df['mae'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(results_df['mae'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    plt.axvline(results_df['mae'].median(), color='orange', linestyle='--', linewidth=2, label='Median')
    plt.title('MAE Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('MAE (€/MWh)')
    plt.ylabel('Frequency')
    plt.legend()
    
    # 3. Seasonal Performance 
    ax3 = plt.subplot(3, 3, 3)
    results_df['season'] = pd.to_datetime(results_df['date']).dt.month % 12 // 3 + 1
    seasonal_mae = results_df.groupby('season')['mae'].mean()
    season_names = ['Winter', 'Spring', 'Summer', 'Autumn']
    bars = plt.bar(range(1, 5), seasonal_mae.values, color=['lightblue', 'lightgreen', 'orange', 'brown'])
    plt.title('Seasonal Performance', fontsize=14, fontweight='bold')
    plt.xlabel('Season')
    plt.ylabel('Average MAE (€/MWh)')
    plt.xticks(range(1, 5), season_names)
    
    # Add values on bars
    for bar, value in zip(bars, seasonal_mae.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Actual vs Predicted (sample)
    ax4 = plt.subplot(3, 3, 4)
    sample_pred = predictions_df.sample(min(1000, len(predictions_df)))
    plt.scatter(sample_pred['actual'], sample_pred['predicted'], alpha=0.5)
    min_val = min(sample_pred['actual'].min(), sample_pred['predicted'].min())
    max_val = max(sample_pred['actual'].max(), sample_pred['predicted'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    plt.xlabel('Actual Price (€/MWh)')
    plt.ylabel('Predicted Price (€/MWh)')
    plt.title('Actual vs Predicted Prices', fontsize=14, fontweight='bold')
    
    # R² annotation
    r2 = r2_score(predictions_df['actual'], predictions_df['predicted'])
    plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
    
    # 5. Error distribution by hour
    ax5 = plt.subplot(3, 3, 5)
    predictions_df['hour'] = predictions_df.index.hour
    hourly_error = predictions_df.groupby('hour')['abs_error'].mean()
    plt.plot(hourly_error.index, hourly_error.values, marker='o', linewidth=2, markersize=4)
    plt.title('Average Error by Hour of Day', fontsize=14, fontweight='bold')
    plt.xlabel('Hour of Day')
    plt.ylabel('Mean Absolute Error (€/MWh)')
    plt.grid(True, alpha=0.3)
    
    # 6. Cumulative accuracy
    ax6 = plt.subplot(3, 3, 6)
    sorted_errors = np.sort(results_df['mae'])
    cumulative_pct = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
    plt.plot(sorted_errors, cumulative_pct, linewidth=2)
    plt.xlabel('MAE (€/MWh)')
    plt.ylabel('Cumulative Percentage')
    plt.title('Cumulative Error Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add benchmark lines
    for pct in [50, 80, 90, 95]:
        idx = int(len(sorted_errors) * pct / 100)
        plt.axvline(sorted_errors[idx], color='red', alpha=0.5, linestyle='--')
        plt.text(sorted_errors[idx], pct-5, f'{pct}%', rotation=90, ha='right')
    
    # 7. Monthly performance heatmap
    ax7 = plt.subplot(3, 3, 7)
    results_df['year'] = pd.to_datetime(results_df['date']).dt.year
    results_df['month'] = pd.to_datetime(results_df['date']).dt.month
    monthly_pivot = results_df.pivot_table(values='mae', index='month', columns='year', aggfunc='mean')
    sns.heatmap(monthly_pivot, cmap='RdYlBu_r', cbar_kws={'label': 'MAE (€/MWh)'})
    plt.title('Monthly MAE Heatmap', fontsize=14, fontweight='bold')
    plt.ylabel('Month')
    
    # 8. Performance metrics summary
    ax8 = plt.subplot(3, 3, 8)
    metrics = {
        'MAE': results_df['mae'].mean(),
        'RMSE': results_df['rmse'].mean(), 
        'MAE_std': results_df['mae'].std(),
        'Days < 5€': (results_df['mae'] < 5).mean() * 100,
        'Days < 10€': (results_df['mae'] < 10).mean() * 100
    }
    
    # Text summary
    text_content = f"""
MODEL PERFORMANCE SUMMARY

Mean Absolute Error: {metrics['MAE']:.2f} ± {metrics['MAE_std']:.2f} €/MWh
Root Mean Squared Error: {metrics['RMSE']:.2f} €/MWh

ACCURACY DISTRIBUTION:
• Days with MAE < 5€/MWh: {metrics['Days < 5€']:.1f}%
• Days with MAE < 10€/MWh: {metrics['Days < 10€']:.1f}%

PERIOD: {results_df['date'].min().strftime('%Y-%m-%d')} to {results_df['date'].max().strftime('%Y-%m-%d')}
TOTAL DAYS: {len(results_df):,}

MODEL: {model_name}
"""
    plt.text(0.05, 0.95, text_content, transform=ax8.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
    plt.axis('off')
    
    # 9. Recent performance (last year)
    ax9 = plt.subplot(3, 3, 9)
    recent_results = results_df[results_df['date'] >= results_df['date'].max() - pd.Timedelta(days=365)]
    if len(recent_results) > 0:
        plt.plot(recent_results['date'], recent_results['mae'], alpha=0.7, linewidth=1.5)
        plt.title('Recent Performance (Last Year)', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('MAE (€/MWh)')
        plt.xticks(rotation=45)
        
        # Highlight worst days
        worst_recent = recent_results.nlargest(3, 'mae')
        plt.scatter(worst_recent['date'], worst_recent['mae'], color='red', s=50, zorder=5)
    
    plt.suptitle(f'{model_name} - Day-Ahead Price Forecasting Analysis', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()

    return fig

dashboard = create_professional_plots(df_results,df_predictions)

dashboard.savefig(project_root / "reports" / "dashboard.png", dpi=300, bbox_inches='tight')

plt.close(dashboard)