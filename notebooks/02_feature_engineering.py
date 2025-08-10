import pandas as pd
from pathlib import Path
from italian_holidays import italian_holidays

cur_path=Path(__file__).resolve().parent

project_root = cur_path.parent

######################## PUN DATA ########################

df_pun = pd.read_pickle(project_root / "data" / "processed_data" / "pun_clean.pkl")

df_pun['year'] = df_pun.index.year  

df_pun['month']=df_pun.index.month

df_pun['season'] = df_pun['month'] % 12 // 3 + 1  # 1=winter, 2=spring, 3 = summer, 4 = Autumn

df_pun['day']=df_pun.index.day # day of the month

df_pun['weekday']=df_pun.index.weekday # day of the week (Monday = 0, Sunday = 6)

df_pun['is_weekend']=df_pun.index.weekday > 4

df_pun['hour'] = df_pun.index.hour

# Including italian holidays

holidays = italian_holidays()  

df_pun['holiday'] = df_pun.index.map(lambda x: holidays.is_holiday(x.to_pydatetime()))

df_pun['is_holiday'] = df_pun['holiday'].notna()

df_pun.drop(columns='holiday', inplace=True)

# including lags of 1, 2, 3 and 7 days before

lags = [24, 48, 72, 168]

for lag in lags:
    df_pun[f'lag_{lag}'] = df_pun['PUN'].shift(lag)

daily_stats = df_pun['PUN'].resample('D').agg(['mean', 'std'])

daily_stats.columns = ['1_day_mean', '1_day_std']

daily_stats['7_day_mean'] = daily_stats['1_day_mean'].shift(1).rolling(window=7).mean()

daily_stats['7_day_std'] = daily_stats['1_day_mean'].shift(1).rolling(window=7).std()

daily_stats['1_day_trend'] = daily_stats['1_day_mean'] - daily_stats['1_day_mean'].shift(1)

daily_stats['7_day_trend'] = daily_stats['1_day_mean'].shift(1) - daily_stats['7_day_mean']

daily_stats['trend_ratio'] = daily_stats['1_day_mean'].shift(1) / daily_stats['7_day_mean']

features = daily_stats.shift(1)

df_pun['date'] = df_pun.index.floor('D')

df_pun = df_pun.merge(features, left_on='date', right_index=True, how='left')

df_pun.drop(columns='date', inplace=True)

df_pun.dropna(inplace=True)

df_pun.to_pickle(project_root / "data" / "processed_data" / "processed_features.pkl")






