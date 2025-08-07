import pandas as pd
from pathlib import Path
from italian_holidays import italian_holidays

cur_path=Path(__file__).resolve().parent

project_root = cur_path.parent

df = pd.read_pickle(project_root / "data" / "pun_clean.pkl")

df['year'] = df.index.year  

df['month']=df.index.month

df['season'] = df['month'] % 12 // 3 + 1  # 1=winter, 2=spring, 3 = summer, 4 = Autumn

df['day']=df.index.day # day of the month

df['weekday']=df.index.weekday # day of the week (Monday = 0, Sunday = 6)

df['is_weekend']=df.index.weekday > 4

df['hour'] = df.index.hour

# Including italian holidays

holidays = italian_holidays()  

df['holiday'] = df.index.map(lambda x: holidays.is_holiday(x.to_pydatetime()))

df['is_holiday'] = df['holiday'].notna()

df.drop(columns='holiday', inplace=True)

# including lags of 1, 2, 3 and 7 days before

lags = [24, 48, 72, 168]

for lag in lags:
    df[f'lag_{lag}'] = df['PUN'].shift(lag)

daily_stats = df['PUN'].resample('D').agg(['mean', 'std'])

daily_stats.columns = ['1_day_mean', '1_day_std']

daily_stats['7_day_mean'] = daily_stats['1_day_mean'].shift(1).rolling(window=7).mean()

daily_stats['7_day_std'] = daily_stats['1_day_mean'].shift(1).rolling(window=7).std()

daily_stats['1_day_trend'] = daily_stats['1_day_mean'] - daily_stats['1_day_mean'].shift(1)

daily_stats['7_day_trend'] = daily_stats['1_day_mean'].shift(1) - daily_stats['7_day_mean']

daily_stats['trend_ratio'] = daily_stats['1_day_mean'].shift(1) / daily_stats['7_day_mean']

features = daily_stats.shift(1)

df['date'] = df.index.floor('D')

df = df.merge(features, left_on='date', right_index=True, how='left')

df.drop(columns='date', inplace=True)

df.dropna(inplace=True)

df.to_pickle(project_root / "data" / "processed_features.pkl")






