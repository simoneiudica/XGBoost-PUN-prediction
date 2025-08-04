import pandas as pd
from pathlib import Path
import os

data_path = Path("data/energy_pun_main_zones.csv")

df=pd.read_csv(data_path)

df['DATE']=pd.to_datetime(df['DATE'],format='%Y%m%d')

df['DATETIME'] = df['DATE']+pd.to_timedelta(df['HOUR'],unit='h')

df.drop(columns=['DATE','HOUR','AUST','FRAN','GREC','SLOV','SVIZ','BSP','XAUS','XFRA','XGRE','IDX'],inplace=True)

df=df.set_index('DATETIME')