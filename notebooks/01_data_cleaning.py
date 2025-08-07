import pandas as pd
from pathlib import Path

cur_path=Path(__file__).resolve().parent

project_root = cur_path.parent

data_path = project_root / "data" / "energy_pun_main_zones.csv"

df=pd.read_csv(data_path)

df['DATE']=pd.to_datetime(df['DATE'],format='%Y%m%d')

df['DATETIME'] = df['DATE']+pd.to_timedelta(df['HOUR'],unit='h')

df.drop(columns=['DATE','HOUR','AUST','FRAN','GREC','SLOV','SVIZ','BSP','XAUS','XFRA','XGRE','IDX'],inplace=True)

df=df.set_index('DATETIME')

df.to_pickle(project_root / "data" / "pun_clean.pkl")
