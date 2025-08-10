import pandas as pd
from pathlib import Path

cur_path=Path(__file__).resolve().parent

project_root = cur_path.parent

######################## PUN DATA ########################

df_pun=pd.read_csv(project_root / "data" / "raw_data" / "energy_pun_main_zones.csv")

df_pun['DATE']=pd.to_datetime(df_pun['DATE'],format='%Y%m%d')

df_pun['DATETIME'] = df_pun['DATE']+pd.to_timedelta(df_pun['HOUR'],unit='h')

df_pun.drop(columns=['DATE','HOUR','AUST','FRAN','GREC','SLOV','SVIZ','BSP','XAUS','XFRA','XGRE','IDX'],inplace=True)

df_pun=df_pun.set_index('DATETIME')

df_pun.to_pickle(project_root / "data" / "processed_data" / "pun_clean.pkl")

######################## DAY AHEAD ITALIAN LOAD FORECASTS (TERNA)  ########################

df_Terna = pd.read_excel(project_root / "data" / "raw_data" / "data.xlsx")

df_Terna = df_Terna.iloc[1:]

df_Terna.columns = df_Terna.iloc[0,:]

df_Terna = df_Terna.iloc[1:].reset_index(drop=True)

df_Terna = df_Terna.convert_dtypes()

df_Terna.rename(columns={'Date':'DATETIME'},inplace=True)

df_Terna.drop(columns=['Total Load (GW)'],inplace=True)

df_Terna = df_Terna.set_index('DATETIME')

df_Terna.groupby([pd.Grouper(freq='h')]).sum()

df_pun.to_pickle(project_root / "data" / "processed_data" / "Load_Forecasts.pkl")
