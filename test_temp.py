import pandas as pd
from datetime import datetime


df = pd.read_csv("BTCUSDT-1m-data.csv")

print(int(df['close_time'].values[0]))

print(datetime.fromtimestamp(int(df['close_time'].values[-1])/1000))
