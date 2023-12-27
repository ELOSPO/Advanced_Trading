import pandas as pd 
import numpy as np
from Easy_Trading import Basic_funcs
import statsmodels.api as stat
import MetaTrader5 as mt5

nombre = 67043467
clave = 'Genttly.2022'
servidor = 'RoboForex-ECN'
path = r'C:\Program Files\RoboForex - MetaTrader 5\terminal64.exe'

bfs = Basic_funcs(nombre, clave, servidor, path)

series_a = bfs.extract_data('GBPUSD',mt5.TIMEFRAME_H4,1000)[['time','close']]

## Sin diferenciar ##
df_series = series_a.set_index('time')
stat.graphics.tsa.plot_acf(df_series)
stat.graphics.tsa.plot_pacf(df_series)

## Diferenciando ##

series_a['log return'] = np.log(series_a['close']).diff()
df_series_diff = series_a.set_index('time').dropna()
stat.graphics.tsa.plot_acf(df_series_diff['log return'])
stat.graphics.tsa.plot_pacf(df_series_diff['log return'],lags=10)

stat.tsa.stattools.pacf(df_series_diff['log return'],nlags=1)

