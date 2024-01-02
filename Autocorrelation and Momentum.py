import pandas as pd 
import numpy as np
from Easy_Trading import Basic_funcs
import statsmodels.api as stat
import MetaTrader5 as mt5

#############                       Autocorrelation             ############# 

# Autocorrelation measures the correlation between a time series observation and
# its lagged values. It quantifies the linear relationship between an observation
# and its previous observations at different lags.

# ACF measures the overall correlation at each lag without considering the influence
# of intermediate lags. It helps identify the presence of significant patterns and
# trends in the data.

# ACF is useful for detecting seasonality, identifying the order of an autoregressive (AR)
# model, and determining the appropriate lag values for forecasting.

#############                       Partial Autocorrelation             ############# 

# Partial autocorrelation measures the direct correlation between an observation and
# its lagged values, while removing the indirect correlation through intermediate lags.

# PACF helps identify the specific lag(s) that directly influence an observation without
# the influence of other lags. It provides insights into the unique contribution of each
# lag to the current observation.

# PACF is useful for determining the order of a moving average (MA) model, identifying the
# presence of significant lags, and building autoregressive integrated moving average (ARIMA)
# models.

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
stat.graphics.tsa.plot_pacf(df_series_diff['log return'])

pacf_series = pd.Series(stat.tsa.stattools.pacf(df_series_diff['log return'])[1:])
pacf_img = pd.Series(stat.tsa.stattools.pacf(df_series_diff['log return'])[1:]).plot()
pacf_img.axhline(pacf_series.mean(),color = 'green')
pacf_img.axhline(pacf_series.mean() + 2*pacf_series.std(),color = 'red')
pacf_img.axhline(pacf_series.mean() - 2*pacf_series.std(),color = 'red')

acf_series = pd.Series(stat.tsa.stattools.acf(df_series_diff['log return'])[1:])
acf_img = pd.Series(stat.tsa.stattools.acf(df_series_diff['log return'])[1:]).plot()
acf_img.axhline(acf_series.mean(),color = 'green')
acf_img.axhline(acf_series.mean() + 2*acf_series.std(),color = 'red')
acf_img.axhline(acf_series.mean() - 2*acf_series.std(),color = 'red')



