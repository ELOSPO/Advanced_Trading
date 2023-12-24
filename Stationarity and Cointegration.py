### Arbitraje Estadístico Capitulo 1: Cointegración y Estacionareidad ###

# Una serie de tiempo tiende a retornar a la media si es estacionaria. Emtonces es un buen concepto para usarlo en trading

# Dick and Fuller Test
# H0 No es Estacionaria: (Buscamos rechazar H0 es decir que el p_valor < 0.05 )

import statsmodels.api as stat
import statsmodels.tsa.stattools as ts
import MetaTrader5 as mt5
import pandas as pd 
import numpy as np
from Easy_Trading import Basic_funcs
import itertools
from scipy.stats import pearsonr

nombre = 67043467
clave = 'Genttly.2022'
servidor = 'RoboForex-ECN'
path = r'C:\Program Files\RoboForex - MetaTrader 5\terminal64.exe'

bfs = Basic_funcs(nombre, clave, servidor, path)

def test_cointegration(serie_a, serie_b,p_value):
    model = stat.OLS(serie_a,serie_b).fit()
    coint_res = ts.adfuller(model.resid)
    if coint_res[1] <= p_value:
        return True,model
    else:
        return False,model

pair_a = 'EURUSD'
pair_b = 'GBPUSD'



list_of_symb = ['EURUSD','USDCAD','GBPUSD','XAUUSD','CADJPY','EURAUD','USDCHF','AUDUSD',
                'BTCUSD','ETHUSD','AUDNZD','AUDCAD','GBPCAD','GBPNZD','USDJPY']

# combinaciones de las monedas

lists = list(itertools.combinations(list_of_symb, 2))

list_of_a = []
list_of_b = []
result_coint = []
result_corr = []


for list_symb in lists:
    s_a = list_symb[0]
    s_b = list_symb[1]
    series_a = bfs.extract_data(s_a,mt5.TIMEFRAME_M1,9999)['close']
    series_b = bfs.extract_data(s_b,mt5.TIMEFRAME_M1,9999)['close']
    print(f'Combinación {s_a} y {s_b}')
    list_of_a.append(s_a)
    list_of_b.append(s_b)
    result_coint.append(test_cointegration(series_a, series_b,0.1)[0])
    result_corr.append(pearsonr(series_a,series_b)[0])

results_df = pd.DataFrame(zip(list_of_a,list_of_b,result_coint,result_corr),
                          columns = ['asset_1','asset_2','cointegrated','p_corr_coef'])

### Graficamos los residuales ###

series_b = bfs.extract_data('GBPCAD',mt5.TIMEFRAME_M1,9999)['close']
series_a = bfs.extract_data('GBPNZD',mt5.TIMEFRAME_M1,9999)['close']

res_adf, model = test_cointegration(series_a, series_b,0.1)
residuals = model.resid
g1 = residuals.plot()
g1.axhline(residuals.mean(),color = 'red')


residuals.mean()
residuals.hist(bins = 40)
