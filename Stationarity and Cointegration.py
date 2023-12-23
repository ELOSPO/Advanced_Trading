### Arbitraje Estadístico Capitulo 1: Cointegración y Estacionareidad ###

# Una serie de tiempo tiende a retornar a la media si es estacionaria. Emtonces es un buen concepto para usarlo en trading

# Dick and Fuller Test
# H0 No es Estacionaria: (Buscamos rechazar H0 es decir que el p_valor < 0.05 )

import statsmodels.api as stat
import statsmodels.tsa.stattools as ts

def test_cointegration(serie_a, serie_b,p_value):
    model = stat.OLS(serie_a,serie_b)
    coint_res = ts.adfuller(model.resid)
    if coint_res[1] <= p_value:
        return True
    else:
        return False
