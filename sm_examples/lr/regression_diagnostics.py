"""
The code example are modified from below source to confirm my understanding.
Source: https://www.statsmodels.org/stable/index.html

Regression diagnostics
======================
a few of the statsmodels regression diagnostic tests

More information:
http://www.statsmodels.org/stable/diagnostic.html
"""
import statsmodels
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
import matplotlib.pyplot as plt
from statsmodels.compat import lzip
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.graphics.regressionplots import plot_leverage_resid2

def load_data(url):
    return pd.read_csv(url)

def fit_linear_model(formula, data):
    res = smf.ols(formula, data=data).fit()
    print(res.summary())
    return res

if __name__ == "__main__":
    url = 'http://vincentarelbundock.github.io/Rdatasets/csv/HistData/Guerry.csv'
    data = load_data(url)
    print(data.shape)
    print(data.head())

    formula = "Lottery ~ Literacy + np.log(Pop1831)"
    res = fit_linear_model(formula=formula, data=data)

    # Normality of the residuals
    print("\n * Jarque-Bera test:")
    name = ['Jarque-Bera', 
        'Chi^2 two-tail prob.', 
        'Skew', 
        'Kurtosis']
    test = sms.jarque_bera(res.resid)
    print(lzip(name, test))

    print("\n * Omni test: ")
    name = ['Chi^2', 
        'Two-tail probability']
    test = sms.omni_normtest(res.resid)
    print(lzip(name, test))

    # Multicollinearity
    print("\n * Condition number: ", np.linalg.cond(res.model.exog))

    # Heteroskedasticity tests
    print("\n * Breush-Pagan test:")
    name = ['Lagrange multiplier statistic', 
        'p-value', 
        'f-value', 
        'f p-value']
    test = sms.het_breushpagan(res.resid, res.model.exog)
    print(lzip(name, test))

    # Linearity
    # Harvey-Collier multiplier test for Null hypothesis 
    # that the linear specification is correct:
    print("\n * Harvey-Collier multiplier test:")
    name = ['t value', 
        'p value']
    test = sms.linear_harvey_collier(res)
    print(lzip(name, test))

    # Influence tests
    print("\n * Influence test")
    test_class = OLSInfluence(res)
    print("First few rows of DFbetas:")
    test_class.dfbetas[:5,:]
    # Influence plot
    fig, ax = plt.subplots(figsize=(8,6))
    fig = plot_leverage_resid2(res, ax = ax)
    plt.show()

