"""
The code example are modified from below source to confirm my understanding.
Source: https://www.statsmodels.org/stable/index.html

Analysis of Variance (ANOVA)
============================
Analyze the differences among group means in a sample.

In its simplest form, ANOVA provides a statistical test of whether the population means of several groups 
are equal, and therefore generalizes the t-test to more than two groups. 

Source: https://en.wikipedia.org/wiki/Analysis_of_variance
Source: https://web.stanford.edu/class/stats191/notebooks/Interactions.html
"""
import statsmodels.api as sm
from statsmodels.compat import urlopen
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.graphics.api import abline_plot
from statsmodels.stats.anova import anova_lm

np.set_printoptions(precision=4, suppress=True)

def get_data():
    """
    Variable	Description
    =====================================================================
    TEST	    Job aptitude test score
    MINORITY	1 if applicant could be considered minority, 0 otherwise
    JPERF	    Job performance evaluation

    TEST     MINORITY  JPERF
    0  0.28         1   1.83
    1  0.97         1   4.59
    2  1.25         1   2.97
    3  2.46         1   8.14
    4  2.51         1   8.00
    """
    url = 'http://stats191.stanford.edu/data/jobtest.table'
    fh = urlopen(url)
    df = pd.read_table(fh)
    print(df.head())
    return df 

def plot_data(data):
    factor_group = data.groupby(['MINORITY'])
    fig, ax = plt.subplots(figsize=(6,6))
    colors = ['purple', 'green']
    markers = ['o', 'v']
    for factor, group in factor_group:
        ax.scatter(group['TEST'], 
                group['JPERF'], 
                color=colors[factor],
                marker=markers[factor], 
                s=12**2)
    ax.set_xlabel('TEST')
    ax.set_ylabel('JPERF')
    return fig, ax

def fit_linear_model(formula, data, subset=None, show=False):
    """
    In theory, there may be a linear relationship between JPERF and TEST
    But it could be different by group

    Model:
        JPERF_i = β0 + β1*TEST_i + β2*MINORITY_i + β3*MINORITY_i∗TEST_i + ε_i

    Regression functions:
        Y_i = β0 + β1*TEST_i + ε_i                (if MINORITY_i = 0) and
        Y_i = (β0 + β2) + (β1 + β3)*TEST_i + ε_i  (if MINORITY_i = 1)
    """
    lm = ols(formula, data, subset).fit()
    print(lm.summary())
    return lm


if __name__=="__main__":
    jobtest_table = get_data()
    fig, ax = plot_data(jobtest_table)
    plt.title("Raw data")
    plt.show()

    # β2 = β3 = 0
    # This has no effect for MINORITY
    formula = "JPERF ~ TEST" 
    lm = fit_linear_model(formula, data=jobtest_table)
    fig, ax = plot_data(jobtest_table)
    fig = abline_plot(model_results = lm, ax=ax)
    plt.title("JPERF ~ TEST")
    plt.show()

    # β3 = 0
    # This model allows for an effect of MINORITY 
    # but no interaction between MINORITY and TEST.
    formula2 = "JPERF ~ TEST + MINORITY"
    lm2 = fit_linear_model(formula2, data=jobtest_table)
    fig, ax = plot_data(jobtest_table)
    fig = abline_plot(intercept = lm2.params['Intercept'],
                    slope = lm2.params['TEST'], 
                    ax=ax, color='purple')

    fig = abline_plot(intercept = lm2.params['Intercept'] + lm2.params['MINORITY'],
                    slope = lm2.params['TEST'], 
                    ax=ax, color='green')

    plt.title("JPERF ~ TEST + MINORITY")
    plt.show()
 
    # β2 = 0
    # This model includes an interaction between TEST and MINORITY. 
    # These lines have the same intercept 
    # but possibly different slopes within the MINORITY groups.
    formula3 = "JPERF ~ TEST + TEST:MINORITY"
    lm3 = fit_linear_model(formula3, data=jobtest_table)
    fig, ax = plot_data(jobtest_table)
    fig = abline_plot(intercept = lm3.params['Intercept'],
                    slope = lm3.params['TEST'], 
                    ax=ax, color='purple')

    fig = abline_plot(intercept = lm3.params['Intercept'],
                    slope = lm3.params['TEST'] + lm3.params['TEST:MINORITY'],
                    ax=ax, color='green')

    plt.title("JPERF ~ TEST + TEST:MINORITY")
    plt.show()
    
    # no constraints
    # This model allows for different intercepts and different slopes.
    # The expression TEST*MINORITY is shorthand for TEST + MINORITY + TEST:MINORITY.
    formula4 = "JPERF ~ TEST * MINORITY"
    lm4 = fit_linear_model(formula4, data=jobtest_table)
    fig, ax = plot_data(jobtest_table)
    fig = abline_plot(intercept = lm4.params['Intercept'],
                    slope = lm4.params['TEST'], 
                    ax=ax, color='purple')

    fig = abline_plot(intercept = lm4.params['Intercept'] + lm4.params['MINORITY'],
                    slope = lm4.params['TEST'] + lm4.params['TEST:MINORITY'],
                    ax=ax, color='green')
    plt.title("JPERF ~ TEST * TEST:MINORITY")
    plt.show()

    # is there any effect of MINORITY on slope or intercept?
    table = anova_lm(lm, lm4)
    print("TEST vs. TEST * MINORITY")
    print(table)
    print("\n")
    """
    TEST vs. TEST * MINORITY
    ========================
        df_resid        ssr  df_diff    ss_diff         F    Pr(>F)
    0      18.0  45.568297      0.0        NaN       NaN       NaN
    1      16.0  31.655473      2.0  13.912824  3.516061  0.054236
    """

    # is there any effect of MINORITY on slope
    # NOTE: assumption. the slope is the same within each group
    table = anova_lm(lm, lm3)
    print("TEST vs. TEST:MINORITY")
    print(table)
    print("\n")
    """
    TEST vs. TEST:MINORITY
    ======================
        df_resid        ssr  df_diff    ss_diff         F    Pr(>F)
    0      18.0  45.568297      0.0        NaN       NaN       NaN
    1      17.0  34.707653      1.0  10.860644  5.319603  0.033949
    """

    # is there any effect of MINORITY on intercept
    # NOTE: assumption. the slope is the same within each group
    table = anova_lm(lm, lm2)
    print("TEST vs. TEST + MINORITY")
    print(table)
    print("\n")
    """
    TEST vs. TEST + MINORITY
    ========================
        df_resid        ssr  df_diff   ss_diff         F    Pr(>F)
    0      18.0  45.568297      0.0       NaN       NaN       NaN
    1      17.0  40.321546      1.0  5.246751  2.212087  0.155246
    """

    # is it just the slope or both? (slope & intercept)
    table = anova_lm(lm2, lm4)
    print("TEST + MINORITY vs. TEST * MINORITY")
    print(table)
    print("\n")
    """
    TEST + MINORITY vs. TEST * MINORITY
    ===================================
        df_resid        ssr  df_diff   ss_diff         F   Pr(>F)
    0      17.0  40.321546      0.0       NaN       NaN      NaN
    1      16.0  31.655473      1.0  8.666073  4.380196  0.05265
    """