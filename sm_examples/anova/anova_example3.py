"""
The code example are modified from below source to confirm my understanding.
Source: https://www.statsmodels.org/stable/index.html

Analysis of Variance (ANOVA)
============================
Analyze the differences among group means in a sample.

In its simplest form, ANOVA provides a statistical test of whether the population means of several groups 
are equal, and therefore generalizes the t-test to more than two groups. 

One-way ANOVA
- completely randomized experiment with a single factor. 

Two-way ANOVA
- ANOVA generalizes to the study of the effects of multiple factors. 
- When the experiment includes observations at all combinations of levels of each factor, it is termed factorial. 

Source: https://en.wikipedia.org/wiki/Analysis_of_variance
Source: https://web.stanford.edu/class/stats191/notebooks/ANOVA.html
"""
import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.graphics.api import interaction_plot
from statsmodels.stats.anova import anova_lm
import os

np.set_printoptions(precision=4, suppress=True)

def get_data(url):
    df = pd.read_csv(url, delimiter=",")
    print(df.head())
    return df       

def fit_linear_model(formula, data, subset=None, show=False):
    lm = ols(formula, data, subset).fit()
    print(lm.summary())
    return lm

if __name__=="__main__":
    #
    # One-way ANOVA
    #
    """
    Simplest question: 
        is there any group (main) effect?
        H0: α_1 = ... = α_r = 0 ?

    r groups with n_i, 1 < i < r observations per group
    Model:
        Y_ij = μ + α_i + ε_ij, ε_ij ~ IID N(0,σ^2).
    
    Constraint: 
        ∑ α_i = 0.
    """
    url = 'http://stats191.stanford.edu/data/rehab.csv'
    rehab_table = get_data(url)
    """
        Fitness  Time
    0        1    29
    1        1    42
    2        1    38
    3        1    40
    4        1    43
    """
    # plot raw data
    fig, ax = plt.subplots(figsize=(8,6))
    fig = rehab_table.boxplot('Time', 'Fitness', ax=ax, grid=False)
    plt.show()

    formula = 'Time ~ C(Fitness)'
    rehab_lm = fit_linear_model(formula, data=rehab_table)

    table = anova_lm(rehab_lm)
    print(table)
    """
                  df  sum_sq     mean_sq          F    PR(>F)
    C(Fitness)   2.0   672.0  336.000000  16.961538  0.000041
    Residual    21.0   416.0   19.809524        NaN       NaN
    """
    print("")
    print(rehab_lm.model.data.orig_exog)
    """
        Intercept  C(Fitness)[T.2]  C(Fitness)[T.3]
    0         1.0              0.0              0.0
    1         1.0              0.0              0.0
    2         1.0              0.0              0.0
    3         1.0              0.0              0.0
    4         1.0              0.0              0.0
    5         1.0              0.0              0.0
    6         1.0              0.0              0.0
    7         1.0              0.0              0.0
    8         1.0              1.0              0.0
    9         1.0              1.0              0.0
    10        1.0              1.0              0.0
    11        1.0              1.0              0.0
    12        1.0              1.0              0.0
    13        1.0              1.0              0.0
    14        1.0              1.0              0.0
    15        1.0              1.0              0.0
    16        1.0              1.0              0.0
    17        1.0              1.0              0.0
    18        1.0              0.0              1.0
    19        1.0              0.0              1.0
    20        1.0              0.0              1.0
    21        1.0              0.0              1.0
    22        1.0              0.0              1.0
    23        1.0              0.0              1.0
    """
    print("\n")

    #
    # Two-way ANOVA
    #
    """
    Are there main effects for the grouping variables?
        H0: α_1 = ... = α_r = 0 
        H0: β_1 = ... = β_m = 0

    Are there interaction effects:
        H0: (αβ)_ij = 0  where  1 ≤ i ≤ r,1 ≤ j ≤ m.

    r groups in first factor
    m groups in second factor
    n_ij in each combination of factor variables

    Model:
        Y_ijk = μ + α_i + β_j + (αβ)_ij + ε_ijk, ε_ijk ~ N(0,σ^2).
    
    """
    filepath = os.path.join(os.path.curdir, "data", "kidney.csv")
    kidney_table = pd.read_csv(filepath)
    print("Weight     (r): ", kidney_table["Weight"].unique())
    print("Duration   (m): ", kidney_table["Duration"].unique())
    print("ID      (n_ij): ", kidney_table["ID"].unique())
    """
       Days  Duration  Weight  ID
    0   0.0         1       1   1
    1   2.0         1       1   2
    2   1.0         1       1   3
    3   3.0         1       1   4
    4   0.0         1       1   5

    r = 3 (weight gain)
    m = 2 (duration of treatment)
    n_ij = 10 for all (i, j)
    """
    print("Balanced panel" + "\n")
    fig = interaction_plot(kidney_table['Weight'], 
                        kidney_table['Duration'], 
                        np.log(kidney_table['Days']+1),
                        colors=['red', 'blue'], 
                        markers=['D','^'], 
                        ms=10, 
                        ax=plt.gca())
    plt.show()

    formula = "np.log(Days+1) ~ C(Duration)"
    lm = fit_linear_model(formula, data=kidney_table)

    formula2 = "np.log(Days+1) ~ C(Weight)"
    lm2 = fit_linear_model(formula2, data=kidney_table)

    formula3 = "np.log(Days+1) ~ C(Duration) + C(Weight)"
    lm3 = fit_linear_model(formula3, data=kidney_table)

    formula4 = "np.log(Days+1) ~ C(Duration) * C(Weight)"
    lm4 = fit_linear_model(formula4, data=kidney_table)

    # formula5 = "np.log(Days+1) ~ C(Duration) * C(Weight)"
    # lm5 = fit_linear_model(formula5, data=kidney_table)  

    table = anova_lm(lm, lm4)
    print(table)
    print("\n")
    