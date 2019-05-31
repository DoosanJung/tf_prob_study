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
Source: https://web.stanford.edu/class/stats191/notebooks/Interactions.html
Source: https://web.stanford.edu/class/stats191/notebooks/ANOVA.html
"""
import statsmodels.api as sm
from statsmodels.compat import urlopen
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.graphics.api import interaction_plot
from statsmodels.stats.anova import anova_lm

np.set_printoptions(precision=4, suppress=True)

def get_data():
    """
    Outcome: S, salaries for IT staff in a corporation.
    Predictors:
        X, experience (years)
        E, education (1=Bachelor’s, 2=Master’s, 3=Ph.D)
        M, management (1=management, 0=not management) 

        S  X  E  M
    13876  1  1  1
    11608  1  3  0
    18701  1  3  1
    11283  1  2  0
    11767  1  3  0
    """
    url = 'http://stats191.stanford.edu/data/salary.table'
    fh = urlopen(url)
    df = pd.read_table(fh)
    print(df.head())
    return df       

def plot_raw_data(data):
    symbols = ['D', '^'] #mgt or not
    colors = ['r', 'g', 'blue'] # bachelor vs. masters vs. ph.D
    factor_groups = data.groupby(['E','M'])
    for (edu, mgt), group in factor_groups:
        plt.scatter(group['X'], group['S'], marker=symbols[mgt], color=colors[edu-1], s=144)
    plt.xlabel('Experience')
    plt.ylabel('Salary')
    plt.title('Experience vs. Salary')
    plt.show()

def fit_linear_model(formula, data, subset=None, show=False):
    """
    Fitting models using R-style formula
    For categorical variables; use C() operator
    Source: https://www.statsmodels.org/stable/example_formulas.html

    """
    res_lm = ols(formula, data, subset).fit()
    print(res_lm.summary())
    if show == True:
        print("created design matrix (model.exog): \n", res_lm.model.exog[:5])
        print("If data is pd.DataFrame, we can also do (model.data.orig_exog)\n", res_lm.model.data.orig_exog[:5])
        print("The original untouched data (model.data.frame:\n", res_lm.model.data.frame[:5])
    return res_lm

def get_influence_table(result):
    print("Influence stats: \n")
    infl = result.get_influence()
    print(type(infl.summary_table())) #SimpleTable
    print(infl.summary_table())

    print("Also available in Pandas DataFrame format: \n")
    infl_df = infl.summary_frame()
    print(infl_df.head())
    print(infl_df.columns)
    return infl

def plot_residuals(result, data):
    """
    plot the reiduals within the groups separately
    """
    resid = result.resid
    symbols = ['D', '^'] #mgt or not
    colors = ['r', 'g', 'blue'] # bachelor vs. masters vs. ph.D
    factor_groups = data.groupby(['E','M'])
    for (edu, mgt), group in factor_groups:
        group_num = edu*2 + mgt - 1  # for plotting purposes
        x = [group_num] * len(group)
        plt.scatter(x, resid[group.index], marker=symbols[mgt], color=colors[edu-1], s=144)
    plt.xlabel('Group')
    plt.ylabel('Residuals')
    plt.title('Group vs. Residuals')
    plt.show()

def anova_lm_check(no_inter_res, inter_res):
    """
    anova_lm: Anova table for one or more fitted linear models.
    """
    table1 = anova_lm(no_inter_res, inter_res)
    print(table1)

def plot_residuals_studentized(result, data):
    """
    The standardized residual is the residual divided by its standard deviation.

    formula: standard_residual_i = residual_i / standard_deviation_of_residual_i
    """
    infl = result.get_influence()
    resid_studentized = infl.summary_frame()["standard_resid"]
    symbols = ['D', '^'] #mgt or not
    colors = ['r', 'g', 'blue'] # bachelor vs. masters vs. ph.D
    factor_groups = data.groupby(['E','M'])
    for (edu, mgt), group in factor_groups:
        i = group.index
        plt.scatter(data.X[i], resid_studentized[i], marker=symbols[mgt], color=colors[edu-1], s=144)
    plt.xlabel('X')
    plt.ylabel('standardized resids')
    plt.title('X vs. Standardized Residuals')
    plt.show()
    return resid_studentized

def rm_outlier(result, data):
    drop_idx = abs(result).argmax()
    print("drop index: ",drop_idx)  # zero-based index
    idx = data.index.drop(drop_idx)
    return drop_idx, idx # subset of data after dropping outlier 

def plot_fitted_values(formula, data, drop_idx=None, subset=None):
    """
    this time, data = data.drop([drop_idx]))
    """
    drop_idx = drop_idx
    data = data.drop([drop_idx])
    lm_final = fit_linear_model(formula=formula, data=data)
    
    mf = lm_final.model.data.orig_exog
    lstyle = ['-','--'] #mgt or not
    symbols = ['D', '^'] #mgt or not
    colors = ['r', 'g', 'blue'] # bachelor vs. masters vs. ph.D
    factor_groups = data.groupby(['E','M'])
    for (edu, mgt), group in factor_groups:
        idx = group.index
        plt.scatter(data.X[idx], data.S[idx], marker=symbols[mgt], color=colors[edu-1], s=144)

        # drop NA because there is no idx 32 in the final model
        plt.plot(mf.X[idx].dropna(), lm_final.fittedvalues[idx].dropna(),
                ls=lstyle[mgt], color=colors[edu-1])
    plt.xlabel('Experience')
    plt.ylabel('Salary')
    plt.show()


if __name__=="__main__":
    salary_table = get_data()
    plot_raw_data(salary_table)

    formula = "S ~ C(E) + C(M) + X" # Categorical variables E, M
    res_lm = fit_linear_model(formula=formula, data=salary_table, show=True)

    # influence variables
    infl = get_influence_table(res_lm)

    # residuals
    plot_residuals(result=res_lm, data=salary_table)

    # test interaction
    # C(E) * X
    formula_interaction_X = "S ~ C(E) * X + C(M)" # C(E) * X
    res_lm_interaction_X = fit_linear_model(formula=formula_interaction_X, data=salary_table)

    # C(E) * C(M)
    formula_interaction_M = "S ~ X + C(E)*C(M)" # C(E) * C(M)
    res_lm_interaction_M = fit_linear_model(formula=formula_interaction_M, data=salary_table)

    # ANOVA check
    print("ANOVA check: no interaction vs. interaction: C(E) * X" + "\n")
    anova_lm_check(res_lm, res_lm_interaction_X)
    """
        df_resid           ssr  df_diff       ss_diff         F    Pr(>F)
    0      41.0  4.328072e+07      0.0           NaN       NaN       NaN
    1      39.0  3.941068e+07      2.0  3.870040e+06  1.914856  0.160964
    """

    print("ANOVA check: no interaction vs. interaction: C(E) * C(M)" + "\n")
    anova_lm_check(res_lm, res_lm_interaction_M)
    """
        df_resid           ssr  df_diff       ss_diff           F        Pr(>F)
    0      41.0  4.328072e+07      0.0           NaN         NaN           NaN
    1      39.0  1.178168e+06      2.0  4.210255e+07  696.844466  3.025504e-31
    """

    # influence variables: interaction: C(E) * C(M)
    infl_inter_M = get_influence_table(res_lm_interaction_M)

    # residuals: interaction: C(E) * C(M)
    resid_studentized = plot_residuals_studentized(result=res_lm_interaction_M, data=salary_table)

    # Looks like one observation is an outlier.
    drop_idx, idx = rm_outlier(resid_studentized, data=salary_table)
    
    # after dropping outlier
    res_lm_subset = fit_linear_model(formula=formula, 
                                    data=salary_table, 
                                    subset=idx)

    res_lm_interaction_X_subset = fit_linear_model(formula=formula_interaction_X,
                                                data=salary_table,
                                                subset=idx)
    
    res_lm_interaction_M_subset = fit_linear_model(formula=formula_interaction_M,
                                                data=salary_table,
                                                subset=idx)

    anova_lm_check(res_lm_subset, res_lm_interaction_X_subset)
    """
        df_resid           ssr  df_diff       ss_diff         F    Pr(>F)
    0      40.0  4.320910e+07      0.0           NaN       NaN       NaN
    1      38.0  3.937424e+07      2.0  3.834859e+06  1.850508  0.171042
    """

    anova_lm_check(res_lm_subset, res_lm_interaction_M_subset)
    """
        df_resid           ssr  df_diff       ss_diff            F        Pr(>F)
    0      40.0  4.320910e+07      0.0           NaN          NaN           NaN
    1      38.0  1.711881e+05      2.0  4.303791e+07  4776.734853  2.291239e-46
    """

    resid_studentized_subset = plot_residuals_studentized(result=res_lm_interaction_M_subset, 
                                                data=salary_table)

    # fitted value plotting
    plot_fitted_values(formula=formula_interaction_M, data=salary_table, drop_idx=drop_idx)

    # the difference between Master's and PhD in the management group is different 
    # than in the non-management group. (interaction between the two qualitative variables M and E) 
    # => first remove the effect of experience, 
    # => then plot the means within each of the 6 groups using interaction.plot.
    U = salary_table.S - salary_table.X * res_lm_interaction_X_subset.params['X']

    # Interaction plot for factor level statistics.
    interaction_plot(x=salary_table.E, 
                    trace=salary_table.M, 
                    response=U, 
                    colors=['red','blue'], 
                    markers=['^','D'],
                    markersize=10, 
                    ax=plt.gca())
    plt.show()