"""
The code example are modified from below source to confirm my understanding.
Source: https://www.statsmodels.org/stable/index.html

Regression plots
Source: https://www.statsmodels.org/dev/examples/notebooks/generated/regression_plots.html
"""
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt

class RegressionPlotExample(object):
    def get_data(self):
        """
                    type  income  education  prestige
        accountant  prof      62         86        82
        pilot       prof      72         76        83
        architect   prof      75         92        90
        author      prof      55         90        76
        chemist     prof      64         86        90
        """
        self.data = sm.datasets.get_rdataset("Duncan", "carData", cache=True).data
        print("prestige.shape : ", self.data.shape)
        print("prestige.head(): \n", self.data.head())
        return self.data

    def fit_linear_model(self, formula, data, subset=None, cache=True):
        """
        """
        res_lm = ols(formula, data, subset).fit()
        print(res_lm.summary())
        return res_lm

    def get_influence_table(self, result):
        print("Influence stats: \n")
        infl = result.get_influence()
        print(type(infl.summary_table())) #SimpleTable
        print(infl.summary_table())

        print("Also available in Pandas DataFrame format: \n")
        infl_df = infl.summary_frame()
        print(infl_df.head())
        print(infl_df.columns)
        return infl

    def get_influence_plot(self, result, criterion="cooks"):
        """
        - Influence 'bubble' plots show the 
            (1) (externally) studentized residuals vs. 
            (2) the leverage of each observation as measured by the hat matrix

        - The areas of the circles representing the observations proportional to the value Cook's distance.
        “Cook’s distance” measures how much the predicted values from the regression would change 
        if each observation were removed from the data used to fit the model. 

        - The influence plot helps you identify individual data points that might have undue influence 
        over the fitted regression equation. Unusual data points can be unusual because they have an unusual 
        combination of X values, or because their Y value is unusual given their X values.

        - Points that have unusual X’s are called “high leverage points,” and points with unusual Y values
        (given X) are “outliers.” Finding outliers is easy, you just look for observations with large residuals
        (in absolute value), which you can do on a residual plot.

        Source:https://sites.google.com/site/statisticsforspreadsheets/regression-modeling/influence-plot
        """
        fig, ax = plt.subplots(figsize=(12,8))
        
        # The influence of each point can be visualized by the criterion keyword argument. 
        # Options are Cook's distance and DFFITS, two measures of influence.
        fig = sm.graphics.influence_plot(result, ax=ax, criterion=criterion)
        plt.show()

    def get_partial_residual_plots(self, endog, exog_i, exog_others, data=None):
        """
        Multivariate regressions setting.
        Need to look at the relationship of the dependent variable and independent variables 
        conditional on the other independent variables.
        
        (1) compute the residuals by regressing the response variable vs. 
        the independent variables excluding X_k. (kth variable). Let's call it X~k
        
        (2) then, compute the residuals by regressing X_k on X~k. 
        
        The partial regression plot is the plot of the (1) versus the (2). 

        Parameters
        ==========
        endog: ndarray or string
            endogenous or response variable

        exog_i: ndarray or string
            exogenous or explanatory variable

        exog_others: dndarray or list of strings
            other exogenous, explanatory variables

        data: DataFrame, dict, or recarray
            Some kind of data structure with names if the other variables are given as strings.
        """

        fig, ax = plt.subplots(figsize=(12,8))
        fig = sm.graphics.plot_partregress(endog=endog, 
                                        exog_i=exog_i, 
                                        exog_others=exog_others, 
                                        data=data, 
                                        ax=ax)
        plt.show()

    def get_partial_residual_grid(self, result, title=None):
        """
        Plot partial regression for a set of regressors

        A subplot is created for each explanatory variable given by exog_idx. 
        The partial regression plot shows the relationship between the response 
        and the given explanatory variable after removing the effect of all other 
        explanatory variables in exog.
        """
        fig = plt.figure(figsize=(12,8))
        fig = sm.graphics.plot_partregress_grid(result, fig=fig)
        if title:
            plt.title(title)
        plt.show()

if __name__=="__main__":
    # Regression Plot Example 1
    regression_plot_example = RegressionPlotExample()
    data = regression_plot_example.get_data()

    formula = "prestige ~ income + education"
    res_lm = regression_plot_example.fit_linear_model(formula, data=data)
    regression_plot_example.get_influence_table(result=res_lm)
    regression_plot_example.get_influence_plot(result=res_lm)
    regression_plot_example.get_partial_residual_plots(endog="prestige", 
                                                    exog_i="income", 
                                                    exog_others=["income", "education"], 
                                                    data=data)
    
    regression_plot_example.get_partial_residual_plots(endog="prestige", 
                                                    exog_i="income", 
                                                    exog_others=["education"], 
                                                    data=data)

    # => The influence of conductor, minister, and RR.engineer on the partial relationship between income and prestige. 
    # => remove those three
    subset = ~data.index.isin(["conductor", "RR.engineer", "minister"])    
    res_lm_rm = regression_plot_example.fit_linear_model(formula, data=data, subset=subset)

    # For a quick check of all the regressors
    regression_plot_example.get_partial_residual_grid(res_lm, title="original data")
    regression_plot_example.get_partial_residual_grid(res_lm_rm, title="remove conductor,RR.engineer, minister")
