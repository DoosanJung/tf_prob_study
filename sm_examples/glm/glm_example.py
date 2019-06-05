"""
The code example are modified from below source to confirm my understanding.
Source: https://www.statsmodels.org/stable/index.html

Generalized Linear Models
=========================
A flexible generalization of ordinary linear regression that allows for response variables 
that have error distribution models other than a normal distribution.

The GLM generalizes linear regression 
- by allowing the linear model to be related to the response variable via a link function
- by allowing the magnitude of the variance of each measurement to be a function of its predicted value.

Each outcome Y of the dependent variables is assumed to be generated from a particular distribution 
in an exponential family, e.g. normal, binomial, Poisson and gamma distributions.

The mean, E(Y) = mu = g^(-1)(Xβ) where 
    - E(Y) is the expected value of Y; 
    - Xβ is the linear predictor, a linear combination of unknown parameters β; 
    - g is the link function. 

The variance, Var(Y) = V(mu) = V(g^(-1)(Xβ)) 

Only the following combinations make sense for family and link:

    Family	    ident	log	logit   probit	cloglog	  pow	opow    nbinom	loglog	logc
    ====================================================================================
    Gaussian	    x	  x	    x	    x	      x     x	   x	    x	     x	
    inv Gaussian	x	  x				                x				
    binomial	    x	  x	    x	    x	      x	    x	   x		x	     x     x
    Poission	    x	  x				                x				
    neg binomial	x	  x				                x		        x		
    gamma	        x	  x				                x				
    Tweedie 	    x	  x				                x				


Source: https://en.wikipedia.org/wiki/Generalized_linear_model
Source: https://www.statsmodels.org/stable/examples/notebooks/generated/glm.html
"""
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from statsmodels import graphics
from statsmodels.graphics.api import abline_plot
from scipy import stats

class GLMExample(object):
    def __init__(self, title):
        print("=="*20)
        print(title)
        print("=="*20)
        
    def get_data(self, data, note=False, desc=False):
        """
        Star98 dataset which was taken with permission 
        from Jeff Gill (2000) Generalized linear models: A unified approach 
        """
        if note == True:
            print(data.NOTE)
        if desc == True:
            print(data.DESCRLONG)
        self.data = data.load()
        
    def prep_data(self):
        """
        adding constant
        """
        self.data.exog = sm.add_constant(self.data.exog, prepend=False)
        return self.data
        
    def fit_glm(self, endog, exog, family=None):
        """
        endog: Binomial family models accept a 2d array with two columns.
        exog: A nobs x k array where 
            - nobs is the number of observations and 
            - k is the number of regressors. 
            - An intercept is not included by default
        family: The default is Gaussian
        """
        glm_binom = sm.GLM(endog, exog, family)
        self.res = glm_binom.fit()
        print(self.res.summary())
        return self.res

    def plot(self, y, x, title, ylabel, xlabel, type, xmin=0, xmax=1):
        fig, ax = plt.subplots()
        ax.scatter(y, x)
        if type == "fit":
            line_fit = sm.OLS(endog=x, exog=sm.add_constant(y, prepend=True)).fit()
            abline_plot(model_results=line_fit, ax=ax)
        elif type == "residual":
            ax.set_xlim(xmin, xmax)
            ax.hlines(0, xmin, xmax)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        plt.show()

    def hist(self, resid, title):
        fig, ax = plt.subplots()
        resid_std = stats.zscore(resid)
        ax.hist(resid_std, bins=25)
        ax.set_title(title)
        plt.show()
    
    def qqplot(self, resid):
        graphics.gofplots.qqplot(resid, line='r')
        plt.show()


# func outside class
def plot_suite(y, yhat, res, xmin=0, xmax=1):
    # plot yhat vs. y
    glm_example.plot(y=yhat, x=y, 
        title='Model Fit Plot', 
        ylabel='Observed values', 
        xlabel='Fitted values',
        type="fit")

    # plot yhat vs. Peasrson residuals
    glm_example.plot(y=yhat, x=res.resid_pearson, 
        title='Residual Dependence Plot', 
        ylabel='Pearson Residuals', 
        xlabel='Fitted values',
        type="residual",
        xmin=xmin,
        xmax=xmax)

    # histogram of standardized deviance residuals
    resid = res.resid_deviance.copy()
    title = 'Histogram of standardized deviance residuals'
    glm_example.hist(resid=resid, title=title)

    # QQ Plot of Deviance Residuals
    glm_example.qqplot(resid)

if __name__=="__main__":
    """
    GLM: Binomial response data
    ===========================
    """
    glm_example = GLMExample("GLM: Binomial response data")
    glm_example.get_data(data=sm.datasets.star98, note=True, desc=True)
    binom_data = glm_example.prep_data()
    print("endog variables: NABOVE, NBELOW")
    print(binom_data.endog[:5,:])
    print("exog variables: all others")
    print(binom_data.exog[:2, :])
    binom_res = glm_example.fit_glm(endog=binom_data.endog, 
                                    exog=binom_data.exog, 
                                    family=sm.families.Binomial())

    # plotting
    binom_nobs = binom_res.nobs
    binom_y = binom_data.endog[:,0]/binom_data.endog.sum(1)
    binom_yhat = binom_res.mu
    plot_suite(y=binom_y, yhat=binom_yhat, res=binom_res)

    """
    GLM: Gamma for proportional count response
    ==========================================
    """
    glm_example = GLMExample("GLM: Gamma for proportional count response")
    glm_example.get_data(data=sm.datasets.scotland, note=True, desc=True)
    gamma_data = glm_example.prep_data()
    gamma_res = glm_example.fit_glm(endog=gamma_data.endog, 
                                    exog=gamma_data.exog, 
                                    family=sm.families.Gamma())

    # plotting
    gamma_obs = gamma_res.nobs
    gamma_y = gamma_data.endog
    gamma_yhat = gamma_res.mu
    plot_suite(y=gamma_y, yhat=gamma_yhat, res=gamma_res, xmin=0, xmax=100)

    """
    GLM: Gaussian distribution with a noncanonical link
    ===================================================
    """
    # artificial data
    gaussian_nobs = 100
    x = np.arange(gaussian_nobs)
    np.random.seed(54321)
    gaussian_X = np.column_stack((x, x**2))
    gaussian_X = sm.add_constant(gaussian_X, prepend=False)
    lny = np.exp(-(.03*x + .0001*x**2 - 1.0)) + .1 * np.random.rand(gaussian_nobs)
    # import pdb;pdb.set_trace()
    """
    print(gaussian_X[:10, :])
    array([[  0.,   0.,   1.],
        [  1.,   1.,   1.],
        [  2.,   4.,   1.],
        [  3.,   9.,   1.],
        [  4.,  16.,   1.],
        [  5.,  25.,   1.],
        [  6.,  36.,   1.],
        [  7.,  49.,   1.],
        [  8.,  64.,   1.],
        [  9.,  81.,   1.]])

    print(lny[:10])
    [ 2.71919347  2.6383045   2.55974943  2.48251743  2.40758836  2.3342185
    2.26242637  2.19340381  2.1251239   2.05839088]
    """

    glm_example = GLMExample("Gaussian distribution with a noncanonical link")
    gaussian_res = glm_example.fit_glm(endog=lny,
                                    exog=gaussian_X, 
                                    family=sm.families.Gaussian(sm.families.links.log))

    # plotting
    yhat = gaussian_res.mu
    plot_suite(y=lny, yhat=yhat, res=gaussian_res, xmin=0, xmax=2.5)