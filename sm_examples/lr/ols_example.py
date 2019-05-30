"""
The code example are modified from below source to confirm my understanding.
Source: https://www.statsmodels.org/stable/index.html

Linear Regression
=================
Linear models with independently and identically distributed errors, 
and for errors with heteroscedasticity or autocorrelation.

Y = X * β + μ, where μ ~ N(0, Σ).

Depending on the properties of Σ
    - OLS   : ordinary least squares for i.i.d. errors Σ=I
    - WLS   : weighted least squares for heteroskedastic errors diag(Σ)
    - GLS   : generalized least squares for arbitrary covariance Σ
    - GLSAR : feasible generalized least squares with autocorrelated AR(p) errors Σ=Σ(ρ)
"""
import numpy as np
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import matplotlib.pyplot as plt
from statsmodels.datasets.longley import load_pandas

class OLSExample(object):
    def set_config(self, x_start, x_stop, n_samples, beta, dummy_slices=None):
        self.x_start = x_start
        self.x_stop = x_stop
        self.n_samples = n_samples
        self.beta = beta
        self.dummy_slices = dummy_slices

    def make_data(self):
        """
        Artificial data
        """
        x = np.linspace(self.x_start, self.x_stop, self.n_samples)
        self.X = np.column_stack((x, x**2)) 
        self.X = sm.add_constant(self.X) #add ones to first column
        print("X.shape: ", self.X.shape)
        print("len(beta): ", len(self.beta))

        # introducing independent error term
        e = np.random.normal(size=self.n_samples)

        self.y = np.dot(self.X, self.beta) + e
        print("y.shape: ", self.y.shape)       
        return self.y, self.X

    def make_non_linear(self, sigma):
        """
        Artificial data: a non-linear relationship between x and y
        """
        self.sigma = sigma
        self.x = np.linspace(self.x_start, self.x_stop, self.n_samples)
        self.X = np.column_stack((self.x, np.sin(self.x), (self.x-5)**2, np.ones(self.n_samples)))
        print("X.shape: ", self.X.shape)
        print("len(beta): ", len(self.beta))
        
        self.y_true = np.dot(self.X, self.beta)
        print("y_true.shape: ", self.y_true.shape)
        
        # introducing independent error term
        e = np.random.normal(size=self.n_samples)
        self.y = self.y_true + self.sigma * e 
        print("y.shape: ", self.y.shape)
        return self.y, self.X

    def make_dummy_vars(self):
        self.groups = np.zeros(self.n_samples, int)
        self.groups[self.dummy_slices[0]:self.dummy_slices[1]] = 1
        self.groups[self.dummy_slices[1]:] = 2
        self.dummy = sm.categorical(self.groups, drop=True)

        self.x = np.linspace(self.x_start, self.x_stop, self.n_samples)
        # drop reference category
        X = np.column_stack((self.x, self.dummy[:,1:]))
        self.X = sm.add_constant(X, prepend=False)
          
        self.y_true = np.dot(self.X, self.beta)
        print("y_true.shape: ", self.y_true.shape)
        
        # introducing independent error term
        e = np.random.normal(size=self.n_samples)
        self.y = self.y_true + e 
        print("y.shape: ", self.y.shape)

        # inspect the data
        print("groups:", self.groups)
        print("dummy (head) :", self.dummy[:5,:])
        print("X     :", self.X[:5,:])
        print("y     :", self.y[:5])
        return self.y, self.X
        
    def fit_data(self, y, x):
        """
        Fit and summarize OLS model
        """
        ols_model = sm.OLS(y, x)
        self.res_ols = ols_model.fit()
        print(self.res_ols.summary())
        return self.res_ols

    def plot(self, result):
        """
        Draw a plot to compare the true relationship to OLS predictions. 
        Confidence intervals around the predictions are built using `wls_prediction_std`.
        """
        prstd, iv_l, iv_u = wls_prediction_std(result)

        fig, ax = plt.subplots(figsize=(8,6))

        ax.plot(self.x, self.y, 'o', label="data")
        ax.plot(self.x, self.y_true, 'b-', label="True")
        ax.plot(self.x, self.res_ols.fittedvalues, 'r--.', label="OLS")
        ax.plot(self.x, iv_u, 'r--')
        ax.plot(self.x, iv_l, 'r--')
        ax.legend(loc='best')
        plt.show()

if __name__=="__main__":
    # basic example
    ols = OLSExample()
    ols.set_config(x_start = 0,
        x_stop = 10,
        n_samples = 100, 
        beta = [1.0, 0.1, 10])
    (y, X) = ols.make_data()
    res_ols = ols.fit_data(y, X)
    print("Parameters     :", res_ols.params) 
    print("Standard errors: ", res_ols.bse) 
    print("R^2            :", res_ols.rsquared) 

    # a non-linear relationship between x and y
    ols_non_linear = OLSExample()
    ols_non_linear.set_config(x_start = 0,
        x_stop = 20,
        n_samples = 50, 
        beta = [0.5, 0.5, -0.02, 5.])
    (y, X) = ols_non_linear.make_non_linear(sigma=0.5)
    res_ols_nonlinear = ols_non_linear.fit_data(y, X)
    print("Parameters     :", res_ols_nonlinear.params) 
    print("Standard errors: ", res_ols_nonlinear.bse) 
    print("R^2            :", res_ols_nonlinear.rsquared)
    ols_non_linear.plot(result=res_ols_nonlinear)

    # OLS with dummy variables
    ols_dummy_vars = OLSExample()
    ols_dummy_vars.set_config(x_start = 0,
        x_stop = 20,
        n_samples = 50, 
        beta = [1., 3, -3, 10],
        dummy_slices=[20, 40])
    (y, X) = ols_dummy_vars.make_dummy_vars()
    res_ols_dummy = ols_dummy_vars.fit_data(y, X)
    ols_dummy_vars.plot(result=res_ols_dummy)

    # Joint hypothesis test (OLS with dummy variables)
    """
    F-test
    - H0: both coefficients on the dummy variables are equal to zer0: R × β = 0
    - H1: Not H0
    
    An r x k array where 
    - r is the number of restrictions to test 
    - k is the number of regressors. 
    """
    R = np.array([[0, 1, 0, 0], 
                  [0, 0, 1, 0]])
    print("r_matrix.shape:", R.shape)
    # An F test leads us to strongly reject the null hypothesis of identical constant in the 3 groups:
    print(res_ols_dummy.f_test(R))
    print(res_ols_dummy.f_test("x2 = x3 = 0"))

    """
    small group effect
    If we generate artificial data with smaller group effects, 
    the T test can no longer reject the Null hypothesis
    """
    ols_dummy_vars_small = OLSExample()
    ols_dummy_vars_small.set_config(x_start = 0,
        x_stop = 20,
        n_samples = 50, 
        beta = [1.0, 0.3, -0.0, 10],
        dummy_slices=[20, 40])
    (y, X) = ols_dummy_vars_small.make_dummy_vars()
    res_ols_dummy_small = ols_dummy_vars_small.fit_data(y, X)
    print(res_ols_dummy_small.f_test(R))
    print(res_ols_dummy_small.f_test("x2 = x3 = 0"))

    """
    Multicollinearity: the exogenous predictors are highly correlated.
    This is problematic because it can affect the stability of our coefficient estimates 
    as we make minor changes to model specification.
    """
    y_multicol = load_pandas().endog
    X_multicol = load_pandas().exog
    X_multicol = sm.add_constant(X_multicol)
    print("X: ", X_multicol)

    res_ols_multicollinearity = ols.fit_data(y_multicol, X_multicol)
    print("Parameters     :", res_ols_multicollinearity.params) 
    print("Standard errors: ", res_ols_multicollinearity.bse) 
    print("R^2            :", res_ols_multicollinearity.rsquared)     

    """
    condition number: to assess multicollinearity
    - Values over 20 are worrisome (see Greene 4.9)
    """
    #The first step is to normalize the independent variables to have unit length
    norm_x = X_multicol.values
    for i, name in enumerate(X_multicol):
        if name == "const":
            continue
        norm_x[:,i] = X_multicol[name]/np.linalg.norm(X_multicol[name])
    norm_xtx = np.dot(norm_x.T,norm_x)

    # square root of the ratio of the biggest to the smallest eigen values.
    eig_vals = np.linalg.eigvals(norm_xtx)
    condition_number = np.sqrt(eig_vals.max() / eig_vals.min())
    print("the condition number (over 20 is worrisome): ", condition_number)

    """
    Greene also points out that dropping a single observation can have a dramatic effect 
    on the coefficient estimates.
    
    In general, DBETAS measures the difference in each parameter estimate 
    with and without the influential point. 
        - Belsley, Kuh, and Welsch recommend 2 as a general cutoff value 
        to indicate influential observations and 2/sqrt(n) as a size-adjusted cutoff
        - an influential observation: an observation for a statistical calculation 
        whose deletion from the dataset would noticeably change the result of the calculation

    DFBETA measures the difference in each parameter estimate with and without 
    the influential point. There is a DFBETA for each point and each observation. 
    """
    print("DFBETAs")
    infl = res_ols_multicollinearity.get_influence()
    threshold = 2.0/len(X_multicol)**0.5 # 0.5
    print("threshold value:", threshold)
    print(infl.summary_frame().filter(regex="dfb"))







