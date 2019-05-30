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
from scipy.linalg import toeplitz

class OLSExample(object):
    def get_data(self):
        """
        >>> data.data
            rec.array(
            [(2.66, 20.0, 0.0, 0.0), (2.89, 22.0, 0.0, 0.0), (3.28, 24.0, 0.0, 0.0),
            (2.92, 12.0, 0.0, 0.0), (4.0, 21.0, 0.0, 1.0), (2.86, 17.0, 0.0, 0.0),
            ...
            (2.67, 24.0, 1.0, 0.0), (3.65, 21.0, 1.0, 1.0), (4.0, 23.0, 1.0, 1.0),
            (3.1, 21.0, 1.0, 0.0), (2.39, 19.0, 1.0, 1.0)], 
            dtype=[('GPA', '<f8'), ('TUCE', '<f8'), ('PSI', '<f8'), ('GRADE', '<f8')])
        """
        self.data = sm.datasets.spector.load()

    def prep_data(self):
        """
        An intercept is not included by default and should be added by the user.
        => Adds a column of ones to an array
        """
        self.data.exog = sm.add_constant(self.data.exog, prepend=False) 
        #if prepend=True, the constant is in the first column
        print("shape               : ", self.data.data.shape)
        return self.data

    def fit_data(self, y, x):
        """
        Fit and summarize OLS model
        """
        ols_model = sm.OLS(y, x)
        res_ols = ols_model.fit()
        print(res_ols.summary())
        self.params = res_ols.params # Parameters
        self.bse = res_ols.bse # Standard errors

class WLSExample(object):
    def set_config(self, x_start, x_stop, n_samples, beta, sigma):
        self.x_start = x_start # used in np.linspace
        self.x_stop = x_stop # used in np.linspace
        self.n_samples = n_samples
        self.beta = beta
        self.sigma = sigma

    def make_data(self):
        """
        Artificial data: Heteroscedasticity 2 groups.

        Model assumptions:
            - Misspecification: true model is quadratic, estimate only linear
            - Independent noise/error term
            - Two groups for error variance, low and high variance groups
        
        """
        x = np.linspace(self.x_start, self.x_stop, self.n_samples)
        X = np.column_stack((x, (x - 5)**2)) 
        X = sm.add_constant(X) #add ones to first column
        print("X.shape: ", X.shape)

        # w is the standard deviation of the error.
        self.w = np.ones(self.n_samples)
        self.w[self.n_samples * 6//10:] = 3 # Two groups for error variance
        print("variance group 1. w[:{0}]: {1}".format((self.n_samples * 6//10), self.w[:self.n_samples * 6//10]))
        print("variance group 2. w[{0}:]: {1}".format((self.n_samples * 6//10), self.w[self.n_samples * 6//10:]))
        print("len(w): ", len(self.w))
        
        print("len(beta): ", len(self.beta))
        self.y_true = np.dot(X, self.beta)
        print("y_true.shape: ", self.y_true.shape)
        
        # introducing independent error term
        e = np.random.normal(size=self.n_samples)
        self.y = self.y_true + self.sigma * self.w * e 
        
        self.X = X[:,[0,1]]
        return self.y, self.X, self.w

    def fit_data(self, y, X, w):
        """
        WLS requires that the weights are proportional to the inverse of the error variance.
        """
        wls_model = sm.WLS(y, X, weights=1./(w ** 2))
        res_wls = wls_model.fit()
        print(res_wls.summary())
        self.params = res_wls.params
        self.bse = res_wls.bse

class GLSExample(object):
    def get_data(self):
        """
        A time series dataset.
        Let's assume that 
            - the data is heteroskedastic
            - we know the nature of the heteroskedasticity. 
            - we can then define sigma and use it to give us a GLS model
        """
        self.data = sm.datasets.longley.load()

    def prep_data(self):
        self.data.exog = sm.add_constant(self.data.exog)
        print("first 5 rows of data: ", self.data.exog[:5])
        print("shape               : ", self.data.data.shape)
        return self.data

    def make_sigma(self, y, x):
        """
        First step: run OLS to get residuals.
        Assume that the error terms follow an AR(1) process with a trend

        then, regress the residuals on the lagged residuals.
        ϵi = β0 + ρ * ϵi−1 + ηi, where η ~ N(0, Σ^2) 

        using toeplitz matrix, mimic AR(1)
        get autocorrelation structure
        """
        #get residuals
        ols_resid = sm.OLS(y, x).fit().resid
        print("len(ols_resid)      : ", len(ols_resid))

        #regress on lagged residuals
        # ols_resid[1:] : drop the first resid, len = 15
        # ols_resid[:-1]: drop the last  resid, len = 15
        resid_fit = sm.OLS(ols_resid[1:], sm.add_constant(ols_resid[:-1])).fit()

        #check the t-value / p-value of rho
        print("t-value for rho     :", resid_fit.tvalues[1])
        print("p-value for rho     :", resid_fit.pvalues[1]) # ~ 0.174

        #we don't have strong evidence that the errors follow AR(1), but continue...
        rho = resid_fit.params[1]

        # an AR(1) process means that near-neighbors have a stronger relation 
        # => we can give this structure by using a toeplitz matrix
        print("e.g. toeplitz of range(5): ", toeplitz(range(5)))
        order = toeplitz(range(len(ols_resid))) # toeplitz of range(15)
        print("toeplitz of range(15): ", order)

        # error covariance structure!
        # autocorrelation structure = rho ** order 
        sigma = rho**order
        return sigma
        
    def fit_data(self, y, x, sigma):    
        gls_model = sm.GLS(y, x, sigma=sigma)
        gls_res = gls_model.fit()
        print(gls_res.summary())
        self.params = gls_res.params
        self.bse = gls_res.bse
        
class GLSARExample(object):
    def fit_data(self, y, x, rho=1):
        """
        iterative_fit(maxiter=3): Perform an iterative two-stage procedure to estimate a GLS model.
        The model is assumed to have AR(p) errors, AR(p) parameters 
        and regression coefficients are estimated iteratively.

        rho: Order of the autoregressive covariance
        """
        glsar_model = sm.GLSAR(y, x, rho=rho)
        glsar_results = glsar_model.iterative_fit(1)
        print(glsar_results.summary())
        self.params = glsar_results.params
        self.bse = glsar_results.bse


if __name__=="__main__":
    # OLS Example
    ols = OLSExample()
    ols.get_data()
    data = ols.prep_data()
    ols.fit_data(data.endog, data.exog)

    # WLS Example
    wls = WLSExample()
    wls.set_config(x_start = 0,
        x_stop = 20,
        n_samples = 50, 
        beta = [5.0, 0.5, -0.01], 
        sigma = 0.5)
    (y, X, w) = wls.make_data()
    wls.fit_data(y, X, w)

    # OLS vs WLS
    ols_vs = OLSExample()
    ols_vs.fit_data(y, X)
    print("ols_vs.params", ols_vs.params)
    print("wls.params", wls.params)

    # GLS Example
    gls = GLSExample()
    gls.get_data()
    data_ts = gls.prep_data()
    sigma = gls.make_sigma(y=data_ts.endog, x=data_ts.exog)
    gls.fit_data(y=data_ts.endog, x=data_ts.exog, sigma=sigma)
    
    # GSLAR Example (with one lag) using GLS data
    # differences in param estimates <= might be smal number of observations in the data..
    gslar = GLSARExample()
    gslar.fit_data(y=data_ts.endog, x=data_ts.exog, rho=1) # with one lag

    print("gls.params                 :", gls.params)
    print("gslar.params               :", gslar.params)
    print("gls.bse (Standard errors)  :", gls.bse)
    print("gslar.bse (Standard errors):", gslar.bse)