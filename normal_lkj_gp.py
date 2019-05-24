"""
TensorFlow Probability is a library for probabilistic reasoning and statistical analysis in TensorFlow.

Low-level building blocks
- Distributions
- Bijectors

High(er)-level constructs
- Markov chain Monte Carlo
- Probabilistic Layers
- Structural Time Series
- Generalized Linear Models
- Edward2
- Optimizers
"""
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

tfd = tfp.distributions
tf.enable_eager_execution()

def normal_dist():
    """
    https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Normal
    pdf(x; mu, sigma) = exp(-0.5 (x - mu)**2 / sigma**2) / Z
    where Z = (2 pi sigma**2)**0.5

    """
    # A standard normal
    print("\n* Normal dist")
    dist = tfd.Normal(loc=0.0, scale=1.0)
    print(dist)

    # Evaluate the cdf at 0, returning a scalar.
    print("cdf(0) value        : ", dist.cdf(0.)) # 0.5
    
    # Standard deviation and coverage
    print("cdf(-1.0, 1.0) value: ", dist.cdf(1.0) - dist.cdf(-1.0)) # ~ 0.682
    print("cdf(-2.0, 2.0) value: ", dist.cdf(2.0) - dist.cdf(-2.0)) # ~ 0.954
    print("cdf(-3.0, 3.0) value: ", dist.cdf(3.0) - dist.cdf(-3.0)) # ~ 0.997
    
    # Evaluate the pdf of the distribution on 0 and 1.5,returning a length two tensor.
    print("pdf at 0 and 1.5    : ", dist.prob([0, 1.5])) #[0.3989423  0.12951759]

    # Get 3 samples, returning a (3, ) tensor.
    print("draw 3 samples from the standard normal dist: ", dist.sample([3]))

    # Plot 1000 samples from a standard normal
    samples = dist.sample(1000)
    sns.distplot(samples)
    plt.title("Samples from a standard Normal")
    plt.show()

def batch_of_dist():
    """
    Batches are like "vectorized" distributions: 
    independent instances whose computations happen in parallel.

    TensorFlow Probability Distributions have shape semantics. 
    - Batch shape denotes a collection of Distributions with distinct parameters
    - Event shape denotes the shape of samples from the Distribution.
    Always put batch shapes on the "left" and event shapes on the "right".
    """
    # Create a batch of 3 normals, and plot 1000 samples from each
    print("\n* A batch of normal dists")
    normals = tfd.Normal([-1, 0.0, 1], 1.0)  # The scale parameter broadacasts!
    print("Batch shape:", normals.batch_shape)
    print("Event shape:", normals.event_shape)

    # Samples' shapes go on the left!
    samples = normals.sample(1000)
    print("Shape of samples:", samples.shape)
    
    # The computation broadcasts, 
    # so a batch of normals applied to a scalar also gives a batch of cdfs.
    #[0.8413447  0.5        0.15865526] for tfd.Normal([-1, 0.0, 1], 1.0) respectively
    print(normals.cdf(0.))

    for i in range(3):
        sns.distplot(samples[:, i],norm_hist=True)
    plt.title("Samples from 3 Normals, and their PDF's")
    plt.show()

def multivariate_normal():
    """
    The Multivariate Normal distribution is defined over R^k and parameterized 
    by a (batch of) length-k loc vector (aka "mu") and a (batch of) k x k scale matrix.
    https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/MultivariateNormalDiag
    """
    # Initialize a single 2-variate Gaussian.
    print("\n* Multivariate normal dist")
    mvn = tfd.MultivariateNormalDiag(loc=[0., 0.], scale_diag = [1., 1.])
    print("Batch shape:", mvn.batch_shape)
    print("Event shape:", mvn.event_shape)
    print("Means   :", mvn.mean())
    print("Std devs:", mvn.stddev())

    samples = mvn.sample(1000)
    print("Samples shape:", samples.shape)
    g = sns.jointplot(samples[:, 0], samples[:, 1], kind='scatter')
    plt.show()

def lkj():
    """
    The LKJ distribution on correlation matrices.
    named after Lewandowski, Kurowicka, and Joe, who gave a sampler for the distribution in [(Lewandowski, Kurowicka, Joe, 2009)].
    https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/LKJ
    https://www.sciencedirect.com/science/article/pii/S0047259X09000876
    """
    # Initialize a single 3x3 LKJ with concentration parameter 1.5
    print("\n* A single 3x3 LKJ")
    dist = tfp.distributions.LKJ(dimension=3, concentration=1.5)

    ans = dist.sample()
    sns.heatmap(ans)
    plt.show()

    # Initialize two 10x10 LKJ with concentration parameter 1.5 and 3.0
    print("\n* two 10x10 LKJs")
    lkj = tfd.LKJ(dimension=10, concentration=[1.5, 3.0])
    print("Batch shape: ", lkj.batch_shape)
    print("Event shape: ", lkj.event_shape)

    samples = lkj.sample()
    print("Samples shape: ", samples.shape)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))
    sns.heatmap(samples[0, ...], ax=axes[0], cbar=False) #1st with concentration = 1.5
    sns.heatmap(samples[1, ...], ax=axes[1], cbar=False) #2nd with concentration = 3.0
    fig.tight_layout()
    plt.show()

def gaussian_process():
    """
    * Gaussian Process
    The GP may be thought of as a distribution over functions defined over the index set.
    https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/GaussianProcess

    Gaussian Process Regression
    Assume a Gaussian process prior, f ~ GP(m, k) with IID normal noise on observations of function values.

    well written here:
    https://katbailey.github.io/post/gaussian-processes-for-dummies/
    """
    print("\n* gaussian process")
    psd_kernels = tfp.positive_semidefinite_kernels

    num_points = 100
    index_points = np.linspace(-5.0, 5.0, num_points).reshape([-1, 1])

    # Define a kernel with default parameters.
    # Sometimes called the "squared exponential", "Gaussian" or "radial basis function" 
    kernel = psd_kernels.ExponentiatedQuadratic()

    gp = tfd.GaussianProcess(kernel, index_points)

    print("Batch shape:", gp.batch_shape)
    print("Event shape:", gp.event_shape)


    # plot GP prior mean, 2 sigma intervals and samples
    # ==> 100 independently drawn, joint samples at `index_points`
    upper, lower = gp.mean() + [2 * gp.stddev(), -2 * gp.stddev()]
    plt.plot(index_points, gp.mean())
    plt.fill_between(index_points[..., 0], upper, lower, color='k', alpha=.1)
    for _ in range(5):
        plt.plot(index_points, gp.sample(), c='r', alpha=.3)
    plt.title("GP prior mean, $2\sigma$ intervals, and samples")
    plt.show()

    noisy_gp = tfd.GaussianProcess(
        kernel=kernel,
        index_points=index_points,
        observation_noise_variance=.05)

    # ==> 100 independently drawn, noisy joint samples at `index_points`
    for _ in range(5):
        plt.plot(index_points, noisy_gp.sample(), c='b', alpha=.3)
    plt.title("GP prior mean, $2\sigma$ intervals, and noisy joint samples")
    plt.show()

    print("\n* gaussian process regression")
    # Suppose we have some observed data from a known function y = x * x
    obs_x = np.array([[-3.], [0.], [2.]])  # Shape 3x1 (3 1-D vectors)
    f = lambda x: x*x
    obs_y = f(obs_x).reshape(3, ) # Shape 3   (3 scalars)

    gprm = tfd.GaussianProcessRegressionModel(kernel, index_points, obs_x, obs_y)
    upper, lower = gprm.mean() + [2 * gprm.stddev(), -2 * gprm.stddev()]
    plt.plot(index_points, gprm.mean())
    plt.fill_between(index_points[..., 0], upper, lower, color='k', alpha=.1)
    for _ in range(5):
        plt.plot(index_points, gprm.sample(), c='r', alpha=.3)
    plt.scatter(obs_x, obs_y, c='k', zorder=3)
    plt.title("GP posterior mean, $2\sigma$ intervals, and samples")
    plt.show()

if __name__=="__main__":
    # normal_dist()
    # batch_of_dist()
    # multivariate_normal()
    # lkj()
    gaussian_process()
    