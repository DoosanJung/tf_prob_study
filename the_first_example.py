"""
The First example
"""
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


# Pretend to load synthetic data set.
features = tfp.distributions.Normal(loc=0., scale=1.).sample(int(100e3))
labels = tfp.distributions.Bernoulli(logits=1.618 * features).sample()

# Specify model.
# https://www.tensorflow.org/probability/api_docs/python/tfp/glm/Bernoulli
model = tfp.glm.Bernoulli()

# Fit model given data.
coeffs, linear_response, is_converged, num_iter = tfp.glm.fit(
    model_matrix=features[:, tf.newaxis],
    response=tf.cast(labels, dtype=tf.float32),
    model=model)

sess = tf.Session()
with sess.as_default():
    features_first_ten = features[:10]
    pr = tf.print(features_first_ten)
    print("features[:10]: ")
    sess.run(pr)
    print("mean : ", np.mean(features.eval())) #sample mean
    print("stdev: ", np.std(features.eval(), ddof=1)) #sample stdev
    print("coeffs: ")
    p = tf.print(coeffs)
    sess.run(p)

