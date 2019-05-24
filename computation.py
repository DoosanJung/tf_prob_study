"""
https://www.tensorflow.org/probability/overview

- lots of mathematical operations
- efficient vectorized computation
- automatic differentiation
"""
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import time

def numpy_compatibility():
    """
    conversion between TensorFlow Tensors and NumPy ndarrays is quite simple as:
    - TensorFlow operations automatically convert NumPy ndarrays to Tensors.
    - NumPy operations automatically convert Tensors to NumPy ndarrays.
    """
    ndarray = np.ones([3, 3])
    tensor_op_on_np = tf.multiply(ndarray, 42)
    np_op_on_tensor = np.add(tensor_op_on_np, 1)
    return (ndarray, tensor_op_on_np, np_op_on_tensor)

def custom_timeit(method):
    def timed(*args, **kwargs):
        t1 = time.time()
        result = method(*args, **kwargs)
        t2 = time.time()
        if not kwargs:
            print("{0} (args:{1}): {2} sec".format(method.__name__, args, t2-t1))
        else:
            print("{0} (args:{1}, kwarg:{2}): {3} sec".format(method.__name__, args, kwargs, t2-t1))
        return result
    return timed

@custom_timeit
def for_loop_solve(mats, vecs):
  return np.array([tf.linalg.solve(mats[i, ...], vecs[i, ...]) for i in range(1000)])

@custom_timeit
def vectorized_solve(mats, vecs):
  return tf.linalg.solve(mats, vecs)

def automatic_diff():
    """
    Tensorflow "records" all operations executed inside the context of a tf.GradientTape onto a "tape"
    https://www.tensorflow.org/api_docs/python/tf/GradientTape
    """
    a = tf.constant(np.pi)
    b = tf.constant(np.e)
    with tf.GradientTape() as tape:
        tape.watch([a, b])
        c = .5 * (a**2 + b**2)

    grads = tape.gradient(c, [a, b])
    return (a, b, c, grads)

def persist_gradient():
    """
    By default, the resources held by a GradientTape are released 
    as soon as GradientTape.gradient() method is called. 
    
    To compute multiple gradients over the same computation, create a persistent gradient tape.
    """
    p = tf.constant(3.0)
    with tf.GradientTape(persistent=True) as g:
        g.watch(p)
        q = p * p
        r = q * q
    dr_dp = g.gradient(r, p)  # 108.0 (4*p^3 at p = 3)
    dq_dp = g.gradient(q, p)  # 6.0
    del g  # Drop the reference to the tape
    return (dr_dp, dq_dp)

def higher_order_diff():
    """
    Operations inside of the GradientTape context manager are recorded for automatic differentiation. 
    If gradients are computed in that context, then the gradient computation is recorded as well.
    """
    x = tf.constant(3.0)
    with tf.GradientTape() as tape:
        tape.watch(x)
        with tf.GradientTape() as tape2:
            tape2.watch(x)
            y = x * x
        dy_dx = tape2.gradient(y, x)     # Will compute to 6.0
    d2y_dx2 = tape.gradient(dy_dx, x)  # Will compute to 2.0
    return (dy_dx, d2y_dx2)

if __name__=="__main__":
    #numpy compatibility
    (ndarray, tensor_op_on_np, np_op_on_tensor) = numpy_compatibility()

    #vectorization
    mats = tf.random.uniform(shape=[1000, 10, 10])
    vecs = tf.random.uniform(shape=[1000, 10, 1])
    for_loop_solve(mats, vecs)
    vectorized_solve(mats, vecs)

    # gradients
    (a, b, c, grads) = automatic_diff()
    (dr_dp, dq_dp) = persist_gradient()
    (dy_dx, d2y_dx2) = higher_order_diff()

    sess = tf.Session()
    with sess.as_default():
        print("\n* numpy compatibility")
        print("ndarray        : ", ndarray)
        print("tensor_op_on_np: ", tensor_op_on_np.eval())
        print("np_op_on_tensor: ", np_op_on_tensor.eval())

        print("\n* vectorization")
        print("a (pi)  : ", a.eval())
        print("b (e)   : ", b.eval())
        print("c       : ", c.eval())
        print("grads[0]: ", grads[0].eval())
        print("grads[1]: ", grads[1].eval())

        print("\n* multiple gradients over the same computation")
        print("dr_dp   : ", dr_dp.eval())
        print("dq_dp   : ", dq_dp.eval())

        print("\n* higher-order derivatives")
        print("dy_dx   : ", dy_dx.eval())
        print("d2y_dx2 : ", d2y_dx2.eval())
