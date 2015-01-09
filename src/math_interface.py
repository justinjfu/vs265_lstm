"""
Provide math operations in a CPU/GPU independent manner
"""
__author__ = 'justin'

GNUMPY_MODE = 0
NUMPY_MODE = 1

COMP_MODE = NUMPY_MODE  # Computation mode

new_namespace = lambda: lambda: 0  # HACK: make a function (which can store attributes)

if COMP_MODE == NUMPY_MODE:
    import numpy as np
    import scipy.special

    # Inner/outer products
    dot = np.dot
    outer = np.outer

    # Misc
    array = np.array
    zeros = np.zeros
    ones = np.ones
    random = new_namespace()
    random.uniform = np.random.uniform
    random.seed = np.random.seed
    ndenumerate = np.ndenumerate

    # Math functions
    exp = np.exp
    logistic = scipy.special.expit
    tanh = np.tanh
    log = np.log
    sum = np.sum
    cumsum = np.cumsum

    linalg = new_namespace()
    linalg.norm = np.linalg.norm

elif COMP_MODE == GNUMPY_MODE:
    import math_utils.gnumpy as gnp
    import numpy as np

    # Inner/outer products
    dot = gnp.dot
    outer = gnp.outer

    # Array Initialization
    array = gnp.as_garray
    zeros = gnp.zeros
    ones = gnp.ones
    random = new_namespace()
    random.uniform = lambda min,max,shape: (gnp.rand(shape)-min)*(max-min)
    random.seed = gnp.seed_rand
    ndenumerate = lambda arr: np.ndenumerate(gnp.as_numpy_array(arr))  # HACK!

    # Math functions
    exp = gnp.exp
    tanh = gnp.tanh
    logistic = gnp.logistic
    log = gnp.log
    sum = gnp.sum
    cumsum = lambda arr, **kwargs: gnp.as_garray(np.cumsum(gnp.as_numpy_array(arr), **kwargs))


    linalg = new_namespace()
    linalg.norm = lambda arr: np.linalg.norm(gnp.as_numpy_array(arr))  # HACK!
