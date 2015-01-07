"""
Provide math operations in a CPU/GPU independent manner
"""
__author__ = 'justin'

GNUMPY_MODE = 0
NUMPY_MODE = 1

COMP_MODE = NUMPY_MODE  # Computation mode

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
    random = lambda: 0  # HACK: make a namespace for random
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

    linalg = lambda: 0  # HACK: make a namespace for random
    linalg.norm = np.linalg.norm

elif COMP_MODE == GNUMPY_MODE:
    import gnumpy as gnp

    # Inner/outer products
    dot = gnp.dot
    outer = gnp.outer

    # Array Initialization


    # Math functions


    raise NotImplemented