"""
Math functions not defined in numpy
"""
import math_interface as np

__author__ = 'justin'


def __softmax1d(arr):
    """ Helper softmax for 1D vectors """
    expvec = np.exp(arr)
    denom = np.sum(expvec)
    return expvec/denom


def __softmax2d(arr, axis=0):
    """ Helper softmax for executing across rows """
    if axis > 1:
        raise ValueError("Axis must be 0 or 1")
    shape = arr.shape
    output = np.zeros(arr.shape)
    for i in range(shape[axis]):
        if axis == 1:
            vec = arr[i, :]
            output[i, :] = __softmax1d(vec)
        elif axis == 0:
            vec = arr[:, i]
            output[:, i] = __softmax1d(vec)
    return output


def softmax(arr, axis=0):
    """
    Softmax function implementation.
    Operates on single vectors, or across vectors in a matrix.
    :param arr:
    :param axis: 1 runs softmax across rows, and 0 across columns
    :return:
    >>> softmax([0.1, 0.9])
    array([ 0.31002552,  0.68997448])
    >>> softmax([0.2, 0.8])
    array([ 0.35434369,  0.64565631])
    >>> softmax([0.9, 0.8])
    array([ 0.52497919,  0.47502081])
    >>> softmax([[0.9, 0.8]], axis=1)
    array([[ 0.52497919,  0.47502081]])
    >>> softmax([[0.9, 0.8]], axis=0)
    array([[ 0.52497919,  0.47502081]])
    >>> softmax([[0.1, 0.9],[0.2, 0.8]], axis=1)
    array([[ 0.31002552,  0.68997448],
           [ 0.35434369,  0.64565631]])
    >>> softmax([[0.1, 0.9],[0.2, 0.8]], axis=0)
    array([[ 0.47502081,  0.52497919],
           [ 0.52497919,  0.47502081]])


    """
    arr = np.array(arr)
    shape = arr.shape
    #ndims = len(shape)
    ndims = sum([1 if i > 1 else 0 for i in shape])
    if ndims == 1:
        return __softmax1d(arr)
    elif ndims == 2:
        return __softmax2d(arr, axis=axis)
    else:
        raise NotImplemented