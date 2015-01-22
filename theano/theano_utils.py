import theano
import theano.tensor as T
import numpy as np

DEVICE = theano.config.device
FLOATX = theano.config.floatX

if FLOATX == 'float32':
    NP_FLOATX = np.float32
elif FLOATX == 'float64':
    NP_FLOATX = np.float64

# randn as floatX
randn = lambda *dims: 0.1*np.random.randn(*dims).astype(NP_FLOATX)

def set_device(device):
    global DEVICE
    DEVICE = device


def theano_compile(*args0, **kwargs0):
    """
    A wrapper around theano.function for use with def statements
    Example usage:
    >>> @theano_compile()
    ... def plus(x, y, k=2):
    ...     return x+y*k
    >>> print plus(T.scalar('x'), T.scalar('y'), k=1)(3,3)
    6.0

    >>> @theano_compile
    ... def plus2(x, y, k=1):
    ...     return x+y*k
    >>> print plus(T.scalar('x'), T.scalar('y'), k=5)(2,3)
    17.0
    """
    namespace = lambda : 0
    def wrapper(*args1, **kwargs1):
        result = namespace.f(*args1, **kwargs1)
        compiled = theano.function(args1, result, **kwargs0)
        return compiled

    if (not kwargs0) and args0 and callable(args0[0]): # No-arg wrapper
        namespace.f = args0[0]
        return wrapper
    else:
        def argwrapper(f):
            namespace.f = f
            return wrapper
        return argwrapper


def hasgpu():
    """
    Returns if device == 'gpu*'
    """
    return DEVICE.startswith('gpu')

# Functions defined in gpu mode but not cpu mode, or vice versa
# This allows for cpu/gpu independent code
if hasgpu():
    gpu_host = theano.sandbox.cuda.basic_ops.gpu_from_host
else:
    gpu_host = lambda x: x

rng = np.random.RandomState(23455)


