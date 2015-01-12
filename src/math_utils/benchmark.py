#!/usr/bin/python
import argparse
import time
import sys

NUMPY='np'
GNUMPY='gnp'
THEANO='theano'

parser = argparse.ArgumentParser(description='Benchmark Matrix Multiply on CPU/GPU')
parser.add_argument('-s', '--size', metavar='N', type=int, default=4096, help='Matrix size')
parser.add_argument('-i', '--iters', metavar='I', type=int, default=5, help='Number of test iterations')
parser.add_argument('-m', '--mode', dest='mode',
                   default=NUMPY, help='Run mode (np, gnp, or theano)')
parser.add_argument('-t', '--tensor3', dest='tensor3', action='store_const', default=False,
                   const=True) 
args = parser.parse_args()

N = args.size

flop = 2*N*N*N
if args.tensor3:
    flop *=N
    a_shape = (N,N,N)
else:
    a_shape = (N,N)

if args.mode == GNUMPY:
    import gnumpy as gnp
    print 'Using Gnumpy'
    randn = gnp.randn

    a = randn(*a_shape)
    b = randn(N,N)

    dot = lambda : a.dot(b)

elif args.mode == NUMPY:
    import numpy as np
    print 'Using Numpy'
    randn = lambda *args: np.random.randn(*args).astype(np.float32)

    a = randn(*a_shape)
    b = randn(N,N)

    dot = lambda : a.dot(b)

elif args.mode == THEANO:
    import numpy as np
    import theano.tensor as T
    from theano import function, shared, sandbox, config
    print 'Using Theano'
    randn = lambda *args: np.random.randn(*args).astype(np.float32)
    """
    aa = T.matrix('a')
    bb = T.matrix('b')
    cc = aa.dot(bb)
    dot = function([aa,bb],sandbox.cuda.basic_ops.gpu_from_host(cc))
    #dot = function([aa,bb], (cc))
    """

    # This implementation is faster (less GPU from HOST)
    a = randn(*a_shape)
    b = randn(N,N)
    aa = shared(randn(*a_shape))
    bb = shared(randn(N,N))
    c = shared(randn(N,N))
    dot = function([], updates={c: sandbox.cuda.basic_ops.gpu_from_host(aa.dot(bb))})


if args.tensor3:
    print 'Initialized %dX%dX%d matrices' % (N,N,N)
else:
    print 'Initialized %dX%d matrices' % (N,N)
    a = randn(N,N)


def test(a,b):
    ops = 0
    c = None
    for i in range(args.iters):
        dot()
        ops += flop
        print '.',
        sys.stdout.flush()
    print '!'
    return c, ops


print 'Running for %d iterations' % args.iters
t0 = time.time()
c, ops = test(a,b)
t_tot = time.time()-t0
print 'Finished in ',t_tot, 'seconds!'
flops = ops/t_tot
print 'GFlops:', flops/1e9
