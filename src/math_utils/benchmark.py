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
args = parser.parse_args()

N = args.size

if args.mode == GNUMPY:
    import gnumpy as gnp
    print 'Using Gnumpy'
    randn = gnp.randn
    dot = lambda a,b: a.dot(b)

    a = randn(N,N)
    b = randn(N,N)
elif args.mode == NUMPY:
    import numpy as np
    print 'Using Numpy'
    randn = np.random.randn
    dot = lambda a,b: a.dot(b)

    a = randn(N,N).astype(np.float32)
    b = randn(N,N).astype(np.float32)
elif args.mode == THEANO:
    import numpy as np
    import theano.tensor as T
    from theano import function, shared, sandbox, config
    print 'Using Theano'
    randn = np.random.randn
    aa = T.matrix('a')
    bb = T.matrix('b')
    cc = aa.dot(bb)
    dot = function([aa,bb],sandbox.cuda.basic_ops.gpu_from_host(cc))

    a = randn(N,N).astype(np.float32)
    b = randn(N,N).astype(np.float32)

print 'Initializing %dX%d matrices' % (N,N)

def test(a,b):
    ops = 0
    for i in range(args.iters):
        c= dot(a,b)
        ops += 2*N*N*N
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
