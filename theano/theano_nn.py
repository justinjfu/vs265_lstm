__author__ = 'justin'

import numpy as np
import time
import theano.tensor as T
import theano
import scipy.weave as wv
from theano import pp, shared, function
from theano.tensor.shared_randomstreams import RandomStreams

rng = np.random
randn = lambda *dims: rng.randn(*dims).astype(np.float32)

def theano_compile(f):
    def wrapper(*args, **kwargs):
        result = f(*args, **kwargs)
        compiled = function(args, result)
        return compiled
    return wrapper

@theano_compile
def plus(x, y, k=2):
    return x+y*k

class TheanoLayer(object):
    n_instances = 0

    def __init__(self):
        self.layer_id = TheanoLayer.n_instances 
        TheanoLayer.n_instances += 1

    def forward(self, prev_layer):
        pass

    def params(self):
        pass

# Inner product layer
class IPLayer(TheanoLayer):
    def __init__(self, n_in, n_out):
        super(IPLayer, self).__init__()
        self.w = theano.shared(randn(n_in, n_out), name="w_ip"+str(self.layer_id))

    def forward(self, previous_expr):
        return previous_expr.dot(self.w)#self.w.dot(previous_expr)

    def params(self):
        return [self.w]

class ActivationLayer(TheanoLayer):
    def __init__(self, act):
        super(ActivationLayer, self).__init__()
        self.act = act

    def forward(self, previous):
        return self.act(previous)

    def params(self):
        return []

TanhLayer = ActivationLayer(T.tanh)
SigmLayer = ActivationLayer(T.nnet.sigmoid)
SoftMaxLayer = ActivationLayer(T.nnet.softmax)

class LossLayer(object):
    pass

class SquaredLoss(LossLayer):
    def __init__(self):
        super(SquaredLoss, self).__init__()

    def loss(self, labels, predictions):
        loss = labels-predictions
        loss = T.sum(loss*loss)
        return loss

class CrossEntLoss(LossLayer):
    def __init__(self):
        super(CrossEntLoss, self).__init__()

    def loss(self, labels, predictions):
        loss = T.nnet.categorical_crossentropy(predictions, labels)
        return T.sum(loss)

class Trainable(object):
    @property
    def obj(self):
        pass
    @property
    def params(self):
        pass
    def args(self):
        pass

class Network(object):
    def __init__(self, layers, loss):
        self.layers = layers
        self.loss = loss

        self.data = T.matrix('data')
        self.labels = T.matrix('labels')
        net_out = self.data
        for layer in layers:
            net_out = layer.forward(net_out)
        self.net_out = net_out
        self.obj = loss.loss(self.labels, net_out)

        #setup for gradients
        self.params = []
        for layer in layers:
            self.params.extend(layer.params())

    def args(self):
        return [self.data, self.labels]

    def predict(self):
        return theano.function(inputs=[self.data], outputs=[self.net_out])

def train_gd(trainable, eta=0.01):
    obj = trainable.obj
    params = trainable.params
    gradients = T.grad(obj, params)
    updates = [None]*len(gradients)

    for i in range(len(gradients)):
        updates[i] = (params[i], params[i]-eta*gradients[i])

    train = theano.function(
        inputs=trainable.args(),
        outputs=[obj],
        updates=updates
    )
    return train

def train_gd_momentum(trainable, eta=0.01, momentum=0.5):
    obj = trainable.obj
    params = trainable.params
    gradients = T.grad(obj, params)

    momentums = [theano.shared(np.copy(param.get_value())) for param in params]

    updates = []
    for i in range(len(gradients)):
        updates.append((params[i], params[i]-eta*(gradients[i]+momentums[i])))
    for i in range(len(gradients)):
        updates.append((momentums[i], gradients[i]))

    train = theano.function(
        inputs=trainable.args(),
        outputs=[obj],
        updates=updates
    )
    return train

def one_hot(i, n):
    v = np.zeros(n)
    v[i] = 1
    return v

N = 10
dims = 200
data = randn(N, dims)
#labels = rng.randint(size=(1,N), low=0, high=2).astype(np.float32)
labels = np.array([ one_hot(i%2, 2) for i in range(N)]).astype(np.float32)
print labels

net = Network([IPLayer(dims, 100), TanhLayer, IPLayer(100, 2), SoftMaxLayer],
                CrossEntLoss())
train = train_gd_momentum(net)
predict = net.predict()

t = time.time()
print 'Start!'
for i in range(500):
    loss = train(data, labels)
    print i, loss[0]
print 'Time: ', time.time()-t

print 'GOLD:',labels
pred = predict(data)[0]
rounded = np.round(pred)
print 'PRED:',rounded
