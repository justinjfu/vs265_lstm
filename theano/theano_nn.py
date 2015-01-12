__author__ = 'justin'

import numpy as np
import time
import theano.tensor as T
import theano
import scipy.weave as wv
from theano import pp, shared, function
from theano.tensor.shared_randomstreams import RandomStreams
from theano_utils import gpu_host
from watchers import *

rng = np.random
randn = lambda *dims: rng.randn(*dims).astype(np.float32)

class TheanoLayer(object):
    n_instances = 0
    def __init__(self):
        self.layer_id = TheanoLayer.n_instances 
        TheanoLayer.n_instances += 1

    def forward(self, prev_layer):
        raise NotImplemented

    def params(self):
        """ Return a list of trainable parameters """
        return []

    def __getstate__(self):
        """ Serialization """
        return None

    def __setstate__(self, state):
        """ Deserialization """
        pass

class IPLayer(TheanoLayer):
    """ Inner product layer """
    def __init__(self, n_in, n_out):
        super(IPLayer, self).__init__()
        self.w = theano.shared(randn(n_in, n_out), name="w_ip"+str(self.layer_id))
        self.b = theano.shared(randn(n_out), name="b_ip"+str(self.layer_id))

    def forward(self, previous_expr):
        return previous_expr.dot(self.w)+self.b

    def params(self):
        return [self.w]

    def __getstate__(self):
        return (self.w, self.b)

    def __setstate__(self, state):
        w, b = state
        self.w = w
        self.b = b

class ActivationLayer(TheanoLayer):
    """ Activation layer """
    def __init__(self, act):
        super(ActivationLayer, self).__init__()
        self.act = act

    def forward(self, previous):
        return self.act(previous)


TanhLayer = ActivationLayer(T.tanh)
SigmLayer = ActivationLayer(T.nnet.sigmoid)
SoftMaxLayer = ActivationLayer(T.nnet.softmax)
ReLULayer = ActivationLayer(T.nnet.softplus)

class LossLayer(object):
    def loss(self, labels, predictions):
        """
        Return loss summed across training examples
        """
        raise NotImplemented

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


class Network(object):
    def __init__(self, layers, loss):
        self.layers = layers
        self.loss = loss

        self.data = T.matrix('data')
        self.labels = T.matrix('labels')
        self.net_out, self.train_obj = self.prepare_objective(self.data, self.labels)

        #setup for gradients
        self.params_list = []
        for layer in layers:
            self.params_list.extend(layer.params())

    def prepare_objective(self, data, labels):
        net_out = data
        for layer in self.layers:
            net_out = layer.forward(net_out)
        net_out = net_out
        obj = self.loss.loss(labels, net_out)
        return net_out, obj

    def params(self):
        return self.params_list

    def obj(self):
        return self.train_obj

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
        outputs=[gpu_host(obj)],
        updates=updates
    )
    return train

def train_gd_momentum(trainable, eta=0.01, momentum=0.5):
    obj = trainable.obj()
    params = trainable.params()
    gradients = T.grad(obj, params)
    eta = np.array(eta).astype(np.float32)
    momentum = np.array(momentum).astype(np.float32)

    momentums = [theano.shared(np.copy(param.get_value())) for param in params]

    updates = []
    for i in range(len(gradients)):
        update_gradient = eta*(gradients[i])+momentum*momentums[i]
        updates.append((params[i], gpu_host(params[i]-update_gradient)))
        updates.append((momentums[i], gpu_host(update_gradient)))

    train = theano.function(
        inputs=trainable.args(),
        outputs=[obj],
        updates=updates
    )
    return train

def train_gd_momentum_host(trainable, data, labels, eta=0.01, momentum=0.8):
    params = trainable.params()
    eta = np.array(eta).astype(np.float32)
    momentum = np.array(momentum).astype(np.float32)


    data = theano.shared(data)
    labels = theano.shared(labels)
    _, obj = trainable.prepare_objective(data, labels) 

    gradients = T.grad(obj, params)

    momentums = [theano.shared(np.zeros(param.get_value().shape).astype(np.float32)) for param in params]

    updates = []
    for i in range(len(gradients)):
        update_gradient = eta*(gradients[i])+momentum*momentums[i]
        updates.append((params[i], gpu_host(params[i]-update_gradient)))
        updates.append((momentums[i], gpu_host(update_gradient)))

    train = theano.function(
        inputs=[],
        outputs=[obj],
        updates=updates
    )
    return train

if __name__ == "__main__":
    def one_hot(i, n):
        v = np.zeros(n)
        v[i] = 1
        return v

    N = 1000
    dims = 200
    data = randn(N, dims)
    #labels = rng.randint(size=(1,N), low=0, high=2).astype(np.float32)
    labels = np.array([ one_hot(i%2, 2) for i in range(N)]).astype(np.float32)

    net = Network([IPLayer(dims, 1000), TanhLayer, IPLayer(1000,300), TanhLayer, IPLayer(300, 100), TanhLayer, IPLayer(100,2), SoftMaxLayer],
                    CrossEntLoss())
    predictor = net.predict()

    optimizer = FOptimizer(train_gd_momentum_host, net, data, labels, eta=0.0001)
    #optimizer = FOptimizer(train_gd_momentum, net, eta=0.00001)
    optimizer.addWatcher(InfoWatcher(OnIter(5)))
    #optimizer.addWatcher(PickleWatcher(net, "net.dat", OnTime(10)))
    optimizer.addWatcher(TimeWatcher(OnEnd()))

    optimizer.optimize(400) # 300 iters
    #optimizer.optimize(50, data, labels) # 300 iters

