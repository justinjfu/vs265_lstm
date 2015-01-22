from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
import theano_utils

__author__ = 'justin'

import numpy as np
import theano.tensor as T
import theano
import scipy.weave as wv
from theano import pp, shared, function
from theano.tensor.shared_randomstreams import RandomStreams
from theano_utils import gpu_host, NP_FLOATX, randn
from watchers import *

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
        return [self.w, self.b]

    def __getstate__(self):
        return self.w, self.b

    def __setstate__(self, state):
        w, b = state
        self.w = w
        self.b = b


class ConvPoolLayer(object):
    def __init__(self, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a ConvPoolLayer with shared variable internal parameters.

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """
        assert image_shape[1] == filter_shape[1]
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
                   np.prod(poolsize))
        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            np.asarray(
                theano_utils.rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )
        # the bias is a 1D tensor -- one bias per output feature map
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

    def change_batch_size(self, new_batch):
        raise NotImplementedError

    def forward(self, prev_layer):
        prev_layer = prev_layer.reshape(self.image_shape)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=prev_layer,
            filters=self.W,
            filter_shape=self.filter_shape,
            image_shape=self.image_shape
        )
        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=self.poolsize,
            ignore_border=True
        )
        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        return T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

    def params(self):
        # store parameters of this layer
        return [self.W, self.b]


class Flatten2DLayer(TheanoLayer):
    def __init__(self):
        super(Flatten2DLayer, self).__init__()

    def forward(self, previous_expr):
        return previous_expr.flatten(2)


class ActivationLayer(TheanoLayer):
    """ Activation layer """
    def __init__(self, act):
        super(ActivationLayer, self).__init__()
        self.act = act

    def forward(self, previous):
        return self.act(previous)

    def __getstate__(self):
        return self.act

    def __setstate__(self, state):
        self.act = state

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

    def predict(self, data):
        net_out = data
        for layer in self.layers:
            net_out = layer.forward(net_out)
        net_out = net_out
        return theano.function(inputs=[data], outputs=[net_out])


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
    eta = np.array(eta).astype(NP_FLOATX)
    momentum = np.array(momentum).astype(NP_FLOATX)

    _, obj = trainable.prepare_objective(data, labels) 

    gradients = T.grad(obj, params)

    momentums = [theano.shared(np.zeros(param.get_value().shape).astype(NP_FLOATX)) for param in params]

    updates = []
    for i in range(len(gradients)):
        update_gradient = eta*(gradients[i])+momentum*momentums[i]
        updates.append((params[i], gpu_host(params[i]-update_gradient)))
        updates.append((momentums[i], gpu_host(update_gradient)))

    train = theano.function(
        inputs=[data, labels],
        outputs=[obj],
        updates=updates
    )
    return lambda d, l: train(d, l)[0]


def run_optimize_simple(train_fn, data, labels, iters, batch_size=500):
    epoch = 0
    N = data.shape[0]
    batches_per_epoch = N/batch_size
    n_iter = 0
    while True:
        for i in range(batches_per_epoch):
            start_batch = i*batch_size
            end_batch = (i+1)*batch_size
            loss = train_fn(data[start_batch:end_batch],
                            labels[start_batch:end_batch])
            n_iter += 1
            print n_iter,':', loss*10
            if n_iter >= iters:
                return
        print 'Finished Epoch ', epoch
        epoch += 1

if __name__ == "__main__":
    import mnist
    bsize = 500
    train, valid, test = mnist.get_mnist()

    train_data, train_labels = mnist.preprocess(train)
    N = train_data.shape[0]


    nkerns=[20, 50]
    net = Network([ConvPoolLayer(image_shape=(bsize, 1, 28, 28),
                                 filter_shape=(nkerns[0], 1, 5, 5),
                                 poolsize=(2, 2)),
                   ConvPoolLayer(image_shape=(bsize, nkerns[0], 12, 12),
                                 filter_shape=(nkerns[1], nkerns[0], 5, 5),
                                 poolsize=(2, 2)),
                   Flatten2DLayer(),
                   IPLayer(nkerns[1] * 4 * 4, 500),
                   TanhLayer,
                   IPLayer(500,10),
                   SoftMaxLayer],
                   CrossEntLoss())

    with open('ip.network', 'rb') as netfile:
        net = cPickle.load(netfile)

    c1 = net.layers[0]
    c2 = net.layers[1]
    #net.layers[4] = ActivationLayer(lambda x: 2*T.nnet.sigmoid(x)-1)

    if False:
        optimizer = FOptimizer(train_gd_momentum_host, net, data, labels, eta=0.0001)
        #optimizer = FOptimizer(train_gd_momentum, net, eta=0.00001)
        optimizer.add_watcher(InfoWatcher(OnIter(5)))
        #optimizer.add_watcher(PickleWatcher(net, "net.dat", OnTime(10)))
        optimizer.add_watcher(TimeWatcher(OnEnd()))
        optimizer.iterate(400) # 300 iters
        #optimizer.optimize(50, data, labels) # 300 iters
    else:
        vdata = T.matrix('data')
        #vdata = vdata.reshape((bsize, 1, 28, 28))
        vlabels = T.matrix('labels')
        train_fn = train_gd_momentum_host(net, vdata, vlabels, eta=0.00015, momentum=0.1)
        run_optimize_simple(train_fn, train_data, train_labels, 100, batch_size=bsize)

    with open('ip.network', 'wb') as netfile:
        cPickle.dump(net, netfile)

    test_data, test_labels = mnist.preprocess(test)
    tdata = T.matrix('test_data')
    #tdata = tdata.reshape(test_data.shape)
    c1.image_shape = (10000, c1.image_shape[1], c1.image_shape[2], c1.image_shape[3])
    c2.image_shape = (10000, c2.image_shape[1], c2.image_shape[2], c2.image_shape[3])
    print test_data.shape
    predictor = net.predict(tdata)
    predictions = predictor(test_data)[0]
    print mnist.error(predictions, test_labels)
