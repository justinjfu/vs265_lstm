import theano
import theano.tensor as T
from theano_utils import randn, NP_FLOATX
from theano_nn import SquaredLoss, TanhLayer, SigmLayer, SoftMaxLayer
import numpy as np

__author__ = 'justin'


class BaseLayer(object):
    n_instances = 0

    def __init__(self):
        self.layer_id = BaseLayer.n_instances
        BaseLayer.n_instances += 1

    def forward_time(self, prev_layer, prev_state):
        """
        Run a forward pass for a single layer and single time step
        :param prev_layer:
        :param prev_state:
        :return: (next_layer, next_state)
        """
        raise NotImplementedError

    def initial_state(self):
        raise NotImplementedError

    def params(self):
        """ Return a list of trainable parameters """
        return []


class FeedForwardLayer(BaseLayer):
    """
    A special adapter for feedforward networks
    """
    def __init__(self):
        super(FeedForwardLayer, self).__init__()

    def forward(self, prev_layer):
        raise NotImplementedError

    def forward_time(self, prev_layer, prev_state):
        return self.forward(prev_layer), None

    def initial_state(self):
        return 0  # Unused


class ActivationLayer(FeedForwardLayer):
    def __init__(self, act):
        super(ActivationLayer, self).__init__()
        self.act = act


class FFIPLayer(FeedForwardLayer):
    """ Feedforward inner product layer """
    def __init__(self, n_in, n_out):
        super(FFIPLayer, self).__init__()
        self.w = theano.shared(randn(n_in, n_out), name="ff_ip_w_"+str(self.layer_id))
        self.b = theano.shared(randn(n_out), name="b_ip"+str(self.layer_id))

    def forward(self, prev_layer):
        return prev_layer.dot(self.w) + self.b

    def params(self):
        """ Return a list of trainable parameters """
        return [self.w, self.b]


class RNNIPLayer(BaseLayer):
    """ Recurrent inner product layer """
    def __init__(self, n_in, n_out, act):
        super(RNNIPLayer, self).__init__()
        self.n_out = n_out
        self.w_ff = theano.shared(randn(n_in, n_out), name="rnn_ip_wff_"+str(self.layer_id))
        self.w_r = theano.shared(randn(n_out, n_out), name="rnn_ip_wr_"+str(self.layer_id))
        self.act = act

    def forward_time(self, prev_layer, prev_state):
        output = self.act(prev_layer.dot(self.w_ff) + prev_state.dot(self.w_r))
        return output, output

    def initial_state(self):
        return np.zeros(self.n_out)

    def params(self):
        """ Return a list of trainable parameters """
        return [self.w_ff, self.w_r]


class LSTMLayer(BaseLayer):
    def __init__(self):
        super(LSTMLayer, self).__init__()
        #TODO: Copy implementation from code in master
        raise NotImplementedError


class RecurrentNetwork(object):
    def __init__(self, layers, loss):
        self.layers = layers
        self.loss = loss

        #setup for gradients
        self.params_list = []
        for layer in layers:
            self.params_list.extend(layer.params())

    def params(self):
        return self.params_list

    def predict(self):
        x = T.matrix('X')
        previous_layer = x
        for layer in self.layers:
            hidden_state = layer.initial_state()

            def loop(prev_layer, prev_state):
                return layer.forward_time(prev_layer, prev_state)

            results, updates = theano.scan(fn=loop,
                                            sequences=[previous_layer],
                                            outputs_info=[None, hidden_state])
            next_layer, _ = results
            previous_layer = next_layer
        pred = theano.function([x], previous_layer)
        return pred


    def prepare_objective(self, data, labels):
        # data is a list of matrices
        obj = T.scalar('objective')
        for i in range(len(data)):
            training_ex = data[i]
            label = labels[i]

            previous_layer = theano.shared(training_ex)
            for layer in self.layers:
                hidden_state = layer.initial_state()

                def loop(prev_layer, prev_state):
                    return layer.forward_time(prev_layer, prev_state)

                results, updates = theano.scan(fn=loop,
                                               sequences=[previous_layer],
                                               outputs_info=[None, hidden_state])
                next_layer, _ = results
                previous_layer = next_layer

            obj += self.loss.loss(label, previous_layer)
        return obj


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


def train_gd_host(trainable, data, labels, eta=0.01):
    obj = trainable.prepare_objective(data, labels)
    params = trainable.params()
    gradients = T.grad(obj, params)
    updates = [None]*len(gradients)

    for i in range(len(gradients)):
        updates[i] = (params[i], params[i]-eta*gradients[i])

    train = theano.function(
        outputs=[obj],
        updates=updates)
    return train


def generate_parity_data(num):
    examples = []
    labels = []
    for i in range(num):
        N = np.random.randint(low=0, high=10)

        rand_data = np.random.randint(size=(N, 1), low=0, high=2).astype(np.float32)
        rand_label = np.cumsum(rand_data, axis=0) % 2

        examples.append(rand_data)
        labels.append(rand_label)
    return examples, labels


if __name__ == "__main__":
    np.random.seed(10)

    """
    x = T.scalar('x')
    k = T.scalar('k')

    def f(i, xx):
        return xx+k, k

    results, updates = theano.scan(
        fn=f,
        outputs_info=[np.array(0.0), None],
        sequences=T.arange(5))
    blah = theano.function(inputs=[k], outputs=results, updates=updates)
    print blah(1)
    """

    """
    data = T.matrix('data')
    w = theano.shared(randn(2, 2))
    hidden = T.vector('hidden')

    def fff(i, data_slice, prev_state):
        return w.dot(data_slice)+prev_state, prev_state+1

    results, updates = theano.scan(
        fn=fff,
        outputs_info=[None, np.ones(2)],
        sequences=[T.arange(5), data])

    blah = theano.function(inputs=[data], outputs=results)

    input = randn(5,2)
    print input, input.shape
    a = blah(input)
    print a
    """

    #"""
    data, labels = generate_parity_data(10)

    l1 = RNNIPLayer(1, 1, T.nnet.sigmoid)

    rnn = RecurrentNetwork([l1], SquaredLoss())
    p = rnn.predict()

    print p(data[0])
    #train_fn = train_gd_host(rnn, data, labels, eta=0.001)

    #for i in range(100):
    #    train_fn(data, labels)

    #data, labels = generate_parity_data(1)
    #predicted = p.predict(data)
    #print predicted
    #print labels
    #"""