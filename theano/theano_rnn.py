import theano
import theano.tensor as T
import theano.ifelse
from theano_utils import randn, NP_FLOATX
from theano_nn import SquaredLoss, CrossEntLoss
import numpy as np

__author__ = 'justin'


class BaseLayer(object):
    n_instances = 0

    def __init__(self):
        self.layer_id = BaseLayer.n_instances
        BaseLayer.n_instances += 1

    def forward_time(self, prev_layer, prev_state, prev_output):
        """
        Run a forward pass for a single layer and single time step
        :param prev_layer:
        :param prev_state:
        :return: (next_layer, next_state)
        """
        raise NotImplementedError

    def initial_state(self):
        raise NotImplementedError

    def initial_output(self):
        raise NotImplementedError

    def params(self):
        """ Return a list of trainable parameters """
        return []


class FeedForwardLayer(BaseLayer):
    """
    A special adapter for feedforward networks
    """
    def __init__(self, n=1):
        super(FeedForwardLayer, self).__init__()
        self.zero = theano.shared(np.zeros(n).astype(NP_FLOATX))  # Has to match dimension of output

    def forward(self, prev_layer):
        raise NotImplementedError

    def forward_time(self, prev_layer, prev_state, prev_output):
        return self.forward(prev_layer), self.zero

    def initial_state(self):
        return self.zero  # Unused

    def initial_output(self):
        return self.zero  # Unused

    def params(self):
        raise NotImplementedError


class ActivationLayer(FeedForwardLayer):
    def __init__(self, act, n):
        super(ActivationLayer, self).__init__(n)
        self.act = act

    def forward(self, prev_layer):
        return self.act(prev_layer)

    def params(self):
        return []

def softmax(x):
    e_x = T.exp(x - x.max( keepdims=True))
    smax = e_x / e_x.sum( keepdims=True)
    return smax

SoftmaxLayer = ActivationLayer(softmax)
TanhLayer = ActivationLayer(T.tanh)
SigmLayer = ActivationLayer(T.nnet.sigmoid)

class FFIPLayer(FeedForwardLayer):
    """ Feedforward inner product layer """
    def __init__(self, n_in, n_out):
        super(FFIPLayer, self).__init__(n_out)
        self.w = theano.shared(0.2*randn(n_in, n_out), name="ff_ip_w_"+str(self.layer_id))
        self.b = theano.shared(0.2*randn(n_out), name="b_ip"+str(self.layer_id))

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
        self.w_ff = theano.shared(0.2*randn(n_in, n_out), name="rnn_ip_wff_"+str(self.layer_id))
        self.w_r = theano.shared(0.2*randn(n_out, n_out), name="rnn_ip_wr_"+str(self.layer_id))
        self.bias = theano.shared(0.2*randn(n_out), name="rnn_ip_bias_"+str(self.layer_id))
        self.act = act

    def forward_time(self, prev_layer, _, prev_output):
        new_state = self.w_r.dot(prev_output)
        output = self.act(prev_layer.dot(self.w_ff)+new_state+self.bias)
        return output, output+0

    def initial_output(self):
        return theano.shared(np.zeros(self.n_out).astype(NP_FLOATX))

    def initial_state(self):
        return theano.shared(np.zeros(self.n_out).astype(NP_FLOATX))

    def params(self):
        """ Return a list of trainable parameters """
        return [self.w_ff, self.w_r]


class LSTMLayer(BaseLayer):
    def __init__(self, n_input, n_out, act_f, act_g, act_h):
        super(LSTMLayer, self).__init__()

        self.n_input = n_input
        self.n_out = n_out

        self.act_f = act_f  # activation function on gates
        self.act_g = act_g  # activation function on inputs
        self.act_h = act_h  # activation function on ouputs

        def init_weights(d1, d2, name):
            return theano.shared(np.random.uniform(-0.1, 0.1, (d1, d2)).astype(NP_FLOATX), name=name+"_"+str(self.layer_id))

        self.forgetw_x = init_weights(n_input, n_out, "forgetw_x")  # forget weights from X
        self.forgetw_h = init_weights(n_out, n_out, "forgetw_h")  # forget weights from previous hidden

        self.inw_x = init_weights(n_input, n_out, "inw_x")  # input weights from X
        self.inw_h = init_weights(n_out, n_out, "inw_h") # input weights from previous hidden

        self.outw_x = init_weights(n_input, n_out, "outw_x")  # output weights from X
        self.outw_h = init_weights(n_out, n_out, "outw_h")  # output weights from previous hidden

        self.cellw_x = init_weights(n_input, n_out, "cellw_x") # cell state weights from X
        self.cellw_h = init_weights(n_out, n_out, "cellw_h")  # cell state weights from previous hidden

    def forward_time(self, previous_layer, previous_cell_state, previous_output):
        # Compute input gate
        input_a = previous_layer.dot(self.inw_x) + previous_output.dot(self.inw_h)
        input_b = self.act_f(input_a)  # Input gate outputs

        # Compute forget gate
        forget_a = previous_layer.dot(self.forgetw_x) + previous_output.dot(self.forgetw_h)
        forget_b = self.act_f(forget_a)  # Forget gate outputs

        # Compute new cell states
        a_t_c = previous_layer.dot(self.cellw_x) + previous_output.dot(self.cellw_h)
        new_cell_states = input_b * self.act_g(a_t_c) + forget_b * previous_cell_state

        # Compute output gates
        output_a = previous_layer.dot(self.outw_x) + previous_output.dot(self.outw_h)
        output_b = self.act_f(output_a)  # Input gate outputs

        # Compute new hidden layer outputs
        output = output_b * self.act_h(new_cell_states)
        return output, new_cell_states

    def initial_state(self):
        #return np.zeros((self.n_out, 1))  <---- This produces very different numbers
        return theano.shared(np.zeros(self.n_out).astype(NP_FLOATX))

    def initial_output(self):
        return theano.shared(np.zeros(self.n_out).astype(NP_FLOATX))

    def params(self):
        """ Return a list of trainable parameters """
        #return []
        return [self.forgetw_x, self.forgetw_h, self.inw_x, self.inw_h, self.outw_x, self.outw_h,
                self.cellw_x, self.cellw_h]


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
        pred = theano.function([x], self.forward_across_time(x))
        return pred

    def forward_across_time(self, training_ex):
        previous_layer = training_ex
        for layer in self.layers:
            hidden_state = layer.initial_state()
            init_state = layer.initial_output()
            def loop(prev_layer, prev_output, prev_state):
                nlayer, nstate = layer.forward_time(prev_layer, prev_state, prev_output)
                return nlayer, nstate

            results, updates = theano.scan(fn=loop,
                                            sequences=[previous_layer],
                                            outputs_info=[dict(initial=init_state, taps=[-1]), hidden_state])
            next_layer, hidden_states = results
            previous_layer = next_layer

        return previous_layer

    def prepare_objective(self, data, labels):
        # data is a list of matrices
        obj = None
        for i in range(len(data)):
            training_ex = data[i]
            label = theano.shared(labels[i])
            net_output = self.forward_across_time(theano.shared(training_ex))
            layer_loss = self.loss.loss(label, net_output)
            if obj:
                obj += layer_loss
            else:
                obj = layer_loss
        return obj

    def prepare_objective_var(self, training_ex, label):
        # data is a list of matrices
        net_output = self.forward_across_time(training_ex)
        layer_loss = self.loss.loss(label, net_output)
        obj = layer_loss
        return obj

    def prepare_objective_var_batch(self, data_tens, label_tens):
        # untested

        def loop(training_ex, label_ex, prev_sum):
            net_output = self.forward_across_time(training_ex)
            layer_loss = self.loss.loss(label_ex, net_output)
            return prev_sum+layer_loss

        init_sum = np.zeros(0).astype(NP_FLOATX)
        results, updates = theano.scan(fn=loop,
                                       sequences=[data_tens, label_tens],
                                       outputs_info=[init_sum])
        return results[-1]


def train_gd_batch(obj, params, args, batch_size=10, eta=0.01):
    gradients = T.grad(obj, params)

    ii = theano.shared(0)
    total_grad = [theano.shared(np.zeros_like(param.get_value()).astype(NP_FLOATX)) for param in params]

    updates = []

    for i in range(len(params)):
        updates.append((params[i], theano.ifelse.ifelse(T.eq(T.mod(ii, batch_size), 0),
                                                        params[i]-eta*total_grad[i],
                                                        params[i] )))
    updates.append((ii, ii+1))

    for i in range(len(params)):
        updates.append((total_grad[i], theano.ifelse.ifelse(T.eq(T.mod(ii, batch_size), 0),
                                                            T.zeros_like(total_grad[i]),
                                                            total_grad[i]+gradients[i]) ))

    train = theano.function(
        inputs=args,
        outputs=[obj, ii],
        updates=updates
    )
    return train

def train_gd_batch_momentum(obj, params, args, batch_size=10, eta=0.01, momentum=0.5):
    gradients = T.grad(obj, params)

    ii = theano.shared(0)
    total_grad = [theano.shared(np.zeros_like(param.get_value()).astype(NP_FLOATX)) for param in params]
    momentums = [theano.shared(np.zeros_like(param.get_value()).astype(NP_FLOATX)) for param in params]

    updates = []


    updates.append((ii, ii+1))
    for i in range(len(params)):
        update_gradient = eta*total_grad[i]+momentum*momentums[i]
        updates.append((params[i], theano.ifelse.ifelse(T.eq(T.mod(ii, batch_size), 0),
                                                        params[i]-update_gradient,
                                                        params[i] )))
        updates.append((total_grad[i], theano.ifelse.ifelse(T.eq(T.mod(ii, batch_size), 0),
                                                            T.zeros_like(total_grad[i]),
                                                            total_grad[i]+gradients[i]) ))
        updates.append((momentums[i], theano.ifelse.ifelse(T.eq(T.mod(ii, batch_size), 0),
                                                        update_gradient,
                                                        momentums[i] )))

    train = theano.function(
        inputs=args,
        outputs=[obj, ii],
        updates=updates
    )
    return train

def generate_parity_data(num, l):
    examples = []
    labels = []
    for i in range(num):
        N = np.random.randint(low=l, high=l+1)

        rand_data = np.random.randint(size=(N, 1), low=0, high=2).astype(np.float32)

        rand_label = np.cumsum(rand_data, axis=0) % 2
        #rand_data = np.hstack((rand_data, rand_label))

        examples.append(rand_data)
        labels.append(rand_label)
    return examples, labels


def test_parity():
    data, labels = generate_parity_data(20, 10)
    print 'Example Data:', data[0]

    emb = FFIPLayer(1, 4)  # Embedding layer
    l1 = LSTMLayer(4, 4, T.nnet.sigmoid, T.nnet.sigmoid, T.tanh)
    l2 = FFIPLayer(4, 1)
    l3 = ActivationLayer(T.nnet.sigmoid, 1)

    rnn = RecurrentNetwork([emb, l1, l2, l3], SquaredLoss())
    p = rnn.predict()

    data_var = T.matrix('data')
    label_var = T.matrix('labels')
    ob = rnn.prepare_objective_var(data_var, label_var)
    loss_func = theano.function([data_var, label_var], ob)

    print p(data[0])
    train_fn = train_gd_batch(ob, rnn.params(), [data_var, label_var], batch_size=20, eta=0.05)

    for i in range(2000):
        for j in range(len(data)):
            train_fn(data[j], labels[j])

        if i % 20 == 0:
            loss_tot = 0
            for j in range(len(data)):
                loss_tot += loss_func(data[j], labels[j])
            print i, ':', loss_tot

    data, labels = generate_parity_data(1, 25)
    print "Test example:"
    print labels[0]
    print p(data[0])

    data, labels = generate_parity_data(1, 25)
    print "Test example:"
    print labels[0]
    print p(data[0])


def one_hot(i, n, one=1.0):
    v = np.zeros(n).astype(NP_FLOATX)
    v[i] = one
    return v

def test_memorize():
    examples = []
    labels = []

    def gen_example(n, n_max, t=5):
        vecs = [one_hot(n, n_max)]
        for i in range(t):
            vecs.append(one_hot(n, n_max, one=0.0))
        data = np.vstack(vecs)

        vecs[0] = one_hot(n, n_max, one=0.0)
        vecs[-1] = one_hot(n, n_max)
        label = np.vstack(vecs)
        return data, label

    for i in range(10):
        randi = np.random.randint(low=0, high=10)
        data, label = gen_example(randi, 10)
        examples.append(data)
        labels.append(label)

    l1 = LSTMLayer(10, 20, T.nnet.sigmoid, T.nnet.sigmoid, T.tanh)
    l2 = LSTMLayer(20, 10, T.nnet.sigmoid, T.nnet.sigmoid, T.tanh)

    l3 = ActivationLayer(softmax, 10)
    rnn = RecurrentNetwork([l1, l2, l3], CrossEntLoss())
    p = rnn.predict()

    data_var = T.matrix('data')
    label_var = T.matrix('labels')
    ob = rnn.prepare_objective_var(data_var, label_var)
    loss_func = theano.function([data_var, label_var], ob)

    eta = theano.shared(np.array(0.05).astype(NP_FLOATX))
    train_fn = train_gd_batch(ob, rnn.params(), [data_var, label_var], batch_size=20, eta=eta)

    for i in range(4000):
        for j in range(len(data)):
            train_fn(examples[j], labels[j])

        if i % 20 == 0:
            loss_tot = 0
            for j in range(len(data)):
                loss_tot += loss_func(examples[j], labels[j])
            print i, ':', loss_tot

    eta.set_value(np.array(0.01).astype(NP_FLOATX))
    #train_fn = train_gd_batch_momentum(ob, rnn.params(), [data_var, label_var], batch_size=20, eta=0.005)

    for i in range(1000):
        for j in range(len(data)):
            train_fn(examples[j], labels[j])

        if i % 20 == 0:
            loss_tot = 0
            for j in range(len(data)):
                loss_tot += loss_func(examples[j], labels[j])
            print i, ':', loss_tot

    print p(examples[0]), labels[0]
    print p(examples[1]), labels[1]
    print p(examples[5]), labels[5]


if __name__ == "__main__":
    np.random.seed(10)
    #test_parity()
    test_memorize()


