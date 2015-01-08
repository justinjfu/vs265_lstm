"""
LSTM Objective, weights, and network
"""
import pickle
import math_interface as np
from objective import Objective, Weights
from activations import Logistic, Tanh
from loss import Squared
from layers import NNLayer, LSTMLayerWeights, LSTMWeights
from descend import gd


class LSTMObjective(Objective):
    def __init__(self, trainingIn, trainingOut, network, l2reg=0.0):
        super(LSTMObjective, self).__init__()
        self.training_in = trainingIn
        self.training_out = trainingOut
        self.network = network
        self.l2reg = l2reg

    def gradient_at(self, wts):
        self.network.set_weights(wts.wts)
        gradient = self.network.gradient(self.training_in, self.training_out)
        return LSTMObjectiveWeights(gradient)+wts*self.l2reg

    def value_at(self, wts):
        self.network.set_weights(wts.wts)
        err, _ = self.network.eval_objective(self.training_in, self.training_out)
        return err + wts*wts*self.l2reg


class LSTMObjectiveWeights(Weights):
    def __init__(self, wts):
        super(LSTMObjectiveWeights, self).__init__()
        self.wts = wts

    def add_weight(self, other_weight):
        return LSTMObjectiveWeights([(self.wts[L] + other_weight.wts[L]) for L in range(len(self.wts))])
        
    def mul_scalar(self, other_scalar):
        return LSTMObjectiveWeights([layer * other_scalar for layer in self.wts])

    def dot_weight(self, other):
        total = 0
        for L in range(len(self.wts)):
            other_mat = other.wts[L]
            self_mat = self.wts[L]
            total += self_mat.dot_weight(other_mat)
        return total

    def save_to_file(self, filename_prefix, _):
        with open(filename_prefix, 'wb') as netfile:
            pickle.dump(self.network, netfile)

    @classmethod
    def read_from_file(cls, filename):
        net = LSTMObjectiveWeights(None)
        with open(filename, 'rb') as netfile:
            net.network = pickle.load(netfile)
        return net

    def __str__(self):
        raise NotImplemented


class LSTMNetwork(object):
    def __init__(self, layers, loss):
        self.layers = layers
        self.loss = loss
    
    def mul_scalar(self, scalar):
        for layer in self.layers:
            layer.mul_scalar(scalar)

    def add_weight(self, other_network):
        for i in range(len(self.layers)):
            self.layers[i].update_layer_weights( other_network.layers[i].to_compact_weights())

    def forward_across_time(self, inputs):
        """
        Run forward pass on the input, computing network outputs at each timestep
        :param input: A N x IN_DIM x T input. IN_DIM is the input dimension of the network
        :return: An N x OUT_DIM x T output. OUT_DIM is the output dimension of the network
        """
        all_outputs = []
        for i in range(len(inputs)):
            current_in = inputs[i]
            for layer in self.layers:
                intermediates = layer.forward_across_time(current_in)
                current_in = np.array([intermed.output for intermed in intermediates])
            all_outputs.append(current_in)
        return all_outputs

    def gradient(self, inputs, outputs):
        Nlayers = len(self.layers)
        layer_intermediates = [None]*Nlayers
        gradients = []
        for layer in self.layers:
            weights_in_layer = layer.make_zero_weights() #[np.zeros(weight.shape) for weight in layer.to_compact_weights()]
            gradients.append(weights_in_layer)
        for i in range(len(inputs)):  # loop over training examples
            current_in = inputs[i]
            for j in range(len(self.layers)):
                intermediates = self.layers[j].forward_across_time(current_in)
                layer_intermediates[j] = intermediates
                current_in = np.array([intermed.output for intermed in intermediates])
            final_layer_output = np.array([intermed.output for intermed in layer_intermediates[Nlayers-1]])

            next_layer_del_k = self.output_backprop_error(final_layer_output, outputs[i])
            for j in range(len(self.layers))[::-1]:
                gradient, next_layer_del_k = self.layers[j].gradient(layer_intermediates[j], next_layer_del_k)
                gradients[j] += gradient
                #for k in range(len(gradients[j])):
                #    gradients[j][k] += gradient[k]
        return gradients

    def update_layer_weights(self, d_weights, K=1):
        for i in range(len(self.layers)):
            self.layers[i].update_layer_weights(d_weights[i], K=K)

    def eval_objective(self, trainingIn, trainingOut):
        outputs = self.forward_across_time(trainingIn)
        error = 0
        for i in range(len(trainingIn)):
            labels = trainingOut[i].reshape(outputs[i].shape)
            predicted = outputs[i]
            error += self.loss.eval(predicted, labels)
        return error, outputs

    def output_backprop_error(self, output, trainingOut):
        labels = trainingOut.reshape(output.shape)
        return self.loss.backward(output, labels)

    def to_compact_weights(self):
        return [x.to_compact_weights() for x in self.layers]

    def set_weights(self, dwts):
        for i in range(len(self.layers)):
            self.layers[i].set_weights(dwts[i])

    def numerical_gradient(self, d_weights, trainingIn, trainingOut, perturb_amount = 1e-5):
        #original_objective, _ = self.eval_objective(trainingIn, trainingOut)
        for i in range(len(self.layers)):
            lstm = self.layers[i]
            weights = lstm.to_compact_weights()
            dweights = d_weights[i]
            for wi in range(len(weights)):  # need to iterate through each weight independently
                weight = weights[wi]
                dweight = dweights[wi]  # dweight is where we store the gradients for each particular weight
                for index, val in np.ndenumerate(weight):  # a 'single' weight is a matrix, so we iterate within as well
                    weight[index] -= perturb_amount  # w_ij - e
                    lessObj, _ = self.eval_objective(trainingIn, trainingOut)
                    #lessObj = original_objective

                    weight[index] += 2* perturb_amount  # w_ij + e
                    moreObj, _ = self.eval_objective(trainingIn, trainingOut)
                    diff = moreObj - lessObj  # difference
                    # import pdb; pdb.set_trace()
                    grad = 1 * diff / (2* perturb_amount)  # negative gradient
                    weight[index] -= perturb_amount  # set weight back to normal
                    dweight[index] = grad  # save gradient, will update all the weights at the end

    
if __name__ == '__main__':
    np.random.seed(20)
    # test on bitstring parity checker - tests feed forward only with numerical gradient calculation
    N = 3
    trainingIn0 = np.array([[1, 0, 1] * N, [0, 0, 0 ]*N]).T.reshape(3*N,2,1)
    trainingIn1 = np.array([[1, 1, 1, 1, 1, 1, 0, 1] * N, [0,0,0,0,0,0,0,0]*N]).T.reshape(8*N,2,1)
    trainingIn2 = np.array([[1, 0, 1, 0] * N, [0,0,0,0]*N]).T.reshape(4*N,2,1)
    trainingIn3 = np.array([[0, 0, 0, 1, 0, 1, 1, 1] * N, [0,0,0,0,0,0,0,0]*N]).T.reshape(8*N,2,1)
    trainingIn4 = np.array([[1, 0, 1, 1, 1, 1] * N, [0,0,0,0,0,0]*N]).T.reshape(6*N,2,1)
    trainingIn5 = np.array([[0, 1, 0, 0, 1, 0, 1] * N, [0,0,0,0,0,0,0]*N]).T.reshape(7*N,2,1)

    trainingIn = [trainingIn1, trainingIn2, trainingIn3, trainingIn4, trainingIn5]
    #trainingIn = [trainingIn0]

    trainingOut0 = (np.cumsum(trainingIn0, axis=0) % 2)[:,0,:]
    trainingOut1 = (np.cumsum(trainingIn1, axis=0) % 2)[:,0,:]
    trainingOut2 = (np.cumsum(trainingIn2, axis=0) % 2)[:,0,:]
    trainingOut3 = (np.cumsum(trainingIn3, axis=0) % 2)[:,0,:]
    trainingOut4 = (np.cumsum(trainingIn4, axis=0) % 2)[:,0,:]
    trainingOut5 = (np.cumsum(trainingIn5, axis=0) % 2)[:,0,:]

    trainingOut = [trainingOut1, trainingOut2, trainingOut3, trainingOut4, trainingOut5]
    #trainingOut = [trainingOut0]

    f, g, h = Logistic(), Logistic(), Tanh()
    lstm_layer1 = LSTMLayerWeights(2, 4, f, g, h)
    lstm_layer2 = NNLayer(4, 4, Tanh(), usebias=False)
    lstm_layer3 = NNLayer(4, 1, Tanh(), usebias=False)
    #lstm_layer1 = LSTMLayerWeights(2, 4, f, g, h)
    #lstm_layer2 = NNLayer(4, 1, h)
    #d_weight1 = [np.zeros(w.shape) for w in lstm_layer1.to_weights_array()]
    #d_weight2 = [np.zeros(w.shape) for w in lstm_layer2.to_weights_array()]
    #d_weights = [d_weight1, d_weight2]
    #d_weights = [d_weight1]

    lstm = LSTMNetwork([lstm_layer1, lstm_layer2, lstm_layer3], loss=Squared())
    #lstm = LSTMNetwork([lstm_layer1, nn_layer1])


    """
    for trial in range(200):
        #import pdb; pdb.set_trace()
        #lstm.numerical_gradient(d_weights, trainingIn, trainingOut, perturb_amount = 5e-6)
        d_weights = lstm.gradient(trainingIn, trainingOut)
        lstm.update_layer_weights(d_weights, K=-0.5)
        if (trial+1) % 100 == 0:
            err, output = lstm.eval_objective(trainingIn, trainingOut)
            print "Trial =", trial+1
            print err
            print output
    """



    wt = LSTMObjectiveWeights(lstm.to_compact_weights())
    obj = LSTMObjective(trainingIn, trainingOut, lstm, l2reg=0.000)
    wt = gd(obj, wt, iters=2000, heartbeat=200, learning_rate=0.05, momentum_rate=0.5)


    print "FINAL WEIGHTS"
    for layer_id in range(len(lstm.layers)):
        final_weights = lstm.layers[layer_id].to_compact_weights()
        for final_wt in final_weights:
            print final_wt
            print ""
    print "TESTING INPUT NAO"
    N=1
    testIn1 = np.array([[1, 0, 1] * N, [0,0,0]*N]).T.reshape(3,2,1)
    testIn2 = np.array([[1, 0, 0, 1] * N, [0,0,0,0]*N]).T.reshape(4,2,1)
    testIn3 = np.array([[1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1] * N, [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]*N]).T.reshape(23,2,1)
    testIn = [testIn1, testIn2, testIn3]
    #testIn = [testIn1, testIn2]
    testOut1 = (np.cumsum(testIn1, axis=0) % 2)[:,0,:]
    testOut2 = (np.cumsum(testIn2, axis=0) % 2)[:,0,:]
    testOut3 = (np.cumsum(testIn3, axis=0) % 2)[:,0,:]

    testOut = [testOut1, testOut2, testOut3]
    #testOut = [testOut1, testOut2]
    obj, output = lstm.eval_objective(testIn, testOut)
    print testOut
    print output
    print obj


