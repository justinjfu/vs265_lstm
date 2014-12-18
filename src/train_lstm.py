from lstm import *
from activations import *
import numpy as np
from descend import gd
import argparse, pickle, socket, traceback 

def onehot(i, N):
    v = [0]*N
    v[i] = 1.0
    return v

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train an LSTMNetwork")
    parser.add_argument('trainingFiles')

    args = parser.parse_args()

    trainingIn = []
    trainingOut = []

    fidx=0
    files = args.trainingFiles.split(',')
    for file in files:
        with open(file, 'rb') as f:
            training = pickle.load(f)
            #import pdb; pdb.set_trace()
            training_shaped = [np.array([point[1:] for point in blah]).reshape(len(blah),2,1) for blah in training]
            training_out = [np.array([onehot(fidx,len(files)) for i in range(len(blah))]) for blah in training_shaped]
            trainingIn.extend(training_shaped)
            trainingOut.extend(training_out)
            fidx+=1

    f, g, h = Logistic(), Logistic(), Tanh()
    lstm_layer1 = LSTMLayerWeights(8, 2, len(files), f, g, h)
    lstm_layer2 = LSTMLayerWeights(10, 10, 3, f, g, h)
    lstm_layer3 = LSTMLayerWeights(4, 6, 1, f, g, h)


    lstm = LSTMNetwork([lstm_layer1])

    d_weight1 = [np.zeros(w.shape) for w in lstm_layer1.to_weights_array()]
    d_weights = [d_weight1]

    """
    for trial in range(100):
        #import pdb; pdb.set_trace()
        #lstm.numerical_gradient(d_weights, trainingIn, trainingOut, perturb_amount = 1e-5)
        d_weights = lstm.gradient(trainingIn, trainingOut)
        lstm.update_layer_weights(d_weights, K=-0.0001)
        if (trial+1) % 1 == 0:
            err, output = lstm.eval_objective(trainingIn, trainingOut)
            print "Trial =", trial+1
            print err
            #print output
            print Softmax.softmax(output[-1][-1])
            print trainingOut[-1][-1]
    """

    wt = LSTMWeights(lstm.to_weights_array())
    obj = LSTMObjective(trainingIn, trainingOut, lstm)
    wt = gd(obj, wt, iters=500, heartbeat=5, learning_rate = 0.0005, momentum_rate = 0.1)

    print "FINAL WEIGHTS"
    final_weights = lstm.layers[0].to_weights_array()
    for final_wt in final_weights:
        print final_wt
        print ""
    with open('gestures.lstm', 'wb') as datfile:
        pickle.dump(lstm, datfile)

