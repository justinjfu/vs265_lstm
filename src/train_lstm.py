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
            training_shaped = [np.array([point[1:] for point in blah]).reshape(len(blah),2,1) for blah in training]
            training_out = [np.array([onehot(fidx,len(files)) for i in range(len(blah))]) for blah in training_shaped]
            trainingIn.extend(training_shaped)
            trainingOut.extend(training_out)
            fidx+=1

    f, g, h = Logistic(), Logistic(), Tanh()
    lstm_layer1 = LSTMLayerWeights(10, 2, 10, f, g, h)
    lstm_layer2 = LSTMLayerWeights(6, 10, len(files), f, g, h)
    lstm_layer3 = LSTMLayerWeights(4, 6, 1, f, g, h)


    lstm = LSTMNetwork([lstm_layer1, lstm_layer2])

    with open('gestures4.lstm', 'rb') as f:
        lstm = pickle.load(f)
        pass

    wt = LSTMWeights(lstm.to_weights_array())
    obj = LSTMObjective(trainingIn, trainingOut, lstm)
    def callback(i):
        with open('gestures4.lstm', 'wb') as datfile:
            pickle.dump(lstm, datfile)
    wt = gd(obj, wt, iters=1000, heartbeat=2, learning_rate = 0.0001, momentum_rate = 0.1, callback=callback)

    print "FINAL WEIGHTS"
    final_weights = lstm.layers[0].to_weights_array()
    for final_wt in final_weights:
        print final_wt
        print ""
    with open('gestures4.lstm', 'wb') as datfile:
        pickle.dump(lstm, datfile)

