from lstm import *
import numpy as np
from client_test import collect_data
import argparse, pickle
from activations import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train an LSTMNetwork")
    parser.add_argument('LSTMNetworks')
    parser.add_argument('testFiles')
    args = parser.parse_args()
    classes = "M,Z,O,L,S,J".split(',')

    networkfiless = args.LSTMNetworks.split(',')

    networks = []
    for netfile in networkfiless:
        with open(netfile, 'rb') as f:
            network = pickle.load(f)
            networks.append(network)
    #print lstm.to_weights_array()

    fidx = 0
    files = args.testFiles.split(',')
    for file in files:
        with open(file, 'rb') as f:
            training = pickle.load(f)
            training_shaped = [np.array([point[1:] for point in blah]).reshape(len(blah),2,1) for blah in training]

            good = 0
            for i in range(10):
                sumval = 0
                for lstm in networks:
                    output = lstm.forward_across_time(training_shaped)
                    netout = Softmax.softmax(output[i])
                    netout = np.sum(netout, axis=0)
                    sumval = netout + sumval
                sumval = sumval/np.sum(sumval)
                argmax = np.argmax(sumval)

                if argmax == fidx:
                    #print 'GOOD:',fidx
                    good +=1
                else:
                    pass
                    #print 'BAD:', fidx
            print classes[fidx], ' - # Correct:', good

        fidx+=1





