from lstm import *
import numpy as np
from client_test import collect_data
import argparse, pickle


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train an LSTMNetwork")
    parser.add_argument('LSTMNetwork')
    args = parser.parse_args()
    
    class1 = "Z"
    class2 = "O"
    THRESH = 0.5
   
    with open(args.LSTMNetwork, 'rb') as f:
        lstm = pickle.load(f)

    while True:
        usr_in = raw_input("Press Enter to Start Recording A Sample and Test it, 'q' to Exit Program")
        if usr_in == "":
            trainingIn = collect_data()
            gravity = np.array([0,9.8])

            trainingIn_shaped = [np.array([point[1:] - gravity for point in trainingIn]).reshape(len(trainingIn),2,1)]
            output = lstm.forward_across_time(trainingIn_shaped)[0]
            val = np.sum(output)/len(output)
            if val > THRESH:
                print "RECOGNIZED:",class1
            else:
                print "RECOGNIZED:",class2
            print val
        elif usr_in.lower() == "q":
            break





