import socket, traceback, pickle
import numpy as np

def collect_data(host='', port=5555, SAMPLE_PERIOD=0.05):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    s.bind((host, port))

    sample = []
    print "Starting to Record!"
    
    mini_sample = [0,0,0] 
    message, address = s.recvfrom(8192)
    message_parts = message.replace(" ","").split(",")
    Time = float(message_parts[0])
    start_time = Time
    while True:
        try:
            message, address = s.recvfrom(8192)
            message_parts = message.replace(" ","").split(",")
            Time, AccX, AccY = float(message_parts[0]), float(message_parts[2]), float(message_parts[3])
            mini_sample[0] += 1.0
            mini_sample[1] += AccX
            mini_sample[2] += AccY
            print (message)
            if Time - start_time >= SAMPLE_PERIOD:
                averaged_sample = [Time,mini_sample[1]/mini_sample[0],mini_sample[2]/mini_sample[0]]
                print "MINI SAMPLE:", averaged_sample
                sample.append(averaged_sample)
                mini_sample = [0,0,0]
                start_time = Time

        except (KeyboardInterrupt, SystemExit):
            # pre process to only record samples in 0.1 sec intervals
            np_sample = np.array(sample)
            np_sample[:,0] -= np_sample[0,0]
            print "Done collecting a sample! Total points in this Sample: ", len(sample)
            return np_sample


    
if __name__ == "__main__":
    data = []
    while True:
        usr_in = raw_input("Press Enter to Start Recording A Sample, 'q' to Exit Program")
        if usr_in == "":
            data.append(collect_data())
        elif usr_in.lower() == "q":
            print "Exiting... Total Samples: ", len(data) 
            save_in = raw_input("Would you like to save this data? 'y' to save.")
            if save_in.lower() == "y":
                filename = raw_input("Enter name of file to save to: ")
                with open(filename, 'wb') as f:
                    pickle.dump(data, f)
            else:
                print "Did not save. Exiting..."
            break

