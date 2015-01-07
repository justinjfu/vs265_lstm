"""
Implementations of optimization algorithms
"""
import time


def gd(objective, initial, iters=100, heartbeat=10, save_to_file=None,
       learning_rate=1e-2, momentum_rate=0.0, callback=None):
    """
    Perform a simple gradient descent with momentum.

    :param objective: An Objective object
    :param initial: Initial weights
    :param iters: Number of iterations to descend
    :param heartbeat: Rate at which to print a status message and save intermediate
        weights (if save_to_file is specified)
    :param learning_rate: Scaling factor on gradient
    :param momentum_rate: Scaling factor on momentum
    :param save_to_file: Filename prefix to save intermediate weights to. If left None, does
        not save
    """
    start_time = time.time()
    last_time = start_time

    weights = initial
    last_gradient = None
    for i in range(iters):
        gradient = objective.gradient_at(weights) * -learning_rate
        weights += gradient

        if last_gradient:
            weights += last_gradient * momentum_rate
        last_gradient = gradient

        if i % heartbeat == 0:
            current_time = time.time()
            print '[descend.gd] %f(%f) Completed %d iterations. Current objective: %f' % \
                (current_time, current_time-last_time, i, objective.value_at(weights))
            last_time = current_time
            if save_to_file:
                weights.save_to_file(save_to_file, i)
            if callback:
                callback(i)

    end_time = time.time()
    print 'Total time elapsed: ', end_time-start_time
    return weights
