def gd(objective, initial, iters=100, heartbeat=10, save_to_file=None,
       learning_rate=1e-2, momentum_rate=0.3):
    """
    Perform a simple gradient descent.

    :param objective: An Objective object
    :param initial: Initial weights
    :param iters: Number of iterations to descend
    :param hearbeat: Rate at which to print a status message and save intermediate
        weights (if save_to_file is specified)
    :param save_to_file: Filename to save intermediate weights to. If left None, does
        not save
    """
    weights = initial
    last_gradient = None
    for i in range(iters):
        gradient = objective.gradient_at(weights) * -learning_rate
        weights += gradient

        if last_gradient:
            weights += last_gradient * momentum_rate
        last_gradient = gradient

        if i % heartbeat == 0:
            print 'Completed %d iterations. Objective: %f' % (i, objective.value_at(weights))
            if save_to_file:
                weights.save_to_file(save_to_file)

    return weights
