class Objective(object):
    """
    Abstract class for training an objective
    """

    def __init__(self):
        pass

    def gradient_at(self, wts):
        """
        Return a Weights object holding the gradient evaluated at 
        the point wts
        """
        raise NotImplemented

    def value_at(self, wts):
        """
        Evaluate the objective at a the point wts
        """
        raise NotImplemented

    def save_to_file(self, filename, weights):
        """
        Save weights to a file
        """
        raise NotImplemented

    def read_from_file(self, filename, weights):
        """
        Read weights from a file
        """
        raise NotImplemented

class Weights(object):
    def __init__(self):
        pass

    def __add__(self, other):
        # For operator overloading
        return self.add_weight(other)

    def __mul__(self, other):
        # For operator overloading
        return self.mul_scalar(other)

    def add_weight(self, other_weight):
        """
        Add this weight object to another weight object
        """
        raise NotImplemented

    def mul_scalar(self, other):
        """
        Multiply the values in this weight object by a
        single scalar
        """
        raise NotImplemented

