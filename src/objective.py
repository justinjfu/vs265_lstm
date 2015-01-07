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

class Weights(object):
    def __init__(self):
        pass

    def __add__(self, other):
        # For operator overloading
        return self.add_weight(other)

    def __mul__(self, other):
        # For operator overloading
        if isinstance(other, Weights):
            return self.dot_weight(other)
        return self.mul_scalar(other)

    def add_weight(self, other_weight):
        """
        Add this weight object to another weight object
        """
        raise NotImplemented

    def dot_weight(self, other):
        """
        Dot product with another weight
        """
        raise NotImplemented

    def mul_scalar(self, other):
        """
        Multiply the values in this weight object by a
        single scalar
        """
        raise NotImplemented

    def save_to_file(self, filename_prefix, iteration):
        """
        Save weights to a file
        :param filename_prefix:
        :param iteration: Descend iteration #
        """
        raise NotImplemented

    @classmethod
    def read_from_file(self, filename):
        """
        Read weights from a file
        """
        raise NotImplemented

