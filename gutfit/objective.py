import numpy as np

class Objective(object):
    """
    The objective is a function of free parameter only.
    """
    def __init__(self, dim):
        self.has_gradient = hasattr(self, "gradient")
        self.bounds_ = np.empty((dim,2)) # Important
        self.dim_ = dim

    # @property
    # def has_gradient(self): return self.has_gradient

    @property
    def bounds(self):
        return self.bounds_

    def objective(self, x):
        """
        This needs to be implemented
        """
        raise Exception("Not implemented")

    def __call__(self, x):
        """
        This is what the minimiser calls
        """
        return self.objective(x)

if __name__ == "__main__":
    o = Objective(4)
