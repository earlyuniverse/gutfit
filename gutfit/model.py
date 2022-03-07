import numpy as np

class Model(object):
    def __init__(self, parnames):
        """
        Base evaluator class.
        Ensures consistent input.
        """
        self.pnames_ = parnames
        self.dim_ = len(self.pnames_)

    @property
    def dim(self): return self.dim_

    def __call__(self, xdict):
        self.__dict__.update((k, v) for k, v in xdict.items() if k in self.pnames_)
        return self.val

    @property
    def val(self):
        raise Exception("val method must be implemented in derived class")
