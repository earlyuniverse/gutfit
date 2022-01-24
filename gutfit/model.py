import numpy as np

class Model(object):
    def __init__(self, pars):
        """
        Base evaluator class.
        Ensures consistent input.
        pars ... list of tuples. Parameters are either free or fixed.
        """
        self.dim_ = len(pars)
        self.fixed_, self.free_  = [], []

        for num, p in enumerate(pars):
            if (isinstance(p, tuple)) or (isinstance(p, list)):
                if len(p) == 2: self.free_.append(num)
                else:
                    raise Exception("A free parameter must have exactly two values, not {}".format(len(p)))
            elif isinstance(p,int) or isinstance(p, float):
                self.fixed_.append(num)
            else:
                raise Exception("Input type {} of dimension {} not understood".format(type(p), num ))


        self.nfixed_ = len(self.fixed_ )
        self.nfree_  = len(self.free_)
        if not(self.nfixed_ +  self.nfree_ == self.dim_):
            raise Exception("Invalid setup")

        self.x_ = np.empty(self.dim_)
        for num, i in enumerate(self.fixed_): self.x_[num] = pars[i]

        self.xmin_ = np.empty(self.nfree_)
        self.xmax_ = np.empty(self.nfree_)
        for num, i in enumerate(self.free_):
            self.xmin_[num] = pars[i][0]
            self.xmax_[num] = pars[i][1]

    @property
    def dim(self): return self.dim_

    @property
    def nfree(self): return self.nfree_

    @property
    def nfixed(self): return self.nfixed_

    @property
    def val(self):
        """ Implement and return model prediction here using self.x_ as parameter point """
        return sum(self.x_)

    def setParams(self, x, checks=False):
        assert(len(x) == self.nfree)
        for num, xx in enumerate(x):
            idx = self.free_[num]
            self.x_[idx] = xx
            if checks:
                if xx < self.xmin_[idx] or xx > self.xmax_[idx]:
                    raise Exception("Par{} = {} outside bound [{},{}]".format(
                        num, xx, self.xmin_[idx], self.xmax_[idx]))

    def __call__(self, x, checks=False):
        """
        Evaluate model at point x where x are the free parameters.
        """
        if self.nfree>0:
            self.setParams(x, checks)
        return self.val

    def __str__(self):
        s="{}D-Model with {} free parameters, {} fixed parameters\n".format(self.dim, self.nfree, self.nfixed)

        for d in range(self.dim):
            if d in self.fixed_:
                s+= "Dim {} =   {}\n".format(d, self.x_[self.fixed_.index(d)])
            else:
                s+= "Dim {} in [{},{}]\n".format(d, self.xmin_[self.free_.index(d)], self.xmax_[self.free_.index(d)])

        return s

    def readExperimentalDataCSV(self, fname, ):
        import csv
        with open(fname, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=' ', quotechar='|')
            for row in reader:
                print(', '.join(row))


if __name__ == "__main__":

    m = Model([
        (1,2),
        (2,3),
        4
        ])

    print(m)
    print(m([2,3]))

