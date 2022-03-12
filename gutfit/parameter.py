import numpy as np

# TODO
# 1 value: fixed
# 2 values: open range
# 3 values:mean, errup, errdn

class Parameter(object):
    def __init__(self, *args, name):
        self.name_ = name
        self.fixed_=False
        self.free_=False
        self.measurement_=False


        if len(args)==1:
            self.val_ = args[0]
            self.fixed_ = True
        elif len(args)==2:
            self.valmin_ = args[0]
            self.valmax_ = args[1]
            self.free_ = True
        elif len(args)==5:
            self.mean_ = args[0]
            self.errup_ = args[1]
            self.errdn_ = args[2]
            self.limup_ = args[3]
            self.limdn_ = args[4]
            self.measurement_ = True
            assert(self.limup_ >= self.errup_)
            assert(self.limdn_ >= self.errdn_)
        else:
            raise Exception("Number of arguments has to be either 1, 2 or 5. You provided {}".format(len(args)))

    @property
    def mean(self):
        assert(self.measurement_)
        return self.mean_

    @property
    def errup(self):
        assert(self.measurement_)
        return self.errup_

    @property
    def errdn(self):
        assert(self.measurement_)
        return self.errdn_

    @property
    def limup(self):
        assert(self.measurement_)
        return self.limup_

    @property
    def limdn(self):
        assert(self.measurement_)
        return self.limdn_

    @property
    def min(self):
        assert(self.free_)
        return self.valmin_

    @property
    def max(self):
        assert(self.free_)
        return self.valmax_

    @property
    def val(self):
        assert(self.fixed_)
        return self.val_

    @property
    def name(self):
        return self.name_

    @property
    def isfixed(self): return self.fixed_

    @property
    def isfree(self): return self.free_

    @property
    def ismeasurement(self): return self.measurement_


    def __repr__(self):
        if self.measurement_:
            return "Parameter {}:  {} + {} ({}) - {} ({})".format(self.name, self.mean, self.errup, self.limup, self.errdn, self.limdn)
        elif self.fixed_:
            return "Fixed parameter {}: {}".format(self.name_, self.val)
        elif self.free_:
            return "Free parameter {}: {} ... {}".format(self.name_, self.min, self.max)
        else:
            raise Exception("Something is wrong")

    def sample(self):
        if self.measurement_:
            # Coin toss first to see if we are in + or - land
            if np.random.random()>0.5:
                vs = abs(np.random.normal(0, self.errup))
                if vs > self.limup:
                    return self.sample()
                else:
                    return self.mean + vs
            else:
                vs = abs(np.random.normal(0, self.errdn))
                if vs > self.limdn:
                    return self.sample()
                else:
                    return self.mean - vs
        elif self.free_:
            return np.random.uniform(self.min, self.max)
        elif self.fixed_:
            return self.val
        else:
            raise Exception("Something went wrong")


    def __call__(self):
        return self.sample()
