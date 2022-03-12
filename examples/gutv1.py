from gutfit.objective import Objective
from gutfit.models import gutv1

class GUTV1(Objective):
    def __init__(self, params):

        self.configure(params)
        super(GUTV1, self).__init__(self.dim_)

        idx=0
        for i in range(self.data_.nfree):
            self.bounds_[idx][0] = self.data_.xmin_[i]
            self.bounds_[idx][1] = self.data_.xmax_[i]
            idx+=1
        for i in range(self.theo_.nfree):
            self.bounds_[idx][0] = self.theo_.xmin_[i]
            self.bounds_[idx][1] = self.theo_.xmax_[i]
            idx+=1


    def configure(self, pdict):
        """
        Sort out free and fixed parameters and do sanity checks.
        """
        pnames_data = ["m1"]
        pnames_theo = ["mR", "r1", "Rer2", "Imr2"]

        # Sanity check, make sure all parameters are given in pdict
        for k in pnames_data: assert(k in pdict)
        for k in pnames_theo: assert(k in pdict)

        assert(len(pnames_data) + len(pnames_theo) == len(pdict))

        self.config_theo = [ pdict[k] for k in pnames_theo]
        self.config_data = [ pdict[k] for k in pnames_data]
        self.theo_ = gutv1.GUTv1_theory(self.config_theo)
        self.data_ = gutv1.GUTv1_data(self.config_data)
        self.dim_ = self.theo_.nfree + self.data_.nfree

    @property
    def dim(self): return self.dim_

    def objective(self, x):
        assert(len(x) == self.dim)

        if self.data_.nfree>0:
            raise Exception("Not implemented")
        else:
            data = self.data_(None)
            theo = self.theo_(x)


        l2 = 0
        for d in range(self.dim_):
            l2 += (abs(data[d]) - abs(theo[d]))**2

        return l2




if __name__ == "__main__":

    pD = {
            "m1": 0.01, # 1e-5 ... 0.1,
            "mR":   [1,20],
            "r1" :  [1,10],
            "Rer2": [0,10],
            "Imr2": [-2,2],
            }

    oo = GUTV1(pD)
    print(oo((1,1,1,1)))


    from gutfit.minimiser import Minimiser

    mm = Minimiser(oo)

#    res = mm.minimise(1)
    res = mm.minimise(600)
    xbest = res["x"]

    print("Best fit point at", xbest, "with objective", oo(res["x"]))

    res = mm.minimise(100, method="lbfgsb")
    xbest = res["x"]

    print("Best fit point at", xbest, "with objective", oo(res["x"]))

    # from IPython import embed
    # embed()



