import numpy as np

class Minimiser(object):
    def __init__(self, objective):
        self.objective_ = objective
        if objective.has_gradient:
            self.gradient_ = objective.gradient
            self.has_gradient = True
        else:
            self.gradient_ = None
            self.has_gradient = False

        self.bounds_ = objective.bounds_

    def startPoint(self, method="uniform"):
        if method=="uniform":
            import numpy as np
            return  np.random.uniform(low=self.bounds_[:,0], high=self.bounds_[:,1])
        else:
            raise Exception("Unknown starpoint sampling method {}".format(method))


    def minimise(self, nrestart=1, method="tnc", tol=1e-6):
        from scipy import optimize
        minobj = np.Infinity
        finalres = None
        import time
        t0=time.time()
        for t in range(nrestart):
            x0 = self.startPoint()

            if   method=="tnc":    res = self.minimiseTNC(   x0, tol=tol)
            elif method=="lbfgsb": res = self.minimiseLBFGSB(x0, tol=tol)
            else: raise Exception("Unknown minimiser {}".format(method))

            if res["fun"] < minobj:
                minobj = res["fun"]
                finalres = res
        t1=time.time()
        print("Minimisation took {} seconds".format(t1-t0))
        return finalres



    def minimiseTNC(self, x0, tol=1e-6):
        from scipy import optimize
        res = optimize.minimize(
                lambda x: self.objective_(x),
                x0,
                bounds=self.bounds_,
                # jac=lambda x:self.gradient(x, sel=sel),
                method="TNC", tol=tol, options={'maxiter':1000, 'accuracy':tol})
        return res

    def minimiseLBFGSB(self, x0, tol=1e-6):
        from scipy import optimize
        res = optimize.minimize(
                lambda x: self.objective_(x),
                x0,
                bounds=self.bounds_,
                # jac=lambda x:self.gradient(x, sel=sel),
                method="L-BFGS-B", tol=tol)
        return res
