from gutfit.experimentalneutrinomassmatrix import ExperimentalNeutrinoMassMatrix
from gutfit.type1and2seesaw import Type1And2SeeSaw
from gutfit import parameterlist

def sample(E,PL,m1):
    p=PL()
    p["data_neutrino_m1"] = m1
    return E(p)

#https://stackoverflow.com/questions/13377046/scipy-fill-a-histogram-reading-from-a-db-event-by-event-in-a-loop
import numpy as np
class Hist1D(object):

    def __init__(self, nbins, xlow, xhigh):
        self.nbins = nbins
        self.xlow  = xlow
        self.xhigh = xhigh
        self.hist, edges = np.histogram([], bins=nbins, range=(xlow, xhigh))
        self.bins = (edges[:-1] + edges[1:]) / 2.

    def fill(self, arr):
        hist, edges = np.histogram(arr, bins=self.nbins, range=(self.xlow, self.xhigh))
        self.hist += hist

    @property
    def data(self):
        return self.bins, self.hist


if __name__=="__main__":
    import sys
    E = ExperimentalNeutrinoMassMatrix()
    S = Type1And2SeeSaw()
    PL =  parameterlist.ParameterList.fromConfigFile(sys.argv[1])#"examples/param_card.dat")


    from matplotlib import pyplot as plt
    import matplotlib as mpl
    plt.style.use('ggplot')


    D = []
    T = []

    mass = float(sys.argv[3])
    for _ in range(int(sys.argv[2])):
        data = sample(E, PL, mass)
        theo = sample(S, PL, mass)
        D.append(data)
        T.append(theo)

    fig, axes = plt.subplots(figsize=(10, 10), sharex=False, sharey=False, ncols=3, nrows=3)
    for i in range(3):
        for j in range(3):
            if i<j:
                axes[i, j].axis('off')
            else:
                values = [d[i][j] for d in D]
                theos  = [t[i][j] for t in T]
                axes[i, j].hist(values, bins=30)
                axes[i, j].hist(theos, bins=30)
                axes[i, j].set_xscale("log")
                axes[i, j].set_xscale("log")



    plt.savefig("both_{}.pdf".format(sys.argv[3]))

    # from IPython import embed
    # embed()
