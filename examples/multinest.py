from gutfit.experimentalneutrinomassmatrix import ExperimentalNeutrinoMassMatrix
from gutfit.type1and2seesaw import Type1And2SeeSaw
from gutfit import parameterlist

def overlap(A,B):
    A.sort()
    B.sort()

    minA = min(A)
    minB = min(B)
    maxA = max(A)
    maxB = max(B)

    Ni = 0
    if   maxA < minB: return 0
    elif maxB < minA: return 0
    elif minA < minB and maxA > maxB: return 1
    elif minA > minB and maxA < maxB: return 1


    elif minA < minB and maxA < maxB:
        i,j = 0,0
        while(A[i]<minB): i+=1
        while(B[j]<maxA): j+=1
        return (len(A)-i + j)/float((len(A)+len(B)))

    elif minA > minB and maxA > maxB:
        i,j = 0,0
        while(A[i]<maxB): i+=1
        while(B[j]<minA): j+=1
        return (len(A) + i - j)/float((len(A)+len(B)))

    else:
        print("whoopsiedoodles")


def sample(x,E,S,PL,N,pnames):
    D = []
    T = []
    for _ in range(N):
        p=PL()
        for num, pn in enumerate(pnames):
            p[pn] = x[num]
        data = E(p)
        p=PL()
        for num, pn in enumerate(pnames):
            p[pn] = x[num]
        theo = S(p)
        D.append(data)
        T.append(theo)

    OL = [
            overlap([d[0][0] for d in D], [t[0][0] for t in T]),
            overlap([d[0][1] for d in D], [t[0][1] for t in T]),
            overlap([d[0][2] for d in D], [t[0][2] for t in T]),
            overlap([d[1][2] for d in D], [t[1][2] for t in T]),
            overlap([d[1][1] for d in D], [t[1][1] for t in T]),
            # overlap([d[2][2] for d in D], [t[2][2] for t in T])
            ]

    DS = [
            abs(np.mean([d[0][0] for d in D]) - np.mean([t[0][0] for t in T])),
            abs(np.mean([d[0][1] for d in D]) - np.mean([t[0][1] for t in T])),
            abs(np.mean([d[0][2] for d in D]) - np.mean([t[0][2] for t in T])),
            abs(np.mean([d[1][2] for d in D]) - np.mean([t[1][2] for t in T])),
            abs(np.mean([d[1][1] for d in D]) - np.mean([t[1][1] for t in T])),
            # abs(np.mean([d[2][2] for d in D]) - np.mean([t[2][2] for t in T]))
            ]

    from functools import reduce
    # return reduce(lambda x,y :x*y,[ np.log(1+o) for d,o in zip(DS,OL)]) # Not finding viable points
    # return reduce(lambda x,y :x*y,[ np.exp(-d) for d,o in zip(DS,OL)])
    # return np.prod([ np.log(1+o) for d, o in in zip(DS,OL)])
    # return sum([d   for d,o in zip(DS,OL)]) # works technically
    return sum([d*d for d,o in zip(DS,OL)]) # works technically
    # return max([d[0][0] for d in D])-min([t[0][0] for t in T])

#https://stackoverflow.com/questions/13377046/scipy-fill-a-histogram-reading-from-a-db-event-by-event-in-a-loop
import numpy as np

if __name__=="__main__":

    import optparse, os, sys
    op = optparse.OptionParser(usage=__doc__)
    op.add_option("-o", "--output",    dest="OUTPUT",      default="nestout", type=str, help="Prefix for outputs (default: %default)")
    op.add_option("-v", "--debug",     dest="DEBUG",       default=False, action="store_true", help="Turn on some debug messages")
    op.add_option("-q", "--quiet",     dest="QUIET",       default=False, action="store_true", help="Turn off messages")
    op.add_option("--mn-seed",         dest="SEED",        default=-1, type=int,              help="Multinest seed (default: %default)")
    op.add_option("--mn-resume",       dest="RESUME",      default=False, action='store_true', help="Resume on previous run.")
    op.add_option("--mn-multi-mod",    dest="MULTIMODE",   default=False, action='store_true', help="Set multimodal to true.")
    op.add_option("--mn-update",       dest="UPDATE",      default=1000, type=int, help="Update inteval (default: %default iterations)")
    op.add_option("--mn-tol",          dest="TOL",         default=0.5, type=float, help="Evidence tolerance (default: %default)")
    op.add_option("--mn-eff",          dest="EFF",         default=0.8, type=float, help="Sampling efficiency (default: %default)")
    op.add_option("--mn-points",       dest="POINTS",      default=40, type=int,              help="Number of live points in PyMultinest (default: %default)")
    op.add_option("--mn-imax",         dest="ITMAX",       default=0, type=int, help="Max number of iterations PyMultinest, 0 is infinite (default: %default)")
    op.add_option("--mn-multimodal",   dest="MULTIMODAL",  default=False, action='store_true', help="Run in multimodal mode.")
    op.add_option("--mn-no-importance",dest="NOIMPSAMP",   default=False, action='store_true', help="Turn off importance sampling.")
    opts, args = op.parse_args()

    # print(overlap(list(np.linspace(0,1,100)), list(np.linspace(0.1,1.1,100)) ))
    # print(overlap(list(np.linspace(0,1,100)), list(np.linspace(3,4,100)) ))
    # print(overlap(list(np.linspace(3,4,100)), list(np.linspace(1,0,100)) ))
    # print(overlap(list(np.linspace(1,4,100)), list(np.linspace(2,3,100)) ))
    # print(overlap(list(np.linspace(2,3,100)), list(np.linspace(1,4,100)) ))

    # print(overlap(list(np.linspace(0,4,100)), list(np.linspace(2,7,100)) ))

    # exit(1)
    try:
        os.makedirs(opts.OUTPUT)
    except:
        pass





    import sys
    E  = ExperimentalNeutrinoMassMatrix()
    # S  = ExperimentalNeutrinoMassMatrix()
    S  = Type1And2SeeSaw()
    PL =  parameterlist.ParameterList.fromConfigFile(sys.argv[1])#"examples/param_card.dat")
    N  = 1000 # number of samples of PL


    # usethese=["data_neutrino_alpha21","data_neutrino_alpha31"]
    usethese = []

    bounds, pnames = PL.getBox(usethese)
    # x0 = [PL()[p] for p in pnames]

    PMIN   = [b[0] for b in bounds]
    PMAX   = [b[1] for b in bounds]
    PLEN   = [PMAX[i] - PMIN[i] for i in range(len(pnames))]

    def scaleParam(p, idx):
        return PMIN[idx] + p * PLEN[idx]

    def myprior(cube, ndim, nparams):
        for i in range(ndim):
            cube[i] = scaleParam(cube[i], i)

    def loglike(cube, ndim, nparams):
        PP=[cube[j] for j in range(ndim)]

        val = sample(PP,E,S,PL,N,pnames)
        print(val)
        if val == 0:
            return -1e101


        loglikelihood = -val # Ad-hoc
        return loglikelihood


    import pymultinest

    pymultinest.run(loglike, myprior, len(pnames),
            importance_nested_sampling = not opts.NOIMPSAMP,
            verbose = False if opts.QUIET else True,
            multimodal=opts.MULTIMODAL,
            resume=opts.RESUME,
            n_iter_before_update=opts.UPDATE,
            evidence_tolerance=opts.TOL,
            sampling_efficiency = opts.EFF,
            init_MPI=False,
            n_live_points = opts.POINTS,
            max_iter=opts.ITMAX,
            seed=opts.SEED,
            outputfiles_basename='%s/GUTFIT'%opts.OUTPUT)

    # # run MultiNest
    # pymultinest.run(loglike, myprior, len(pnames), outputfiles_basename='out/',
        # resume = False, verbose = True, n_live_points=20,
            # evidence_tolerance=0.5,
            # sampling_efficiency = 0.8
        # )
    import json
    json.dump(pnames, open('%s/GUTFITparams.json'%opts.OUTPUT, 'w'))

    # NP = len(pnames)
    # print("Now analyzing output from {}/GUTFIT.txt".format(opts.OUTPUT))
    # a = pymultinest.Analyzer(n_params = NP, outputfiles_basename='%s/GUTFIT'%opts.OUTPUT)
    # a.get_data()
    # try:
        # s = a.get_stats()
    # except:
        # print("There was an error accumulating statistics. Try increasing the number of iterations, e.g. --mn-iterations -1")
        # sys.exit(1)

    # from collections import OrderedDict
    # resraw = a.get_best_fit()["parameters"]
    # PP=OrderedDict.fromkeys(pnames)
    # for num, pname in enumerate(pnames): PP[pname] = resraw[num]
    # out="# Best fit point:\n"
    # for k in PP: out+= "%s %.16f\n"%(k,PP[k])
    # with open("%sconfig.best"%a.outputfiles_basename, "w") as f: f.write(out)




    # from scipy import optimize
    # res = optimize.minimize(lambda x: sample(x,E,S,PL,N), x0,bounds=bounds,method="L-BFGS-B", tol=1e-4, options={"disp":True})
    # print(res)

    # from matplotlib import pyplot as plt
    # import matplotlib as mpl
    # plt.style.use('ggplot')


    # fig, axes = plt.subplots(figsize=(10, 10), sharex=False, sharey=False, ncols=3, nrows=3)
    # for i in range(3):
        # for j in range(3):
            # if i<j:
                # axes[i, j].axis('off')
            # else:
                # values = [d[i][j] for d in D]
                # theos  = [t[i][j] for t in T]
                # axes[i, j].hist(values, bins=30)
                # axes[i, j].hist(theos, bins=30)
                # axes[i, j].set_xscale("log")
                # axes[i, j].set_xscale("log")



    # plt.savefig("both_{}.pdf".format(sys.argv[3]))

