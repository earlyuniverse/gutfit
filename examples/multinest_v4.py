from gutfit.experimentalneutrinomassmatrix import ExperimentalNeutrinoMassMatrix
from gutfit.type1and2seesaw_v4 import Type1And2SeeSaw_v4
from gutfit import parameterlist

def plotting(D, T, measure, fout):
    from matplotlib import pyplot as plt
    import matplotlib as mpl
    plt.style.use('ggplot')

    plt.clf()


    fig, axes = plt.subplots(figsize=(10, 10), sharex=False, sharey=False, ncols=3, nrows=3)
    plt.title("Measure: {}".format(measure))
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
                axes[i, j].set_yscale("log")
#                from IPython import embed
#                embed()
#                exit(0)
#                axes[i, j].set_xscale("log")


    plt.savefig(fout)


def readparameter(fname):
    d = {}
    with open(fname) as f:
        for line in f:
            l = line.strip()
            if len(l) == 0 or l.startswith("#"):
                continue
                print(l)

            k,v  = l.split(" ")
            d[k] = float(v)
    return d
    

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
    trials=0
    for _ in range(N):
        p=PL()
#        from IPython import embed
#        embed()
#        exit(0)
        for num, pn in enumerate(pnames):
            p[pn] = x[num]
        data = E(p)
        
#        foundgoodpoint=False
#        while not foundgoodpoint:
#            if trials>1000000:
#                return D,T,False
#            p=PL()
##            print(pnames, len(pnames))
##            from IPython import embed
##            embed()
##            exit(1)
#            for num, pn in enumerate(pnames):
#                p[pn] = x[num]
#            if S.constraint(p):
#                foundgoodpoint = True
#            trials+=1
        theo = S(p)
        D.append(data)
        T.append(theo)
    return D, T, True


def measure(D, T):
    OL = [
            overlap([d[0][0] for d in D], [t[0][0] for t in T]),
            overlap([d[0][1] for d in D], [t[0][1] for t in T]),
            overlap([d[0][2] for d in D], [t[0][2] for t in T]),
            overlap([d[1][2] for d in D], [t[1][2] for t in T]),
            overlap([d[1][1] for d in D], [t[1][1] for t in T]),
            overlap([d[2][2] for d in D], [t[2][2] for t in T])
            ]

    DS = [
            abs(np.mean([d[0][0] for d in D]) - np.mean([t[0][0] for t in T])),
            abs(np.mean([d[0][1] for d in D]) - np.mean([t[0][1] for t in T])),
            abs(np.mean([d[0][2] for d in D]) - np.mean([t[0][2] for t in T])),
            abs(np.mean([d[1][2] for d in D]) - np.mean([t[1][2] for t in T])),
            abs(np.mean([d[1][1] for d in D]) - np.mean([t[1][1] for t in T])),
            abs(np.mean([d[2][2] for d in D]) - np.mean([t[2][2] for t in T]))
            ]

    from functools import reduce
    return sum([d*d for d,o in zip(DS,OL)]) # works technically

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


    try:
        os.makedirs(opts.OUTPUT)
    except:
        pass




    import sys
    E  = ExperimentalNeutrinoMassMatrix()
    S  = Type1And2SeeSaw_v4()
    PL = parameterlist.ParameterList.fromConfigFile(args[0])#"examples/param_card.dat")
    N  = 1000 # number of samples of PL
    
    if len(args) > 1:
        pdict        =   readparameter(args[1])
        pnames       = [ k for k,v in pdict.items() ]
        D, T, DUMMY  = sample([pdict[pn] for pn in pnames] ,E,S,PL,10*N, pnames)
        valuemeasure = measure(D,T)
        plotting(D, T, valuemeasure, args[2])
    else:
        
    
        

        usethese = []

        bounds, pnames = PL.getBox(usethese)

        PMIN   = [b[0] for b in bounds]
        PMAX   = [b[1] for b in bounds]
        PLEN   = [PMAX[i] - PMIN[i] for i in range(len(pnames))]

        def scaleParam(p, idx):
            return PMIN[idx] + p * PLEN[idx]
    # here the constraint should be applied, i.e. rather than sampling a box in the theory space, a "line/hypersurface" which satisfies the constraint should be sampled from
        def myprior(cube, ndim, nparams):
            for i in range(ndim):
                cube[i] = scaleParam(cube[i], i)

        def loglike(cube, ndim, nparams):
            PP=[cube[j] for j in range(ndim)]
            
            DATA, THEORY, goodpoint = sample(PP, E, S, PL, N, pnames)
            if not goodpoint:
                print("give up")
                return -1e101
            val = measure(DATA, THEORY)
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

        import json
        json.dump(pnames, open('%s/GUTFITparams.json'%opts.OUTPUT, 'w'))
        json.dump(pnames, open('%s/params.json'%opts.OUTPUT, 'w'))

        NP = len(pnames)
        print("Now analyzing output from {}/GUTFIT.txt".format(opts.OUTPUT))
        a = pymultinest.Analyzer(n_params = NP, outputfiles_basename='%s/GUTFIT'%opts.OUTPUT)
        a.get_data()
        try:
            s = a.get_stats()
        except:
            print("There was an error accumulating statistics. Try increasing the number of iterations, e.g. --mn-iterations -1")
            sys.exit(1)

        from collections import OrderedDict
        resraw = a.get_best_fit()["parameters"]
        D, T, DUMMY  = sample(resraw ,E,S,PL,10*N, pnames)
        bestval=measure(D,T)
        PP=OrderedDict.fromkeys(pnames)
        for num, pname in enumerate(pnames): PP[pname] = resraw[num]
        out="# Best fit point (measure: {}):\n".format(bestval)
        for k in PP: out+= "%s %.16f\n"%(k,PP[k])
        with open("%sconfig.best"%a.outputfiles_basename, "w") as f: f.write(out)
        print(out)

        print("Measure at best fit point is {}".format(bestval))
        
        plotting(D, T, bestval, fout)



