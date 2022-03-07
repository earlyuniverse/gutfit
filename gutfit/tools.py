def readParameterConfig(cfile):
    """
    Read in a parameter config file.
    Skip empty lines and those starting with '#'.
    Return dictionary --- key:list
    """
    dd = {}
    import os
    if not os.path.exists(cfile):
        raise Exception("Config file {} does not exist".format(cfile))
    with open(cfile) as f:
        for line in f:
            l=line.strip()
            if l.startswith("#"): continue
            if len(l)==0: continue
            temp = l.split()
            dd[temp[0]] = [float(v) for v in temp[1:]]
    return dd
