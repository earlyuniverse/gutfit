from gutfit import parameter

class ParameterList(object):
    def __init__(self, params):
        for p in params:
            assert(isinstance(p, parameter.Parameter))
        self.params_ = params
        self.pnames_ = self.getParameterNames()

    def sample(self):
        return {p.name:p() for p in self.params_}

    def getParameterNames(self):
        return [p.name for p in self.params_]

    def __call__(self):
        return self.sample()

    def getBox(self, params):
        """
        Get a box of parameters.
        """
        box = []

        if len(params) == 0:
            params = [p.name for p in self.params_ if p.isfree]

        # Make sure we don't specify the same parameter twice or so
        for p in params: assert(params.count(p)==1)

        for p in params:
            if not p in self.pnames_:
                raise Exception("Parameter {} is unknown".format(p))
            ip = self.pnames_.index(p)
            if not self.params_[ip].isfree:
                raise Exception("Parameter {} is not free".format(p))
            box.append( (self.params_[ip].min, self.params_[ip].max) )
        return box, params

    @classmethod
    def fromConfigFile(cls, cfile):
        from gutfit.tools import readParameterConfig
        cfg = readParameterConfig(cfile)

        return cls.fromDict(cfg)

    @classmethod
    def fromDict(cls, cfgdict):
        params = []
        for k, v in cfgdict.items():
            try:
                params.append(parameter.Parameter(*list(v), name=k))
            except:
                params.append(parameter.Parameter(v, name=k))
        return cls(params)
