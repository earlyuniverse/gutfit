from gutfit import parameter

class ParameterList(object):
    def __init__(self, params):
        for p in params:
            assert(isinstance(p, parameter.Parameter))
        self.params_ = params

    def sample(self):
        return {p.name:p() for p in self.params_}

    def __call__(self):
        return self.sample()

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
