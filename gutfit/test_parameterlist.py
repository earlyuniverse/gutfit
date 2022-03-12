import unittest
from gutfit import parameterlist

class TestParameter(unittest.TestCase):
    def testFromDict(self):
        dd = {
                "a":1,
                "b":2,
                "c":[1,3],
                "d":[10, 0.1, 0.09, 0.3, 0.2]
                }
        pl = parameterlist.ParameterList.fromDict(dd)
        print(pl.sample())
        import time
        t0=time.time()
        for _ in range(1000):
            test = pl.sample()
            assert(test["d"] >= 9.8 and test["d"] <= 10.3)
        print(time.time() - t0)

    def testFromFile(self):
        pl = parameterlist.ParameterList.fromConfigFile("examples/param_card.dat")
        print(pl.sample())

    def testBox(self):
        pl = parameterlist.ParameterList.fromConfigFile("examples/param_card.dat")
        box = pl.getBox(["generic_quark_phase_a1", "generic_quark_phase_a2", "model1_mL"])
        print(pl.sample())

if __name__ == "__main__":
    unittest.main();
