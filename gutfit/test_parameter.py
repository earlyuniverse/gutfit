import unittest
from gutfit import parameter

class TestParameter(unittest.TestCase):

    def test_range(self):
        P = parameter.Parameter(0, 1, name="ParA")
        print(P)
        print(P.sample())
        self.assertEqual(P.min, 0)

    def test_meas(self):
        P = parameter.Parameter(1, 0.1, 0.09, 0.3, 0.25, name="ParB")
        print(P)
        print(P.sample())
        for i in range(10000):
            P.sample()
        self.assertEqual(P.mean, 1)

if __name__ == "__main__":
    unittest.main();
