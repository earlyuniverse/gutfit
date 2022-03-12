import numpy as np
from gutfit import model, parameterlist

def matrix_diag3(d1,d2,d3):
    return np.array([[d1, 0.0, 0.0], [0.0, d2, 0.0], [0.0, 0.0, d3]])

# Generic Rotations #
def matrix_rot23(th23):
    s23, c23 = np.sin(th23), np.cos(th23)
    return np.array([[1,    0,   0],
                    [ 0,  c23, s23],
                    [ 0, -s23, c23]])

def matrix_rot12(th12):
    s12, c12 = np.sin(th12), np.cos(th12)
    return np.array([[ c12, s12, 0],
                    [ -s12, c12, 0],
                    [    0,   0, 1]])

def matrix_rot13(th13, delta):
    return np.array([[                     np.cos(th13), 0.0, np.sin(th13) * np.exp(-1j * delta)],
                    [                     0.0         , 1.0, 0.0                               ],
                    [-np.sin(th13)* np.exp(1j * delta), 0.0, np.cos(th13)]],
                    dtype=np.complex64)

def matrix_vckm(th12, th13, th23, delta):
    return matrix_rot23(th23) @ matrix_rot13(th13, delta) @ matrix_rot12(th12)


class ExperimentalNeutrinoMassMatrix(model.Model):
    def __init__(self):
        params = [
                "data_neutrino_deltamsq21bf",
                "data_neutrino_deltamsq31bf",
                "data_neutrino_alpha21",
                "data_neutrino_alpha31",
                "data_neutrino_th23",
                "data_neutrino_th12",
                "data_neutrino_th13",
                "data_neutrino_delta",
                "data_neutrino_m1"
                ]
        super().__init__(params)

    @property
    def val(self):
        return np.abs(
                self.MnuData(
                    self.data_neutrino_deltamsq21bf,
                    self.data_neutrino_deltamsq31bf,
                    self. data_neutrino_alpha21,
                    self.data_neutrino_alpha31,
                    self.data_neutrino_th23,
                    self.data_neutrino_th12,
                    self.data_neutrino_th13,
                    self.data_neutrino_delta,
                    self.data_neutrino_m1)
                )


#    def MnuData(self, deltamsq21bf,deltamsq31bf, alpha21, alpha31, th23l, th12l, th13l, deltal, m1):
#        mnudiag  = matrix_diag3(m1, np.sqrt(m1 * m1 + deltamsq21bf * deltamsq21bf), np.sqrt(m1 * m1 + deltamsq31bf * deltamsq31bf))
#        Majorana = matrix_diag3(alpha21, alpha31, 0.0)
#        angle    = matrix_vckm(th12l, th13l, th23l, deltal)
#        Vpmns    = angle @ Majorana
#        return np.conj(Vpmns) @ mnudiag @ np.transpose(Vpmns)
    def MnuData(self, deltamsq21bf,deltamsq31bf, alpha21, alpha31, th23l, th12l, th13l, deltal, m1):
        mnudiag  = matrix_diag3(1.0, np.sqrt(1.0 + deltamsq21bf/(m1* m1)), np.sqrt(1.0 + deltamsq31bf/(m1* m1))
        Majorana = matrix_diag3(alpha21, alpha31, 0.0)
        angle    = matrix_vckm(th12l, th13l, th23l, deltal)
        Vpmns    = angle @ Majorana
        return np.conj(Vpmns) @ mnudiag @ np.transpose(Vpmns)

if __name__=="__main__":
    E = ExperimentalNeutrinoMassMatrix()
    PL =  parameterlist.ParameterList.fromConfigFile("examples/param_card.dat")
    E(PL())
    import time
    t0 = time.time()
    for _ in range(1000000):
        E(PL())

    print(time.time() - t0)
