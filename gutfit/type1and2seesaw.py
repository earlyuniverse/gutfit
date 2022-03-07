import numpy as np
from gutfit import model, parameterlist

def matrix_diag3(d1,d2,d3):
    return np.array([[d1, 0.0, 0.0], [0.0, d2, 0.0], [0.0, 0.0, d3]])

# Generic Rotations #
def matrix_rot23(th23):
    return np.array([[1.0,          0.0 , 0.0],
                    [0.0,  np.cos(th23), np.sin(th23)],
                    [0.0, -np.sin(th23), np.cos(th23)]])

def matrix_rot12(th12):
    return np.array([[ np.cos(th12), np.sin(th12), 0.0],
                    [-np.sin(th12), np.cos(th12), 0.0],
                    [          0.0,  0.0,         1.0]])

def matrix_rot13(th13, delta):
    return np.array([[                     np.cos(th13), 0.0, np.sin(th13) * np.exp(-1j * delta)],
                    [                     0.0         , 1.0, 0.0                               ],
                    [-np.sin(th13)* np.exp(1j * delta), 0.0, np.cos(th13)]],
                    dtype=np.complex64)

def matrix_vckm(th12, th13, th23, delta):
    return matrix_rot23(th23) @ matrix_rot13(th13, delta) @ matrix_rot12(th12)

# Phase Matrices #

def matrix_phase(a1, a2, a3):
    return np.array([[np.exp(1j * a1),             0.0,             0.0],
                    [            0.0, np.exp(1j * a2),             0.0],
                    [            0.0,             0.0, np.exp(1j * a3)]],
                    dtype=np.complex64)


def matrix_Yd(a1, a2, a3, b1, b2, th12, th13, th23, delta, yd, ys, yb):
    Pa      = matrix_phase(a1, a2, a3)
    Pb      = matrix_phase(b1, b2, 0.0)
    Vckm    = matrix_vckm(th12, th13, th23, delta)
    Yddiag  = matrix_diag3(yd, ys, yb)
    Yukd    = Pa @ Vckm @ Yddiag @ Pb @ np.transpose(Vckm) @  Pa
    return Yukd

class Type1And2SeeSaw(model.Model):
    def __init__(self):
        params = [
           "generic_quark_phase_a1",
           "generic_quark_phase_a2",
           "generic_quark_phase_a3",
           "generic_quark_phase_b1",
           "generic_quark_phase_b2",
           "data_quark_th12",
           "data_quark_th13",
           "data_quark_th23",
           "data_quark_delta",
           "data_quark_yu",
           "data_quark_yc",
           "data_quark_yt",
           "data_quark_yd",
           "data_quark_ys",
           "data_quark_yb",
           "model1_mL",
           "model1_mR",
           "model1_r1",
           "model1_Rer2",
           "model1_Imr2"
           ]
        super().__init__(params)

    @property
    def val(self):
        return np.abs(
                self.MnuTheory(
                    self.generic_quark_phase_a1,
                    self.generic_quark_phase_a2,
                    self.generic_quark_phase_a3,
                    self.generic_quark_phase_b1,
                    self.generic_quark_phase_b2,
                    self.data_quark_th12,
                    self.data_quark_th13,
                    self.data_quark_th23,
                    self.data_quark_delta,
                    self.data_quark_yu,
                    self.data_quark_yc,
                    self.data_quark_yt,
                    self.data_quark_yd,
                    self.data_quark_ys,
                    self.data_quark_yb,
                    self.model1_mL,
                    self.model1_mR,
                    self.model1_r1,
                    self.model1_Rer2,
                    self.model1_Imr2
                    )
                )

    def MnuTheory(self, a1, a2, a3, b1, b2, th12q, th13q, th23q, deltaq, yu, yc, yt, yd, ys, yb, mL, mR, r1, Rer2, Imr2):
        Yd = matrix_Yd(a1, a2, a3, b1, b2, th12q, th13q, th23q, deltaq, yd, ys, yb)
        Yu      = matrix_diag3(yu, yc, yt)
        r2      = Rer2 + 1j * Imr2
        type1p1 = 8.0 * (r2 - 3.0)/(r2-1.0) * Yu
        type1p2 = -16.0 /(r1 * (r2 - 1.0)) * Yd
        type1p3 = (r1 * (r2 - 1.0))/r2 * Yu @ np.linalg.inv(r1 * Yu - Yd) @ Yu
        type1   =  mR * (type1p1 + type1p2 + type1p3)
        type2p1 =  Yu / (r2 - 1)
        type2p2 = -Yd / (r1 * (r2 - 1))
        type2   = mL * (type2p1 + type2p2)
        return type1 + type2


if __name__=="__main__":
    E =  Type1And2SeeSaw()
    PL =  parameterlist.ParameterList.fromConfigFile("examples/param_card.dat")
    from IPython import embed
    embed()
    E(PL())
    import time
    t0 = time.time()
    for _ in range(1000000):
        E(PL())

    print(time.time() - t0)
