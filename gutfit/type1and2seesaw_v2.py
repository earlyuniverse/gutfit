import numpy as np
import random
from random import randint, choice
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

def matrix_Yd(a1, a2, a3,  th12, th13, th23, delta, yd, ys, yb):
    Pa      = matrix_phase(a1, a2, a3)
    Vckm    = matrix_vckm(th12, th13, th23, delta)
#   Since Yd can be hermitian diag(yd, ys, yb) can have an arbitrary sign(\pm 1), the random number generator implements this below
    ydrand  = (2. * random.randint(0, 1) -1.) * yd
    ysrand  = (2. * random.randint(0, 1) -1.) * ys
    ybrand  = (2. * random.randint(0, 1) -1.) * yb
    Yddiag  = matrix_diag3(ydrand, ysrand, ybrand)
    Vckmc   = np.conj(Vckm)
    Yukd    = Pa @ Vckm @ Yddiag  @ np.transpose(Vckmc) @  np.conj(Pa)
    return Yukd

class Type1And2SeeSaw_v2(model.Model):
    def __init__(self):
        params = [
           "generic_quark_phase_a1",
           "generic_quark_phase_a2",
           "generic_quark_phase_a3",
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
           "model2_mR",
           "model2_r1",
           "model2_r2",
           "model2_cnu"
           ]
        super().__init__(params)

    @property
    def val(self):
        return np.abs(
                self.MnuTheory(
                    self.generic_quark_phase_a1,
                    self.generic_quark_phase_a2,
                    self.generic_quark_phase_a3,
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
                    self.model2_mR,
                    self.model2_r1,
                    self.model2_r2,
                    self.model2_cnu
                    )
                )

    def MnuTheory(self, a1, a2, a3, th12q, th13q, th23q, deltaq, yu, yc, yt, yd, ys, yb,  mR, r1, r2, cnu):
       Yd      = matrix_Yd(a1, a2, a3, th12q, th13q, th23q, deltaq, yd, ys, yb)
       Yu      = matrix_diag3(yu, yc, yt)
       ReYd    = np.real(Yd)
       ImYd    = np.imag(Yd)
       type1p1 = (8 * r2 * (r2+1) * Yu)/(r2-1) 
       type1p2 = (16 * r2*r2 * ReYd)/(r1 * (r2-1))
       type1p3 = ((r2-1)/r1) * (r1 * Yu + 1j * cnu * ImYd) @ np.linalg.inv(r1 * Yu - ReYd) @ (r1 * Yu - 1j * cnu * ImYd)
       type1   = mR * (type1p1 + type1p2 + type1p3)
       return  type1 
        
if __name__=="__main__":
    E =  Type1And2SeeSaw_v2()
    PL =  parameterlist.ParameterList.fromConfigFile("examples/param_card.dat")
    from IPython import embed
    embed()
    E(PL())
    import time
    t0 = time.time()
    for _ in range(1000000):
        E(PL())

    print(time.time() - t0)
