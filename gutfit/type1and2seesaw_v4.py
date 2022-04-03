import numpy as np
import random
import math
from random import randint, choice
from gutfit import model, parameterlist
# same as type1and2seesaw_v3.py but now ce is an additional parameter which must be constrained in the multibest file.

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

def matrix_phase2(a1, a2):
    return np.array([[np.exp(1j * a1),             0.0,             0.0],
                    [            0.0, np.exp(1j * a2),             0.0],
                     [            0.0,             0.0,             1.0]],
                        dtype=np.complex64)


class Type1And2SeeSaw_v4(model.Model):
    def __init__(self):
        params = [
           "generic_quark_phase_a1",
           "generic_quark_phase_a2",
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
           "data_lepton_ye",
           "data_lepton_ymu",
           "data_lepton_ytau",
           "model4_mR",
           "model4_r1",
           "model4_r2",
           "model4_cnu",
           "model4_ce"
           ]
        super().__init__(params)
        
        self.randomphase()

    @property
    def val(self):
        value = np.abs(
                self.MnuTheory(
                    self.generic_quark_phase_a1,
                    self.generic_quark_phase_a2,
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
                    self.data_lepton_ye,
                    self.data_lepton_ymu,
                    self.data_lepton_ytau,
                    self.model4_mR,
                    self.model4_r1,
                    self.model4_r2,
                    self.model4_cnu,
                    self.model4_ce
                    )
                )


        self.randomphase()
        return value
        
    def constraint(self, xdict):
        self.__dict__.update((k, v) for k, v in xdict.items() if k in self.pnames_)
        value = self.constraintchargedleptons(
                    self.generic_quark_phase_a1,
                    self.generic_quark_phase_a2,
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
                    self.data_lepton_ye,
                    self.data_lepton_ymu,
                    self.data_lepton_ytau,
                    self.model4_mR,
                    self.model4_r1,
                    self.model4_r2,
                    self.model4_cnu,
                    self.model4_ce
                    )

        return value

    def randomphase(self):
        self.ydrand  = (2. * random.randint(0, 1) -1.)
        self.ysrand  = (2. * random.randint(0, 1) -1.)
        self.ybrand  = (2. * random.randint(0, 1) -1.)
        self.yurand  = (2. * random.randint(0, 1) -1.)
        self.ycrand  = (2. * random.randint(0, 1) -1.)
        self.ytrand  = (2. * random.randint(0, 1) -1.)
   
    def matrix_Yd(self, a1, a2,  th12, th13, th23, delta, yd, ys, yb):
        Pa      = matrix_phase2(a1, a2)
        Vckm    = matrix_vckm(th12, th13, th23, delta)
        ydrand  = self.ydrand * yd
        ysrand  = self.ysrand * ys
        ybrand  = self.ybrand * yb
        Yddiag  = matrix_diag3(ydrand, ysrand, ybrand)
        Vckmc   = np.conj(Vckm)
        Yukd    = Pa @ Vckm @ Yddiag  @ np.transpose(Vckmc) @  np.conj(Pa)
        return  Yukd


    def MnuTheory(self, a1, a2,  th12q, th13q, th23q, deltaq, yu, yc, yt, yd, ys, yb, ye, ymu, ytau,  mR, r1, r2, cnu, ce):
       Yd        = self.matrix_Yd( a1, a2,  th12q, th13q, th23q, deltaq, yd, ys, yb)
       Yu        = matrix_diag3(self.yurand* yu, self.ycrand * yc, self.ytrand * yt)
       ReYd      = np.real(Yd)
       ImYd      = np.imag(Yd)
       cnulogged = 10**cnu
       r2logged  = 10**r2
       r1logged  = 10**r1
       ydrand    = self.ydrand * yd
       ysrand    = self.ysrand * ys
       ybrand    = self.ybrand * yb
       type1p1   = (8 * r2logged * (r2logged+1) * Yu)/(r2logged-1) 
       type1p2   = -(16 * r2logged*r2logged * ReYd)/(r1logged * (r2logged-1))
       type1p3   = ((r2logged-1)/r1logged) * (r1logged * Yu + 1j * cnulogged * ImYd) @ np.linalg.inv(r1logged * Yu - ReYd) @ (r1logged * Yu - 1j * cnulogged * ImYd)
       type1     = 10**mR * (type1p1 + type1p2 + type1p3)
       return  type1
       
    def constraintchargedleptons(self, a1, a2,  th12q, th13q, th23q, deltaq, yu, yc, yt, yd, ys, yb, ye, ymu, ytau,  mR, r1, r2, cnu, ce):
        Yd        = self.matrix_Yd(a1, a2,  th12q, th13q, th23q, deltaq, yd, ys, yb)
        Yu        = matrix_diag3(self.yurand* yu, self.ycrand * yc, self.ytrand * yt)
        ReYd      = np.real(Yd)
        ImYd      = np.imag(Yd)
        r2logged  = 10**r2
        r1logged  = 10**r1
        celogged  = 10**ce
        p1        = -4. * r1logged/(r2logged - 1.) * Yu
        p2        = (r2logged + 3.)/(r2logged - 1.) * ReYd
        p3        = 1j * celogged * ImYd
        Ye        = p1 + p2 + p3
        Yehc      = np.transpose(np.conj(Ye))
        constsq   = ye * ye + ymu * ymu + ytau * ytau
        constdet  = ye * ye * ymu * ymu * ytau * ytau
        detYeYe   = np.linalg.det(Ye @ Yehc)
        Tryeye    = (Ye @ Yehc).trace()
        answer    = False
#        from IPython import embed
#        embed()
#        exit(1)
        if math.isclose(Tryeye.real, constsq, rel_tol=1e-5) and math.isclose(detYeYe.real , constdet, rel_tol=1e-5):
           answer        = True
        return answer
      
        
if __name__=="__main__":
    E =  Type1And2SeeSaw_v4()
    PL =  parameterlist.ParameterList.fromConfigFile("examples/param_card.dat")
    from IPython import embed
    embed()
    E(PL())
    import time
    t0 = time.time()
    for _ in range(1000000):
        E(PL())

    print(time.time() - t0)
