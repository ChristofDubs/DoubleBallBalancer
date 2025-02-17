# Autogenerated using symbolic_dynamics_n.py; don't edit!
from enum import IntEnum

import numpy as np
from numpy import cos, sin

from .dynamic_model_n import NBallDynamicModel


class StateIndex(IntEnum):
    ALPHA_2_IDX = 0
    PHI_IDX = 1
    PSI_0_IDX = 2
    PSI_1_IDX = 3
    ALPHA_DOT_2_IDX = 4
    PHI_DOT_IDX = 5
    PSI_DOT_0_IDX = 6
    PSI_DOT_1_IDX = 7
    NUM_STATES = 8


class DynamicModel(NBallDynamicModel):
    def __init__(self, param, x0):
        super().__init__(StateIndex.NUM_STATES, param, x0)

    def computeOmegaDot(self, x, param, omega_cmd):
        phi = x[StateIndex.PHI_IDX]
        psi_0 = x[StateIndex.PSI_0_IDX]
        psi_1 = x[StateIndex.PSI_1_IDX]
        alpha_dot_2 = x[StateIndex.ALPHA_DOT_2_IDX]
        phi_dot = x[StateIndex.PHI_DOT_IDX]
        psi_dot_0 = x[StateIndex.PSI_DOT_0_IDX]
        psi_dot_1 = x[StateIndex.PSI_DOT_1_IDX]
        x0 = param["r_2"]**2
        x1 = cos(phi)
        x2 = param["r_3"] * x1
        x3 = param["m_3"] * x2
        x4 = -param["r_2"] * x3
        x5 = param["theta_0"] / param["r_0"]**2
        x6 = param["theta_1"] / param["r_1"]**2
        x7 = sin(phi)
        x8 = param["m_3"] * param["r_3"]**2
        x9 = -param["r_1"]
        x10 = -param["r_0"] + x9
        x11 = sin(psi_0)
        x12 = x10 * x11
        x13 = param["r_3"] * x7
        x14 = param["m_3"] * x13
        x15 = cos(psi_0)
        x16 = x10 * x15 + x10
        x17 = x12 * x14 + x16 * x3
        x18 = param["r_0"] + param["r_1"]
        x19 = param["r_2"] * x5
        x20 = param["m_0"] * param["r_2"]
        x21 = param["r_2"] * x16
        x22 = -param["m_1"] * x21 - param["m_2"] * x21 - param["m_3"] * x21 - x10 * x20 + x18 * x19
        x23 = -param["r_2"] + x9
        x24 = sin(psi_1)
        x25 = x23 * x24
        x26 = cos(psi_1)
        x27 = param["r_1"] + param["r_2"]
        x28 = x23 * x26 + x27
        x29 = x14 * x25 + x28 * x3
        x30 = param["m_1"] * x27
        x31 = param["r_2"] * x28
        x32 = -param["m_2"] * x31 - param["m_3"] * x31 - param["r_2"] * \
            x27 * x6 - param["r_2"] * x30 + x19 * x23 - x20 * x27
        x33 = x10**2
        x34 = x11**2
        x35 = x33 * x34
        x36 = param["m_1"] * x18
        x37 = x16**2
        x38 = x12 * x25
        x39 = x16 * x28
        x40 = param["m_0"] * x10 * x27 + param["m_2"] * x38 + param["m_2"] * x39 + \
            param["m_3"] * x38 + param["m_3"] * x39 + x16 * x30 + x18 * x23 * x5
        x41 = x27**2
        x42 = x23**2
        x43 = x24**2 * x42
        x44 = x28**2
        x45 = psi_dot_0**2
        x46 = x12 * x45
        x47 = param["m_1"] * x46
        x48 = psi_dot_1**2
        x49 = x25 * x48
        x50 = -param["m_2"] * x46 - param["m_2"] * x49
        x51 = phi_dot**2
        x52 = x15 * x45
        x53 = x18 * x52
        x54 = x26 * x27 * x48
        x55 = param["g"] * param["m_3"] - param["m_3"] * x53 - param["m_3"] * x54 + x3 * x51
        x56 = -param["m_3"] * x46 - param["m_3"] * x49 - x14 * x51
        x57 = param["g"] * param["m_2"] - param["m_2"] * x53 - param["m_2"] * x54
        A = np.zeros((4, 4))
        b = np.zeros((4, 1))
        A[0, 0] = param["m_0"] * x0 + param["m_1"] * x0 + param["m_2"] * \
            x0 + param["m_3"] * x0 + param["theta_2"] + x0 * x5 + x0 * x6 + x4
        A[0, 1] = param["theta_3"] + x1**2 * x8 + x4 + x7**2 * x8
        A[0, 2] = x17 + x22
        A[0, 3] = x29 + x32
        A[1, 0] = -1
        A[1, 1] = 1
        A[1, 2] = 0
        A[1, 3] = 0
        A[2, 0] = x22
        A[2, 1] = x17
        A[2, 2] = param["m_0"] * x33 + param["m_1"] * x37 + param["m_2"] * x35 + param["m_2"] * \
            x37 + param["m_3"] * x35 + param["m_3"] * x37 - x10 * x34 * x36 + x18**2 * x5
        A[2, 3] = x40
        A[3, 0] = x32
        A[3, 1] = x29
        A[3, 2] = x40
        A[3, 3] = param["m_0"] * x41 + param["m_1"] * x41 + param["m_2"] * x43 + \
            param["m_2"] * x44 + param["m_3"] * x43 + param["m_3"] * x44 + x41 * x6 + x42 * x5
        b[0, 0] = -param["r_2"] * x47 + param["r_2"] * x50 + param["r_2"] * x56 - x13 * x55 - x2 * x56
        b[1, 0] = (alpha_dot_2 + omega_cmd - phi_dot) / param["tau"]
        b[2, 0] = -x12 * x55 - x12 * x57 - x12 * \
            (param["g"] * param["m_1"] - x36 * x52) + x16 * x47 - x16 * x50 - x16 * x56
        b[3, 0] = -x25 * x55 - x25 * x57 - x28 * x50 - x28 * x56 + x30 * x46
        return np.linalg.solve(A, b)

    def computeContactForces(self, x, param, omega_cmd):
        omega_dot = self.computeOmegaDot(x, param, omega_cmd)
        alpha_ddot_2 = omega_dot[StateIndex.ALPHA_2_IDX]
        phi_ddot = omega_dot[StateIndex.PHI_IDX]
        psi_ddot_0 = omega_dot[StateIndex.PSI_0_IDX]
        psi_ddot_1 = omega_dot[StateIndex.PSI_1_IDX]
        phi = x[StateIndex.PHI_IDX]
        psi_0 = x[StateIndex.PSI_0_IDX]
        psi_1 = x[StateIndex.PSI_1_IDX]
        phi_dot = x[StateIndex.PHI_DOT_IDX]
        psi_dot_0 = x[StateIndex.PSI_DOT_0_IDX]
        psi_dot_1 = x[StateIndex.PSI_DOT_1_IDX]
        x0 = -param["r_1"]
        x1 = -param["r_0"] + x0
        x2 = psi_ddot_0 * x1
        x3 = param["r_1"] + param["r_2"]
        x4 = psi_ddot_1 * x3
        x5 = alpha_ddot_2 * param["r_2"]
        x6 = cos(psi_0)
        x7 = psi_ddot_0 * (x1 * x6 + x1)
        x8 = sin(psi_0)
        x9 = param["m_1"] * x8
        x10 = psi_dot_0**2
        x11 = x1 * x10
        x12 = cos(psi_1)
        x13 = -param["r_2"] + x0
        x14 = psi_ddot_1 * (x12 * x13 + x3)
        x15 = x11 * x8
        x16 = x13 * sin(psi_1)
        x17 = psi_dot_1**2
        x18 = param["m_2"] * x17
        x19 = cos(phi)
        x20 = param["m_3"] * param["r_3"]
        x21 = phi_ddot * x20
        x22 = sin(phi)
        x23 = phi_dot**2 * x20
        x24 = param["m_3"] * x16
        x25 = param["m_3"] * x14 - param["m_3"] * x15 - param["m_3"] * \
            x5 + param["m_3"] * x7 - x17 * x24 + x19 * x21 - x22 * x23
        x26 = param["m_2"] * x14 - param["m_2"] * x15 - param["m_2"] * x5 + param["m_2"] * x7 - x16 * x18 + x25
        x27 = param["m_1"] * x4 - param["m_1"] * x5 + param["m_1"] * x7 - x11 * x9 + x26
        x28 = param["r_0"] + param["r_1"]
        x29 = x10 * x28 * x6
        x30 = x2 * x8
        x31 = x12 * x3
        x32 = param["g"] * param["m_3"] - param["m_3"] * x17 * x31 - param["m_3"] * \
            x29 + param["m_3"] * x30 + psi_ddot_1 * x24 + x19 * x23 + x21 * x22
        x33 = param["g"] * param["m_2"] + param["m_2"] * psi_ddot_1 * \
            x16 - param["m_2"] * x29 + param["m_2"] * x30 - x18 * x31 + x32
        x34 = param["g"] * param["m_1"] - param["m_1"] * x29 - psi_ddot_0 * x28 * x9 + x33
        F_0 = np.zeros((3, 1))
        F_1 = np.zeros((3, 1))
        F_2 = np.zeros((3, 1))
        F_3 = np.zeros((3, 1))
        F_0[0, 0] = param["m_0"] * x2 + param["m_0"] * x4 - param["m_0"] * x5 + x27
        F_0[1, 0] = param["g"] * param["m_0"] + x34
        F_0[2, 0] = 0
        F_1[0, 0] = x27
        F_1[1, 0] = x34
        F_1[2, 0] = 0
        F_2[0, 0] = x26
        F_2[1, 0] = x33
        F_2[2, 0] = 0
        F_3[0, 0] = x25
        F_3[1, 0] = x32
        F_3[2, 0] = 0
        return [F_0, F_1, F_2, F_3]

    def computePositions(self, x, param):
        alpha_2 = x[StateIndex.ALPHA_2_IDX]
        phi = x[StateIndex.PHI_IDX]
        psi_0 = x[StateIndex.PSI_0_IDX]
        psi_1 = x[StateIndex.PSI_1_IDX]
        x0 = -alpha_2 * param["r_2"] - param["r_0"] * psi_0 - \
            param["r_1"] * psi_0 + param["r_1"] * psi_1 + param["r_2"] * psi_1
        x1 = param["r_0"] + param["r_1"]
        x2 = x0 - x1 * sin(psi_0)
        x3 = param["r_0"] + x1 * cos(psi_0)
        x4 = param["r_1"] + param["r_2"]
        x5 = x2 - x4 * sin(psi_1)
        x6 = x3 + x4 * cos(psi_1)
        r_OS_0 = np.zeros((3, 1))
        r_OS_1 = np.zeros((3, 1))
        r_OS_2 = np.zeros((3, 1))
        r_OS_3 = np.zeros((3, 1))
        r_OS_0[0, 0] = x0
        r_OS_0[1, 0] = param["r_0"]
        r_OS_0[2, 0] = 0
        r_OS_1[0, 0] = x2
        r_OS_1[1, 0] = x3
        r_OS_1[2, 0] = 0
        r_OS_2[0, 0] = x5
        r_OS_2[1, 0] = x6
        r_OS_2[2, 0] = 0
        r_OS_3[0, 0] = param["r_3"] * sin(phi) + x5
        r_OS_3[1, 0] = -param["r_3"] * cos(phi) + x6
        r_OS_3[2, 0] = 0
        return [r_OS_0, r_OS_1, r_OS_2, r_OS_3]

    def computeBallAngles(self, x, param):
        alpha_2 = x[StateIndex.ALPHA_2_IDX]
        psi_0 = x[StateIndex.PSI_0_IDX]
        psi_1 = x[StateIndex.PSI_1_IDX]
        x0 = alpha_2 * param["r_2"]
        x1 = param["r_1"] * psi_1
        x2 = param["r_2"] * psi_1
        alpha = np.zeros((1, 3))
        alpha[0, 0] = (param["r_0"] * psi_0 + param["r_1"] * psi_0 + x0 - x1 - x2) / param["r_0"]
        alpha[0, 1] = (-x0 + x1 + x2) / param["r_1"]
        alpha[0, 2] = alpha_2
        return [alpha]
