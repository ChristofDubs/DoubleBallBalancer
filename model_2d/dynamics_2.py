# Autogenerated using symbolic_dynamics_n.py; don't edit!
from dataclasses import dataclass
from enum import Enum
import numpy as np
from numpy import sin, cos


class StateIndex(Enum):
    ALPHA_1_IDX = 0,
    PHI_IDX = 1,
    PSI_0_IDX = 2,
    ALPHA_DOT_1_IDX = 3,
    PHI_DOT_IDX = 4,
    PSI_DOT_0_IDX = 5,


def computeOmegaDot(x, param, omega_cmd):
    phi = x[StateIndex.PHI_IDX]
    psi_0 = x[StateIndex.PSI_0_IDX]
    alpha_dot_1 = x[StateIndex.ALPHA_DOT_1_IDX]
    phi_dot = x[StateIndex.PHI_DOT_IDX]
    psi_dot_0 = x[StateIndex.PSI_DOT_0_IDX]
    x0 = param["r_1"]**2
    x1 = cos(phi)
    x2 = param["r_2"] * x1
    x3 = param["m_2"] * x2
    x4 = param["r_1"] * x3
    x5 = param["theta_0"] / param["r_0"]**2
    x6 = sin(phi)
    x7 = param["m_2"] * param["r_2"]**2
    x8 = -param["r_0"] - param["r_1"]
    x9 = sin(psi_0)
    x10 = x8 * x9
    x11 = param["r_2"] * x6
    x12 = param["m_2"] * x11
    x13 = cos(psi_0)
    x14 = x13 * x8 + x8
    x15 = x10 * x12 + x14 * x3
    x16 = param["r_1"] * x8
    x17 = param["r_0"] + param["r_1"]
    x18 = param["r_1"] * x14
    x19 = param["m_0"] * x16 + param["m_1"] * x18 + param["m_2"] * x18 - param["r_1"] * x17 * x5
    x20 = x8**2
    x21 = x9**2
    x22 = param["m_1"] * x17
    x23 = x14**2
    x24 = psi_dot_0**2
    x25 = param["m_1"] * x24
    x26 = phi_dot**2
    x27 = param["m_2"] * x24
    x28 = -x10 * x27 - x12 * x26
    x29 = param["g"] * param["m_2"] - x13 * x17 * x27 + x26 * x3
    A = np.zeros((3, 3))
    b = np.zeros((3, 1))
    A[0, 0] = param["m_0"] * x0 + param["m_1"] * x0 + param["m_2"] * x0 + param["theta_1"] + x0 * x5 + x4
    A[0, 1] = param["theta_2"] + x1**2 * x7 + x4 + x6**2 * x7
    A[0, 2] = x15 + x19
    A[1, 0] = -1
    A[1, 1] = 1
    A[1, 2] = 0
    A[2, 0] = x19
    A[2, 1] = x15
    A[2, 2] = param["m_0"] * x20 + param["m_1"] * x23 + param["m_2"] * \
        x20 * x21 + param["m_2"] * x23 + x17**2 * x5 - x21 * x22 * x8
    b[0, 0] = -param["r_1"] * x28 - x11 * x29 + x16 * x25 * x9 - x2 * x28
    b[1, 0] = (alpha_dot_1 + omega_cmd - phi_dot) / param["tau"]
    b[2, 0] = x10 * x14 * x25 - x10 * x29 - x10 * (param["g"] * param["m_1"] - x13 * x22 * x24) - x14 * x28
    return np.linalg.solve(A, b)


def computeContactForces(x, param, omega_cmd):
    omega_dot = computeOmegaDot(x, param, omega_cmd)
    alpha_ddot_1 = omega_dot[StateIndex.ALPHA_1_IDX]
    phi_ddot = omega_dot[StateIndex.PHI_IDX]
    psi_ddot_0 = omega_dot[StateIndex.PSI_0_IDX]
    phi = x[StateIndex.PHI_IDX]
    psi_0 = x[StateIndex.PSI_0_IDX]
    phi_dot = x[StateIndex.PHI_DOT_IDX]
    psi_dot_0 = x[StateIndex.PSI_DOT_0_IDX]
    x0 = alpha_ddot_1 * param["r_1"]
    x1 = -param["r_0"] - param["r_1"]
    x2 = psi_ddot_0 * x1
    x3 = cos(psi_0)
    x4 = psi_ddot_0 * (x1 * x3 + x1)
    x5 = sin(psi_0)
    x6 = param["m_1"] * x5
    x7 = psi_dot_0**2
    x8 = x1 * x7
    x9 = cos(phi)
    x10 = param["m_2"] * param["r_2"]
    x11 = phi_ddot * x10
    x12 = sin(phi)
    x13 = phi_dot**2 * x10
    x14 = param["m_2"] * x5
    x15 = param["m_2"] * x0 + param["m_2"] * x4 + x11 * x9 - x12 * x13 - x14 * x8
    x16 = param["m_1"] * x0 + param["m_1"] * x4 + x15 - x6 * x8
    x17 = param["r_0"] + param["r_1"]
    x18 = x17 * x3 * x7
    x19 = param["g"] * param["m_2"] - param["m_2"] * x18 + x11 * x12 + x13 * x9 + x14 * x2
    x20 = param["g"] * param["m_1"] - param["m_1"] * x18 - psi_ddot_0 * x17 * x6 + x19
    F_0 = np.zeros((3, 1))
    F_1 = np.zeros((3, 1))
    F_2 = np.zeros((3, 1))
    F_0[0, 0] = param["m_0"] * x0 + param["m_0"] * x2 + x16
    F_0[1, 0] = param["g"] * param["m_0"] + x20
    F_0[2, 0] = 0
    F_1[0, 0] = x16
    F_1[1, 0] = x20
    F_1[2, 0] = 0
    F_2[0, 0] = x15
    F_2[1, 0] = x19
    F_2[2, 0] = 0
    return [F_0, F_1, F_2]


def computePositions(x, param):
    alpha_1 = x[StateIndex.ALPHA_1_IDX]
    phi = x[StateIndex.PHI_IDX]
    psi_0 = x[StateIndex.PSI_0_IDX]
    x0 = alpha_1 * param["r_1"] - param["r_0"] * psi_0 - param["r_1"] * psi_0
    x1 = param["r_0"] + param["r_1"]
    x2 = x0 - x1 * sin(psi_0)
    x3 = param["r_0"] + x1 * cos(psi_0)
    r_OS_0 = np.zeros((3, 1))
    r_OS_1 = np.zeros((3, 1))
    r_OS_2 = np.zeros((3, 1))
    r_OS_0[0, 0] = x0
    r_OS_0[1, 0] = param["r_0"]
    r_OS_0[2, 0] = 0
    r_OS_1[0, 0] = x2
    r_OS_1[1, 0] = x3
    r_OS_1[2, 0] = 0
    r_OS_2[0, 0] = param["r_2"] * sin(phi) + x2
    r_OS_2[1, 0] = -param["r_2"] * cos(phi) + x3
    r_OS_2[2, 0] = 0
    return [r_OS_0, r_OS_1, r_OS_2]
