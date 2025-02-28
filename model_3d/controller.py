"""Controller class for controlling 3D Double Ball Balancer
"""

import context  # noqa: F401
import numpy as np
from numpy import cos, sin
from pyrotation import Quaternion

from model_2d.controller_2 import Controller as Controller2D
from model_2d.dynamics_2 import StateIndex
from model_2d.param import getDefaultParam

ANGLE_MODE = StateIndex.ALPHA_1_IDX
VELOCITY_MODE = StateIndex.ALPHA_DOT_1_IDX


def projectModelState(state):
    [phi_x, phi_y] = state.phi
    [phi_x_dot, phi_y_dot] = state.phi_dot
    [psi_x, psi_y] = state.psi
    [psi_x_dot, psi_y_dot] = state.psi_dot
    [w_2x, w_2y, w_2z] = state.omega_2

    R_IB2 = Quaternion(state.q2).rotation_matrix()

    # construct horizontal ball frame
    z = np.array([0, 0, 1])
    x = np.cross(R_IB2[:, 1], z)
    x *= 1 / np.linalg.norm(x)
    y = np.cross(z, x)

    R_IB2h = np.column_stack([x, y, z])

    # express psi vector in B2h frame
    I_e_S1S2 = np.array([cos(psi_x) * sin(psi_y), -sin(psi_x), cos(psi_x) * cos(psi_y)])
    B2h_e_S1S2 = np.dot(R_IB2h.T, I_e_S1S2)

    # extract psi angles wrt B2h frame
    B2h_psi_x = np.arcsin(-B2h_e_S1S2[1])
    B2h_psi_y = np.arcsin(B2h_e_S1S2[0])

    # express upper ball velocity in B2h frame
    I_omega_B2 = np.dot(R_IB2, state.omega_2)
    B2h_omega_IB2 = np.dot(R_IB2h.T, I_omega_B2)

    # extract upper ball velocity wrt B2h frame
    B2h_omega_2x = B2h_omega_IB2[0]
    B2h_omega_2y = B2h_omega_IB2[1]

    # express angular velocities induced by psi rates in B2h frame
    I_omega_psi = np.array([psi_x_dot * cos(psi_y), psi_y_dot, -psi_x_dot * sin(psi_y)])

    I_ez_B2 = np.cross(R_IB2[:, 1], x)
    yaw_B2h_rate = np.dot(I_omega_B2, I_ez_B2) / np.dot(y, R_IB2[:, 1])
    # yaw_B2h_rate = np.dot(R_IB2, np.array([state.omega_2[0], 0, state.omega_2[2]]))[2] / np.dot(y, R_IB2[:, 1])

    if yaw_B2h_rate != 0:
        I_omega_IB2h = np.array([0, 0, yaw_B2h_rate])
        # compensate "velocity" of I_e_S1S2 resulting from rotating B2h frame
        I_ve_S1S2 = np.cross(I_omega_IB2h, I_e_S1S2)

        # convert velocity back to x/y angular velocity
        I_omega_v_S1S2 = np.cross(I_ve_S1S2, np.array([0, 0, 1]))

        I_omega_psi -= I_omega_v_S1S2

    B2h_omega_psi = np.dot(R_IB2h.T, I_omega_psi)

    # extract psi rates wrt B2h frame
    B2h_psi_x_dot = B2h_omega_psi[0]
    B2h_psi_y_dot = B2h_omega_psi[1]

    # express lever arm directional vector in B2h frame
    R_IB3 = state.q3.rotation_matrix()
    I_e_S2S3 = np.dot(R_IB3, np.array([0, 0, -1]))
    B2h_e_S2S3 = np.dot(R_IB2h.T, I_e_S2S3)

    # extract lever arm angles wrt B2h frame
    B2h_phi_x = np.arcsin(B2h_e_S2S3[1])
    B2h_phi_y = np.arcsin(-B2h_e_S2S3[0])

    # express lever arm angular velocity in B2h frame
    B3_omega_IB3 = np.array(
        [
            phi_x_dot + w_2x * cos(phi_y) - w_2z * sin(phi_y),
            phi_y_dot * cos(phi_x)
            + w_2x * sin(phi_x) * sin(phi_y)
            + w_2y * cos(phi_x)
            + w_2z * sin(phi_x) * cos(phi_y),
            -phi_y_dot * sin(phi_x)
            + w_2x * sin(phi_y) * cos(phi_x)
            - w_2y * sin(phi_x)
            + w_2z * cos(phi_x) * cos(phi_y),
        ]
    )

    B2h_omega_IB3 = np.dot(R_IB2h.T, np.dot(R_IB3, B3_omega_IB3))

    # extract lever arm angular velocity wrt B2h frame
    B2h_phi_x_dot = B2h_omega_IB3[0]
    B2h_phi_y_dot = B2h_omega_IB3[1]

    # heading / angular z velocity
    z = np.array([np.arctan2(x[1], x[0]), yaw_B2h_rate])

    # principal motor axis direction (y-axis)
    y = np.zeros(6)

    y[StateIndex.ALPHA_1_IDX] = B2h_phi_y - phi_y
    y[StateIndex.PHI_IDX] = B2h_phi_y
    y[StateIndex.PSI_0_IDX] = B2h_psi_y

    y[StateIndex.ALPHA_DOT_1_IDX] = B2h_omega_2y
    y[StateIndex.PHI_DOT_IDX] = B2h_phi_y_dot
    y[StateIndex.PSI_DOT_0_IDX] = B2h_psi_y_dot

    # lateral motor axis direction (x-axis)

    x = np.zeros(6)
    x[StateIndex.ALPHA_1_IDX] = B2h_phi_x - phi_x
    x[StateIndex.PHI_IDX] = B2h_phi_x
    x[StateIndex.PSI_0_IDX] = B2h_psi_x

    x[StateIndex.ALPHA_DOT_1_IDX] = B2h_omega_2x
    x[StateIndex.PHI_DOT_IDX] = B2h_phi_x_dot
    x[StateIndex.PSI_DOT_0_IDX] = B2h_psi_x_dot

    return x, y, z


class Controller(object):
    def __init__(self, _):
        self.ctrl_2d = Controller2D(getDefaultParam(2))
        self.ctrl_2d.beta_dot_max = 2.5

        self.omega_2_z_to_omega_2_y_max = 0.3

        self.K = np.array(
            [
                [-0.80754292, 12.15076322, -17.24354006, 10.23122588, -2.94008421, 0.34736014],
                [-3.94498008, 1.86483491, -2.55867509, 1.50560133, -0.42315158, 0.04674095],
                [28.56857473, -46.54088186, 44.42735085, -11.57973861, -1.50021012, 0.75523765],
                [-4.25523481, 0.27597899, 9.92376958, -11.74578938, 4.85308982, -0.68952721],
                [0.14795505, 0.36941466, 1.01868728, -1.43413212, 0.59845657, -0.08073203],
                [24.7124679, 4.79484575, -58.55748287, 64.51408958, -26.28326935, 3.76317956],
            ]
        )

    def compute_ctrl_input(self, state, beta_cmd, mode=ANGLE_MODE, normalized_phi_x_cmd=0):
        """computes desired motor commands [ux, uy]

        args:
            state (ModelState):        current state of the system
            beta_cmd [rad or rad/s]:   desired forward rolling command of upper ball
            mode [-]:                  specifies the unit beta_cmd: ANGLE_MODE: rad; VELOCITY_MODE: rad/s
            normalized_phi_x_cmd [-]:  desired normalized lateral motor angle phi_x [-]
                                       (normalized by maximum allowed given current state)

        Returns:
            lateral (ux) and forward (uy) motor commands
        """
        x, y, _ = projectModelState(state)

        beta_dot_cmd = beta_cmd if mode == VELOCITY_MODE else self.ctrl_2d.compute_beta_dot_cmd(y, beta_cmd)

        omega_y = y[StateIndex.ALPHA_DOT_1_IDX]

        ux_offset = 0

        # no turning commands while reversing rolling direction
        if beta_dot_cmd * omega_y > 0:
            # limit normalized_phi_x_cmd command
            normalized_phi_x_cmd = 0.9 * np.clip(normalized_phi_x_cmd, -1, 1)

            # given current omega_y and desired beta_dot_cmd, pick the lower phi_x_max of the two
            o_y = np.array([omega_y, beta_dot_cmd])
            phi_x_cmd = normalized_phi_x_cmd * min(self.get_phi_x_max(o_y), key=abs)

            # given phi_x_cmd and the current omega_y and desired beta_dot_cmd, pick the lower command offset
            A = phi_x_cmd * np.column_stack(np.abs([np.ones(o_y.shape), o_y, o_y**2]))
            ux_offset = min(np.dot(A, np.array([0.89283332, -0.86604715, 0.77455687])), key=abs)

        else:
            # avoid accelerating fast into the opposite direction
            beta_dot_cmd = np.clip(beta_dot_cmd, -0.2, 0.2)

        uy = self.ctrl_2d.compute_ctrl_input(y, beta_dot_cmd, VELOCITY_MODE)

        Kx = np.dot(self.K, np.array([1, omega_y**2, np.abs(omega_y**3), omega_y**4, np.abs(omega_y**5), omega_y**6]))

        ux = np.dot(Kx, x) + ux_offset

        return np.array([ux, uy])

    def get_phi_x_max(self, omega_y):
        """get maximum lateral motor angle phi_x angular at a given forward rolling velocity y of upper ball

        args:
            omega_y [rad/s]: angular forward rolling velocity y of upper ball

        Returns:
            maximum lateral motor angle phi_x for which the system is still stable
        """
        omega_y = np.array(omega_y)

        A = np.abs(np.column_stack([np.ones(omega_y.shape), omega_y, omega_y**2, omega_y**3, omega_y**4]))

        return np.dot(A, np.array([0.97101945, -0.77551541, -0.99597959, 1.55876344, -0.47934461]))
