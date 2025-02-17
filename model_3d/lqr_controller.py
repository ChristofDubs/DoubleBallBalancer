"""LQR-Controller class for controlling 3D Double Ball Balancer
"""

import context  # noqa: F401
import numpy as np
from pyrotation import Quaternion


class LQRController(object):
    def __init__(self, param):
        self.K = np.array(
            [
                [
                    -4.73012270e01,
                    -4.96206733e-14,
                    1.03556079e01,
                    7.86862983e-15,
                    1.03556079e01,
                    7.86862983e-15,
                    -3.53469304e01,
                    -4.04074365e-14,
                    3.84194544e00,
                    3.26777968e-15,
                    6.05877473e-01,
                    -1.15481340e-15,
                ],
                [
                    -8.25081941e-14,
                    -4.73012270e01,
                    1.55899759e-14,
                    1.03556079e01,
                    1.64910767e-14,
                    1.03556079e01,
                    -5.08647924e-14,
                    -3.53469304e01,
                    4.02842972e-15,
                    3.84194544e00,
                    -1.22464925e-15,
                    6.05877473e-01,
                ],
            ]
        )

    def compute_ctrl_input(self, state, w2_y_cmd):

        delta_phi = state.q3.get_roll_pitch_yaw()
        beta_x = -state.phi_x + delta_phi[0]
        beta_y = -state.phi_y + delta_phi[1]

        w2 = state.omega_2

        R_IB2 = Quaternion(state.q2).rotation_matrix()

        # construct horizontal ball frame (cannot use yaw angle of roll-pitch-yaw,
        # because the primary rotation axis is the y-axis, driving straight into
        # the 90deg pitch singularity)
        z = np.array([0, 0, 1])
        x = np.cross(R_IB2[:, 1], z)
        x *= 1 / np.linalg.norm(x)
        y = np.cross(z, x)

        R_IB2h = np.column_stack([x, y, z])[:2, :2]

        # express psi vector in B2h frame
        B2h_psi = np.dot(R_IB2h.T, state.psi)
        B2h_psi_dot = np.dot(R_IB2h.T, state.psi_dot)

        # [psi_x, psi_y, beta_x, beta_y, phi_x, phi_y, psi_x_dot, psi_y_dot, w_2x, w_2y, phi_x_dot, phi_y_dot]
        reduced_state = np.array(
            [
                B2h_psi[0],
                B2h_psi[1],
                beta_x,
                beta_y,
                state.phi_x,
                state.phi_y,
                B2h_psi_dot[0],
                B2h_psi_dot[1],
                w2[0],
                w2[1] - w2_y_cmd,
                state.phi_x_dot,
                state.phi_y_dot + w2_y_cmd,
            ]
        )

        return -np.dot(self.K, reduced_state) - np.array([0, w2_y_cmd])
