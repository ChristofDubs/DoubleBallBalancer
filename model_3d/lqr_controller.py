"""LQR-Controller class for controlling 3D Double Ball Balancer
"""
import numpy as np
import context

from pyrotation import Quaternion, rot_z


class LQRController(object):
    def __init__(self, param):
        self.K = np.array([[-4.73012270e+01, -6.66979775e-15, 1.03556079e+01, 7.60989720e-15, 1.03556079e+01, 7.60988241e-15, -3.53469304e+01, 2.10246008e-14, 3.84194544e+00, -7.70799102e-15, 6.05877473e-01, -1.59797142e-15],
                           [8.07280161e-14, -4.73012270e+01, -1.15536119e-14, 1.03556079e+01, -1.13735129e-14, 1.03556079e+01, 5.72607634e-14, -3.53469304e+01, -2.69694302e-15, 3.84194544e+00, 2.53048715e-15, 6.05877473e-01]])

    def compute_ctrl_input(self, state, w2_y_cmd):

        delta_phi = state.q3.get_roll_pitch_yaw()
        beta_x = -state.phi_x + delta_phi[0]
        beta_y = -state.phi_y + delta_phi[1]

        w2 = state.omega_2

        R_IB2 = Quaternion(state.q2).rotation_matrix()
        R_IB2h = rot_z(delta_phi[2])

        # express psi vector in B2h frame
        B2h_psi = np.dot(R_IB2h[:2, :2].T, state.psi)
        B2h_psi_dot = np.dot(R_IB2h[:2, :2].T, state.psi_dot)

        # express upper ball velocity in B2h frame
        B2h_omega_IB2 = np.dot(R_IB2h.T, np.dot(R_IB2, state.omega_2))

        # extract upper ball velocity wrt B2h frame
        B2h_omega_2x = B2h_omega_IB2[0]
        B2h_omega_2y = B2h_omega_IB2[1]

        # [psi_x, psi_y, beta_x, beta_y, phi_x, phi_y, psi_x_dot, psi_y_dot, w_2x, w_2y, phi_x_dot, phi_y_dot]
        reduced_state = np.array([B2h_psi[0],
                                  B2h_psi[1],
                                  beta_x,
                                  beta_y,
                                  state.phi_x,
                                  state.phi_y,
                                  B2h_psi_dot[0],
                                  B2h_psi_dot[1],
                                  B2h_omega_2x,
                                  B2h_omega_2y - w2_y_cmd,
                                  state.phi_x_dot,
                                  state.phi_y_dot + w2_y_cmd])

        return -np.dot(self.K, reduced_state) - np.array([0, w2_y_cmd])
