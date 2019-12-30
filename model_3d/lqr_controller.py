"""LQR-Controller class for controlling 3D Double Ball Balancer
"""
import numpy as np
import context


class LQRController(object):
    def __init__(self, param):
        self.K = np.array([[-4.73012270e+01, -4.96206733e-14, 1.03556079e+01, 7.86862983e-15, 1.03556079e+01, 7.86862983e-15, -3.53469304e+01, -4.04074365e-14, 3.84194544e+00, 3.26777968e-15, 6.05877473e-01, -1.15481340e-15],
                           [-8.25081941e-14, -4.73012270e+01, 1.55899759e-14, 1.03556079e+01, 1.64910767e-14, 1.03556079e+01, -5.08647924e-14, -3.53469304e+01, 4.02842972e-15, 3.84194544e+00, -1.22464925e-15, 6.05877473e-01]])

    def compute_ctrl_input(self, state, w2_y_cmd):

        delta_phi = state.q3.get_roll_pitch_yaw()
        beta_x = -state.phi_x + delta_phi[0]
        beta_y = -state.phi_y + delta_phi[1]

        w2 = state.omega_2

        # [psi_x, psi_y, beta_x, beta_y, phi_x, phi_y, psi_x_dot, psi_y_dot, w_2x, w_2y, phi_x_dot, phi_y_dot]
        reduced_state = np.array([state.psi_x,
                                  state.psi_y,
                                  beta_x,
                                  beta_y,
                                  state.phi_x,
                                  state.phi_y,
                                  state.psi_x_dot,
                                  state.psi_y_dot,
                                  w2[0],
                                  w2[1] - w2_y_cmd,
                                  state.phi_x_dot,
                                  state.phi_y_dot + w2_y_cmd])

        return -np.dot(self.K, reduced_state)
