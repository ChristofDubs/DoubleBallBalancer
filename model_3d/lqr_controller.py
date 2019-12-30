"""LQR-Controller class for controlling 3D Double Ball Balancer
"""
import numpy as np
import context


class LQRController(object):
    def __init__(self, param):
        self.K = np.array([[-26.67594832, 0., 4.36469406, 0., 4.36469403, 0., -25.83757989, 0., 3.52602561, 0., 0.28995756, 0.],
                           [0., -26.67594832, 0., 4.36469406, 0., 4.36469403, 0., -25.83757989, 0., 3.52602561, 0., 0.28995756]])

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
