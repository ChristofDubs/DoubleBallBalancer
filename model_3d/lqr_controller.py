"""LQR-Controller class for controlling 3D Double Ball Balancer
"""
import numpy as np
import context
from pyrotation import Quaternion


class LQRController(object):
    def __init__(self, param):
        self.K = np.array([[-26.67594832, 0., 4.36469406, 0., 4.36469403, 0., -25.83757989, 0., 3.52602561, 0., 0.28995756, 0.],
                           [0., -26.67594832, 0., 4.36469406, 0., 4.36469403, 0., -25.83757989, 0., 3.52602561, 0., 0.28995756]])

    def compute_ctrl_input(self, state, beta_cmd):

        # [psi_x, psi_y, beta_x, beta_y, phi_x, phi_y, psi_x_dot, psi_y_dot, w_2x, w_2y, phi_x_dot, phi_y_dot]
        beta = Quaternion(state.q2).get_roll_pitch_yaw()
        w2 = state.omega_2
        reduced_state = np.array([state.psi_x,
                                  state.psi_y,
                                  beta[0],
                                  beta[1],
                                  state.phi_x,
                                  state.phi_y,
                                  state.psi_x_dot,
                                  state.psi_y_dot,
                                  w2[0],
                                  w2[1],
                                  state.phi_x_dot,
                                  state.phi_y_dot])

        return -np.dot(self.K, reduced_state)
