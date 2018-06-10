"""Controller class for controlling 3D Double Ball Balancer
"""
import numpy as np
import controller_2d
from definitions_2d import BETA_IDX, PHI_IDX, PSI_IDX, BETA_DOT_IDX, PHI_DOT_IDX, PSI_DOT_IDX
from dynamic_model_3d import ModelState


class Controller:
    def __init__(self,):
        self.ctrl_2d = controller_2d.Controller()

    def compute_ctrl_input(self, state, beta_cmd):
        [phi_x, phi_y] = state.phi
        [phi_x_dot, phi_y_dot] = state.phi_dot
        [psi_x, psi_y] = state.psi
        [psi_x_dot, psi_y_dot] = state.psi_dot
        [w_2x, w_2y, w_2z] = state.omega_2

        # lever arm orientation wrt. inertial frame
        [phi_ix, phi_iy, _] = state.q3.get_roll_pitch_yaw()

        x = np.zeros(6)

        # rotate along principal motor axis (y-axis)
        x[BETA_IDX] = phi_iy - phi_y
        x[PHI_IDX] = phi_iy
        x[PSI_IDX] = psi_y

        x[BETA_DOT_IDX] = w_2y
        x[PHI_DOT_IDX] = w_2y + phi_y_dot
        x[PSI_DOT_IDX] = psi_y_dot

        uy = self.ctrl_2d.compute_ctrl_input(x, beta_cmd)

        # stabilize lateral axis
        x[BETA_IDX] = phi_ix - phi_x
        x[PHI_IDX] = phi_ix
        x[PSI_IDX] = psi_x

        x[BETA_DOT_IDX] = w_2x
        x[PHI_DOT_IDX] = w_2x + phi_x_dot
        x[PSI_DOT_IDX] = psi_x_dot

        ux = self.ctrl_2d.compute_ctrl_input(x, 0)

        return np.array([ux, uy])
