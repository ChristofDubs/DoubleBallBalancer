"""Controller class for controlling 3D Double Ball Balancer
"""
import numpy as np
from numpy import sin, cos
import controller_2d
from definitions_2d import BETA_IDX, PHI_IDX, PSI_IDX, BETA_DOT_IDX, PHI_DOT_IDX, PSI_DOT_IDX
from dynamic_model_3d import ModelState
from rotation import Quaternion


class Controller:
    def __init__(self,):
        self.ctrl_2d = controller_2d.Controller()

    def compute_ctrl_input(self, state, beta_cmd):
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

        # express angular velocities induced by psi rates in B2h frame
        I_omega_psi = np.array([psi_x_dot * cos(psi_y), psi_y_dot, -psi_x_dot * sin(psi_y)])
        B2h_omega_psi = np.dot(R_IB2h.T, I_omega_psi)

        # extract psi rates wrt B2h frame
        B2h_psi_x_dot = B2h_omega_psi[0]
        B2h_psi_y_dot = B2h_omega_psi[1]

        # express upper ball velocity in B2h frame
        B2h_omega_IB2 = np.dot(R_IB2h.T, np.dot(R_IB2, state.omega_2))

        # extract upper ball velocity wrt B2h frame
        B2h_omega_2x = B2h_omega_IB2[0]
        B2h_omega_2y = B2h_omega_IB2[1]

        # express lever arm directional vector in B2h frame
        R_IB3 = state.q3.rotation_matrix()
        I_e_S2S3 = np.dot(R_IB3, np.array([0, 0, -1]))
        B2h_e_S2S3 = np.dot(R_IB2h.T, I_e_S2S3)

        # extract lever arm angles wrt B2h frame
        B2h_phi_x = np.arcsin(B2h_e_S2S3[1])
        B2h_phi_y = np.arcsin(-B2h_e_S2S3[0])

        # express lever arm angular velocity in B2h frame
        B3_omega_IB3 = np.array([phi_x_dot +
                                 w_2x *
                                 cos(phi_y) -
                                 w_2z *
                                 sin(phi_y), phi_y_dot *
                                 cos(phi_x) +
                                 w_2x *
                                 sin(phi_x) *
                                 sin(phi_y) +
                                 w_2y *
                                 cos(phi_x) +
                                 w_2z *
                                 sin(phi_x) *
                                 cos(phi_y), -
                                 phi_y_dot *
                                 sin(phi_x) +
                                 w_2x *
                                 sin(phi_y) *
                                 cos(phi_x) -
                                 w_2y *
                                 sin(phi_x) +
                                 w_2z *
                                 cos(phi_x) *
                                 cos(phi_y)])

        B2h_omega_IB3 = np.dot(R_IB2h.T, np.dot(R_IB3, B3_omega_IB3))

        # extract lever arm angular velocity wrt B2h frame
        B2h_phi_x_dot = B2h_omega_IB3[0]
        B2h_phi_y_dot = B2h_omega_IB3[1]

        x = np.zeros(6)

        # rotate along principal motor axis (y-axis)
        x[BETA_IDX] = B2h_phi_y - phi_y
        x[PHI_IDX] = B2h_phi_y
        x[PSI_IDX] = B2h_psi_y

        x[BETA_DOT_IDX] = B2h_omega_2y
        x[PHI_DOT_IDX] = B2h_phi_y_dot
        x[PSI_DOT_IDX] = B2h_psi_y_dot

        uy = self.ctrl_2d.compute_ctrl_input(x, beta_cmd)

        # stabilize lateral axis
        x[BETA_IDX] = B2h_phi_x - phi_x
        x[PHI_IDX] = B2h_phi_x
        x[PSI_IDX] = B2h_psi_x

        x[BETA_DOT_IDX] = B2h_omega_2x
        x[PHI_DOT_IDX] = B2h_phi_x_dot
        x[PSI_DOT_IDX] = B2h_psi_x_dot

        ux = self.ctrl_2d.compute_ctrl_input(x, 0)

        return np.array([ux, uy])
