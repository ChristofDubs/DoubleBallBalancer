"""Controller class for controlling 3D Double Ball Balancer
"""
import numpy as np
from dynamic_model_3d import ModelState
from rotation import Quaternion


class LQRController:
    def __init__(self,):
        self.K = np.array([[-2.00368209e+02,
                            4.33763458e-13,
                            4.10947256e+01,
                            -1.12238067e-13,
                            -1.25566071e+02,
                            -1.62299604e-14,
                            4.47213595e+00,
                            6.82370587e-14,
                            7.59597871e+00,
                            3.90852326e-14],
                           [6.97197406e-14,
                            -2.00368209e+02,
                            2.02852999e-14,
                            4.10947256e+01,
                            -2.03721204e-14,
                            -1.25566071e+02,
                            1.43508979e-14,
                            4.47213595e+00,
                            -2.42367057e-15,
                            7.59597871e+00],
                           [0.00000000e+00,
                            0.00000000e+00,
                            0.00000000e+00,
                            0.00000000e+00,
                            0.00000000e+00,
                            0.00000000e+00,
                            0.00000000e+00,
                            0.00000000e+00,
                            0.00000000e+00,
                            0.00000000e+00]])

    def compute_ctrl_input(self, state, omega_2_cmd):
        [phi_x, phi_y, phi_z] = state.phi

        ang = np.concatenate([state.psi, [phi_x, phi_y]])
        omega = np.concatenate([state.psi_dot, state.omega_2[:2], state.omega_3[:2]])

        return -np.dot(self.K, np.concatenate([ang, omega])) + \
            np.concatenate([np.dot(self.K[0:2, 6:8] - 1, omega_2_cmd), [0]])
