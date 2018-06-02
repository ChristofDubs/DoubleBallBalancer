"""Controller class for controlling 3D Double Ball Balancer
"""
import numpy as np
from dynamic_model_3d import ModelState
from rotation import Quaternion


class LQRController:
    def __init__(self,):
        self.K = np.array([[-1.50563963e+02,
                            -1.16735110e-12,
                            2.32011192e+01,
                            8.17677760e-14,
                            -1.09723150e+02,
                            -1.64453044e-12,
                            6.32455532e+00,
                            2.81620898e-13,
                            5.43318319e+00,
                            4.46919531e-14],
                           [-1.28652659e-12,
                            -1.50563963e+02,
                            1.46460222e-13,
                            2.32011192e+01,
                            -1.41935954e-12,
                            -1.09723150e+02,
                            2.17763481e-13,
                            6.32455532e+00,
                            2.46671309e-14,
                            5.43318319e+00],
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

    def compute_ctrl_input(self, state):
        [phi_x, phi_y, phi_z] = state.phi

        ang = np.concatenate([state.psi, [phi_x, phi_y]])
        omega = np.concatenate([state.psi_dot, state.omega_2[:2], state.omega_3[:2]])

        return -np.dot(self.K, np.concatenate([ang, omega]))
