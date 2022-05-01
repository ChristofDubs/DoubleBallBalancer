"""Controller class for controlling 2D Double Ball Balancer
"""
import numpy as np

from .dynamics_3 import StateIndex


def saturate(x, limit):
    """helper function to limit a value x to within [-limit, limit]"""
    return max(-limit, min(limit, x))


class Controller3(object):
    ANGLE_MODE = StateIndex.ALPHA_2_IDX
    VELOCITY_MODE = StateIndex.ALPHA_DOT_2_IDX

    def __init__(self, _):
        self.K = np.array([0., 26.62771567, -199.45731763, -327.80147739, -
                           1.23606798, 9.73717718, -133.08516459, -137.62380736])
        self.kp = 0.2
        self.kd = 0.2
        self.beta_dot_max = None
        self.beta_ddot_max = 5

    def compute_ctrl_input(self, x, beta_cmd, mode=ANGLE_MODE):
        beta_dot_cmd = None
        if mode is self.ANGLE_MODE:
            beta_dot_cmd = self.kp * (beta_cmd - x[StateIndex.ALPHA_2_IDX]) - self.kd * x[StateIndex.ALPHA_DOT_2_IDX]
        elif mode is self.VELOCITY_MODE:
            beta_dot_cmd = beta_cmd
        else:
            assert(False)

        if self.beta_dot_max:
            beta_dot_cmd = saturate(beta_dot_cmd, self.beta_dot_max)

        # beta_dot controller
        beta_ddot_des = (self.K[StateIndex.ALPHA_DOT_2_IDX] - 1) * (beta_dot_cmd - x[StateIndex.ALPHA_DOT_2_IDX])
        beta_ddot_des = saturate(beta_ddot_des, self.beta_ddot_max)

        phi_des = 1 / self.K[StateIndex.PHI_IDX] * (beta_ddot_des - self.K[StateIndex.PSI_1_IDX] * x[StateIndex.PSI_1_IDX] - self.K[StateIndex.PSI_DOT_1_IDX] *
                                                    x[StateIndex.PSI_DOT_1_IDX] - self.K[StateIndex.PSI_0_IDX] * x[StateIndex.PSI_0_IDX] - self.K[StateIndex.PSI_DOT_0_IDX] * x[StateIndex.PSI_DOT_0_IDX])

        phi_des = saturate(phi_des, 0.9 * np.pi / 2)

        phi_dot_des = self.K[StateIndex.PHI_IDX] * \
            (phi_des - x[StateIndex.PHI_IDX]) - self.K[StateIndex.PHI_DOT_IDX] * x[StateIndex.PHI_DOT_IDX]

        return phi_dot_des - x[StateIndex.ALPHA_DOT_2_IDX]
