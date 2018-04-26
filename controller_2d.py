"""Controller class for controlling 2D Double Ball Balancer
"""
import numpy as np

from definitions_2d import *


def saturate(x, limit):
    """helper function to limit a value x to within [-limit, limit]"""
    return max(-limit, min(limit, x))


class Controller:
    def __init__(self,):
        self.K = np.array([2.67619260e-15, 1.03556079e+01, -4.73012271e+01,
                           3.23606798e+00, 6.05877477e-01, -3.53469304e+01])
        self.kp = 0.2
        self.kd = 0.2
        self.beta_dot_max = 2

    def compute_ctrl_input(self, x, beta_cmd):
        # PD beta controller
        beta_dot_cmd = saturate(
            self.kp * (beta_cmd - x[BETA_IDX]) - self.kd * x[BETA_DOT_IDX], self.beta_dot_max)

        # beta_dot controller
        return -np.dot(self.K, x) + (self.K[BETA_DOT_IDX] - 1) * beta_dot_cmd
