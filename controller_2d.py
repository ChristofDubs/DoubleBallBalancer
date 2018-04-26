"""Controller class for controlling 2D Double Ball Balancer
"""
import numpy as np

from definitions_2d import *


def saturate(x, limit):
    """helper function to limit a value x to within [-limit, limit]"""
    return max(-limit, min(limit, x))


class LQRController:
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


class Controller:
    def __init__(self,):
        self.K = np.array([2.67619260e-15, 1.03556079e+01, -4.73012271e+01,
                           3.23606798e+00, 6.05877477e-01, -3.53469304e+01])
        self.kp = 0.2
        self.kd = 0.2
        self.beta_dot_max = 2

    def compute_ctrl_input(self, x, beta_cmd):
        beta_dot_cmd = self.compute_beta_dot_cmd(x, beta_cmd)
        psi_cmd = self.compute_psi_cmd(x, beta_dot_cmd)
        psi_dot_cmd = self.compute_psi_dot_cmd(x, psi_cmd)
        phi_cmd = self.compute_phi_cmd(x, psi_dot_cmd)
        phi_dot_cmd = self.compute_phi_dot_cmd(x, phi_cmd)
        return self.compute_motor_cmd(x, phi_dot_cmd)

    def compute_beta_dot_cmd(self, x, beta_cmd):
        return saturate(
            self.kp * (beta_cmd - x[BETA_IDX]) - self.kd * x[BETA_DOT_IDX], self.beta_dot_max)

    def compute_psi_cmd(self, x, beta_dot_cmd):
        return (self.K[BETA_DOT_IDX] - 1) / self.K[PSI_IDX] * (beta_dot_cmd - x[BETA_DOT_IDX])

    def compute_psi_dot_cmd(self, x, psi_cmd):
        return self.K[PSI_IDX] / self.K[PSI_DOT_IDX] * (psi_cmd - x[PSI_IDX])

    def compute_phi_cmd(self, x, psi_dot_cmd):
        return self.K[PSI_DOT_IDX] / self.K[PHI_IDX] * (psi_dot_cmd - x[PSI_DOT_IDX])

    def compute_phi_dot_cmd(self, x, phi_cmd):
        return self.K[PHI_IDX] * (phi_cmd - x[PHI_IDX]) - self.K[PHI_DOT_IDX] * x[PHI_DOT_IDX]

    def compute_motor_cmd(self, x, phi_dot_cmd):
        return phi_dot_cmd - x[BETA_DOT_IDX]
