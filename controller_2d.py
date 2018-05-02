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

        self.psi_max = 0.2
        self.beta_ddot_max = 2 * self.psi_max

    def compute_ctrl_input(self, x, u, mode=BETA_IDX):

        beta_dot_cmd = None
        if mode is BETA_IDX:
            beta_dot_cmd = self.compute_beta_dot_cmd(x, u)
        elif mode is BETA_DOT_IDX:
            beta_dot_cmd = u

        psi_cmd = None
        if beta_dot_cmd is not None:
            psi_cmd = self.compute_psi_cmd(x, beta_dot_cmd)
        elif mode is PSI_IDX:
            psi_cmd = u

        psi_dot_cmd = None
        if psi_cmd is not None:
            psi_dot_cmd = self.compute_psi_dot_cmd(x, psi_cmd)
        elif mode is PSI_DOT_IDX:
            psi_dot_cmd = u

        phi_cmd = None
        if psi_dot_cmd is not None:
            phi_cmd = self.compute_phi_cmd(x, psi_dot_cmd)
        elif mode is PHI_IDX:
            phi_cmd = u

        phi_dot_cmd = None
        if phi_cmd is not None:
            phi_dot_cmd = self.compute_phi_dot_cmd(x, phi_cmd)
        elif mode is PHI_DOT_IDX:
            phi_dot_cmd = u

        if phi_dot_cmd is not None:
            return self.compute_motor_cmd(x, phi_dot_cmd)

        print('unknown mode: {} is not in [0,{}]'.format(mode, STATE_SIZE - 1))
        return 0

    def compute_beta_dot_cmd(self, x, beta_cmd):
        delta_beta = beta_cmd - x[BETA_IDX]
        beta_dot_cmd = self.kp * delta_beta - self.kd * x[BETA_DOT_IDX]

        # limit beta_dot_max to beta_dot on maximum deceleration trajectory:
        # beta_dot_max(delta_beta) = sqrt(2 * beta_ddot_max * delta_beta)
        beta_dot_cmd_max = np.sqrt(2 * self.beta_ddot_max * abs(delta_beta))

        # augment beta_dot_max such that if beta_dot = beta_dot_max, psi_cmd = psi_cmd_max
        beta_dot_cmd_max += self.psi_max * self.K[PSI_IDX] / (3 * (self.K[BETA_DOT_IDX] - 1))

        # only crop beta_dot_command if the full deceleration feed-forward still
        # leads to a positive beta_dot_cmd
        if beta_dot_cmd_max > 0 and abs(beta_dot_cmd) > beta_dot_cmd_max:
            beta_dot_cmd = np.sign(beta_dot_cmd) * beta_dot_cmd_max

        return beta_dot_cmd

    def compute_psi_cmd(self, x, beta_dot_cmd):
        return 3 * (self.K[BETA_DOT_IDX] - 1) / self.K[PSI_IDX] * (beta_dot_cmd - x[BETA_DOT_IDX])

    def compute_psi_dot_cmd(self, x, psi_cmd):
        # prevent irrecoverable state by limiting psi
        psi_cmd = saturate(psi_cmd, self.psi_max)
        return 1.0 / 3 * self.K[PSI_IDX] / self.K[PSI_DOT_IDX] * (psi_cmd - x[PSI_IDX])

    def compute_phi_cmd(self, x, psi_dot_cmd):
        return self.K[PSI_DOT_IDX] / self.K[PHI_IDX] * \
            (psi_dot_cmd - x[PSI_DOT_IDX]) - 2.0 / 3 * self.K[PSI_IDX] / self.K[PHI_IDX] * x[PSI_IDX]

    def compute_phi_dot_cmd(self, x, phi_cmd):
        # prevent phi commands outside [-pi/2, pi/2]
        phi_cmd = saturate(phi_cmd, np.pi / 2)
        return self.K[PHI_IDX] * (phi_cmd - x[PHI_IDX]) - self.K[PHI_DOT_IDX] * x[PHI_DOT_IDX]

    def compute_motor_cmd(self, x, phi_dot_cmd):
        return phi_dot_cmd - x[BETA_DOT_IDX]
