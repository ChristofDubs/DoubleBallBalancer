"""Controller class for controlling 2D Double Ball Balancer
"""
import numpy as np

from .dynamics_2 import StateIndex


def saturate(x, limit):
    """helper function to limit a value x to within [-limit, limit]"""
    return max(-limit, min(limit, x))


class LQRController(object):
    def __init__(self,):
        self.K = np.array([2.67619260e-15, 1.03556079e+01, -4.73012271e+01,
                           3.23606798e+00, 6.05877477e-01, -3.53469304e+01])
        self.kp = 0.2
        self.kd = 0.2
        self.beta_dot_max = 2

    def compute_ctrl_input(self, x, beta_cmd):
        # PD beta controller
        beta_dot_cmd = saturate(self.kp *
                                (beta_cmd -
                                 x[StateIndex.ALPHA_1_IDX]) -
                                self.kd *
                                x[StateIndex.ALPHA_DOT_1_IDX], self.beta_dot_max)

        # beta_dot controller
        return -np.dot(self.K, x) + (self.K[StateIndex.ALPHA_DOT_1_IDX] - 1) * beta_dot_cmd


def compute_phi_max(param):
    """compute lever arm angle for maximum beta_ddot acceleration"""
    return np.arccos(-param["r_2"] * param["m_2"] * param["r_1"] / (param["theta_0"] * param["r_1"]**2 / \
                     param["r_0"] ** 2 + param["theta_1"] + (param["m_0"] + param["m_1"] + param["m_2"]) * param["r_1"]**2))


def compute_phi_from_beta_ddot(beta_ddot, param):
    """compute lever arm angle during constant beta_ddot acceleration"""
    return -np.arctan(param["r_1"] * beta_ddot / param["g"]) - np.arcsin((param["theta_0"] * param["r_1"]**2 / param["r_0"]**2 + param["theta_1"] + (
        param["m_0"] + param["m_1"] + param["m_2"]) * param["r_1"]**2) * beta_ddot / (param["r_2"] * param["m_2"] * np.sqrt(param["g"]**2 + beta_ddot**2 * param["r_1"]**2)))


def compute_psi_from_beta_ddot(beta_ddot, param):
    """compute angle of upper on lower ball during constant beta_ddot acceleration"""
    return -np.arctan(param["r_1"] * beta_ddot / param["g"]) - np.arcsin((1 + (param["theta_0"] / param["r_0"]**2 + \
                      param["m_0"]) / (param["m_1"] + param["m_2"])) * beta_ddot / np.sqrt(beta_ddot**2 + param["g"]**2 / param["r_1"]**2))


def compute_beta_ddot_to_psi_gain(param):
    """beta_ddot to psi gain for small accelerations"""
    return -param["r_1"] / param["g"] * (2 + (param["theta_0"] / param["r_0"]**2 +
                                              param["m_0"]) / (param["m_1"] + param["m_2"]))


def compute_beta_ddot_from_psi(psi, param):
    """compute acceleration beta_ddot from psi angle"""
    return -(param["m_1"] + param["m_2"]) * param["g"] * np.sin(psi) / ((param["theta_0"] / param["r_0"]
                                                                         ** 2 + param["m_0"] + (param["m_1"] + param["m_2"]) * (1 + np.cos(psi))) * param["r_1"])


def compute_phi_from_psi(psi, param):
    """compute lever arm angle for stabilizing upper and lower ball at angle psi"""
    beta_ddot = compute_beta_ddot_from_psi(psi, param)
    return compute_phi_from_beta_ddot(beta_ddot, param)


class Controller(object):
    ANGLE_MODE = StateIndex.ALPHA_1_IDX
    VELOCITY_MODE = StateIndex.ALPHA_DOT_1_IDX

    def __init__(self, param):
        # beta dot controller gains
        self.K = np.array([0.0, 10.3556079, -1.54595284, 0.268081501, 0.605877477, -3.41331294])

        # beta controller gains
        self.kp = 0.2
        self.kd = 0.2

        # beta_ddot -> psi gain
        self.beta_ddot_to_psi_gain = compute_beta_ddot_to_psi_gain(param)

        # limits
        self.psi_max = 0.20
        self.beta_ddot_max = -self.psi_max / self.beta_ddot_to_psi_gain
        self.phi_max = compute_phi_max(param)
        self.beta_dot_max = None

        # save param
        self.param = param

    def compute_ctrl_input(self, x, u, mode=ANGLE_MODE):
        beta_dot_cmd = None
        if mode is self.ANGLE_MODE:
            beta_dot_cmd = self.compute_beta_dot_cmd(x, u)
        elif mode is self.VELOCITY_MODE:
            beta_dot_cmd = u

        if beta_dot_cmd is not None and self.beta_dot_max:
            beta_dot_cmd = max(-self.beta_dot_max, min(self.beta_dot_max, beta_dot_cmd))

        psi_cmd = None
        if beta_dot_cmd is not None:
            psi_cmd = self.compute_psi_cmd(x, beta_dot_cmd)
        elif mode is StateIndex.PSI_0_IDX:
            psi_cmd = u

        phi_cmd = None
        if psi_cmd is not None:
            phi_cmd = self.compute_phi_cmd(x, psi_cmd)
        elif mode is StateIndex.PHI_IDX:
            phi_cmd = u

        phi_dot_cmd = None
        if phi_cmd is not None:
            phi_dot_cmd = self.compute_phi_dot_cmd(x, phi_cmd)
        elif mode is StateIndex.PHI_DOT_IDX:
            phi_dot_cmd = u

        if phi_dot_cmd is not None:
            return self.compute_motor_cmd(x, phi_dot_cmd)

        print('invalid mode: {} is not in {}'.format(mode,
                                                     [StateIndex.ALPHA_1_IDX,
                                                      StateIndex.PHI_IDX,
                                                      StateIndex.PSI_0_IDX,
                                                      StateIndex.ALPHA_DOT_1_IDX,
                                                      StateIndex.PHI_DOT_IDX]))
        return 0

    def compute_beta_dot_cmd(self, x, beta_cmd):
        delta_beta = beta_cmd - x[StateIndex.ALPHA_1_IDX]
        beta_dot_cmd = self.kp * delta_beta - self.kd * x[StateIndex.ALPHA_DOT_1_IDX]

        # limit beta_dot_max to beta_dot on maximum deceleration trajectory:
        # beta_dot_max(delta_beta) = sqrt(2 * beta_ddot_max * delta_beta)
        beta_dot_cmd_max = np.sqrt(2 * self.beta_ddot_max * abs(delta_beta))

        # beta_dot command that results in psi_cmd = psi_cmd_max for beta_dot = 0
        beta_dot_ff = -self.psi_max / (self.K[StateIndex.ALPHA_DOT_1_IDX] * self.beta_ddot_to_psi_gain)

        # augment beta_dot_max such that if beta_dot = beta_dot_max, psi_cmd = psi_cmd_max
        beta_dot_cmd_max -= beta_dot_ff

        # use beta_dot_ff as lower limit for beta_dot_max such that beta_dot
        # commands below beta_dot_ff are not cropped (since they will generally
        # not lead to psi saturation)
        beta_dot_cmd_max = max(beta_dot_cmd_max, beta_dot_ff)

        # crop beta_dot_command
        if abs(beta_dot_cmd) > beta_dot_cmd_max:
            beta_dot_cmd = np.sign(beta_dot_cmd) * beta_dot_cmd_max

        return beta_dot_cmd

    def compute_psi_cmd(self, x, beta_dot_cmd):
        beta_ddot_des = self.K[StateIndex.ALPHA_DOT_1_IDX] * (beta_dot_cmd - x[StateIndex.ALPHA_DOT_1_IDX])

        return compute_psi_from_beta_ddot(beta_ddot_des, self.param)

    def compute_phi_cmd(self, x, psi_cmd):
        # prevent irrecoverable state by limiting psi
        psi_cmd = saturate(psi_cmd, self.psi_max)

        phi_ff = compute_phi_from_psi(x[StateIndex.PSI_0_IDX], self.param)

        return self.K[StateIndex.PSI_0_IDX] * (psi_cmd - x[StateIndex.PSI_0_IDX]) - \
            self.K[StateIndex.PSI_DOT_0_IDX] * x[StateIndex.PSI_DOT_0_IDX] + phi_ff

    def compute_phi_dot_cmd(self, x, phi_cmd):
        # prevent phi commands outside approx. [-pi/2, pi/2]
        phi_cmd = saturate(phi_cmd, self.phi_max)

        return self.K[StateIndex.PHI_IDX] * (phi_cmd - x[StateIndex.PHI_IDX]) - \
            self.K[StateIndex.PHI_DOT_IDX] * x[StateIndex.PHI_DOT_IDX]

    def compute_motor_cmd(self, x, phi_dot_cmd):
        return phi_dot_cmd - x[StateIndex.ALPHA_DOT_1_IDX]
