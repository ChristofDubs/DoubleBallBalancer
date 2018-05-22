"""Dynamic model of 3D version of Double Ball Balancer

This module contains the class DynamicModel for simulating the non-linear dynamics of the 3D Double Ball Balancer and a corresponding class ModelParam containing the relevant parameters.

author: Christof Dubs
"""
import numpy as np
from numpy import sin, cos
from scipy.integrate import odeint

from definitions_3d import *
from rotation import Quaternion


class ModelParam:
    """Physical parameters of 2D Double Ball Balancer

    The Double Ball Balancer consists of 3 bodies:
        1: lower ball
        2: upper ball
        3: lever arm (todo: add this)

    Physical parameters that multiple bodies have are indexed accordingly.

    Attributes:
        g: Gravitational constant [m/s^2]
        l : Arm length of lever [m] (distance from rotation axis to center of mass)
        m1: Mass of lower ball [kg]
        m2: Mass of upper ball [kg]
        m3: Mass of lever arm [kg]
        r1: Radius of lower ball [m]
        r2: Radius of upper ball [m]
        tau: time constant of speed controlled motor [s] (todo: add this)
        theta1: Mass moment of inertia of lower ball wrt. its center of mass [kg*m^2]
        theta2: Mass moment of inertia of upper ball wrt. its center of mass [kg*m^2]
        theta3: Mass moment of inertia of lever arm wrt. its center of mass [kg*m^2] (todo: add this)
    """

    def __init__(self,):
        """Initializes the parameters to default values"""
        self.g = 9.81
        self.l = 0.5
        self.m1 = 1.0
        self.m2 = 1.0
        self.m3 = 1.0
        self.r1 = 1.0
        self.r2 = 1.0
        self.tau = 0.001
        self.theta1 = 1.0
        self.theta2 = 1.0
        self.theta3 = 1.0

    def is_valid(self,):
        """Checks validity of parameter configuration

        Returns:
            bool: True if valid, False if invalid.
        """
        return self.g > 0 and self.l > 0 and self.m1 > 0 and self.m2 > 0 and self.m3 > 0 and self.r1 > 0 and self.r2 > 0 and self.tau > 0 and self.theta1 > 0 and self.theta2 > 0 and self.theta3 > 0


class DynamicModel:
    """Simulation interface for the 2D Double Ball Balancer

    Attributes:
        p (ModelParam): physical parameters
        x (numpy.ndarray): state [beta, phi, psi, beta_dot, phi_dot, psi_dot]

    Functions that are not meant to be called from outside the class (private methods) are prefixed with a single underline.
    """

    def __init__(self, param, x0=None):
        """Initializes attributes to default values

        args:
            param (ModelParam): parameters of type ModelParam
            x0 (numpy.ndarray, optional): initial state. Set to equilibrium state if not specified
        """
        self.p = param
        if not param.is_valid():
            print('Warning: not all parameters set!')

        if x0 is None or not self.set_state(x0):
            self.x = np.zeros(STATE_SIZE)
            self.x[Q_1_W_IDX] = 1.0
            self.x[Q_2_W_IDX] = 1.0

    def set_state(self, x0):
        """Set the state.

        This function allows to set the initial state.

        args:
            x0 (numpy.ndarray): initial state

        Returns:
            bool: True if state could be set successfully, False otherwise.
        """
        if not isinstance(x0, np.ndarray):
            print(
                'called set_state with argument of type {} instead of numpy.ndarray. Ignoring.'.format(
                    type(x0)))
            return False

        # make 1D version of x0
        x0_flat = x0.flatten()
        if len(x0_flat) != STATE_SIZE:
            print(
                'called set_state with array of length {} instead of {}. Ignoring.'.format(
                    len(x0_flat), STATE_SIZE))
            return False
        self.x = x0_flat

        # todo: fix this by making a nicer state class
        self.x[Q_1_W_IDX] = 1.0
        self.x[Q_2_W_IDX] = 1.0

        return True

    def simulate_step(self, delta_t):
        """Simulate one time step

        Simulates the changes of the state over a time interval.

        args:
            delta_t: time step [s]
        """
        t = np.array([0, delta_t])
        self.x = odeint(self._x_dot, self.x, t)[-1]

        # normalize quaternions
        self.x[Q_1_W_IDX: Q_1_Z_IDX + 1] *= 1.0 / np.linalg.norm(self.x[Q_1_W_IDX: Q_1_Z_IDX + 1])
        self.x[Q_2_W_IDX: Q_2_Z_IDX + 1] *= 1.0 / np.linalg.norm(self.x[Q_2_W_IDX: Q_2_Z_IDX + 1])

    def is_irrecoverable(self, x=None):
        """Checks if system is recoverable

        args:
            x0 (numpy.ndarray, optional): state. If not specified, the internal state is checked

        Returns:
            bool: True if state is irrecoverable, False otherwise.
        """
        if x is None:
            x = self.x

        psi_y = x[PSI_Y_IDX]

        # upper ball falling off the lower ball
        if psi_y < -np.pi / 2 or psi_y > np.pi / 2:
            return True

        # upper ball touching the ground
        if self.p.r2 > self.p.r1:
            psi_crit = np.arccos((self.p.r2 - self.p.r1) /
                                 (self.p.r2 + self.p.r1))
            if np.abs(psi_y) > psi_crit:
                return True

        # todo: decide how to check / what to do if upper ball lifts off
        return False

    def get_visualization(self, x=None):
        """Get visualization of the system for plotting

        Usage example:
            v = model.get_visualization()
            plt.plot(*v['lower_ball'])

        args:
            x (numpy.ndarray, optional): state. If not specified, the internal state is used

        Returns:
            dict: dictionary with keys "lower_ball", "upper_ball" and "lever_arm". The value for each key is a list with three elements: a list of x coordinates, a list of y coordinates and a list of z coordinates.
        """
        if x is None:
            x = self.x

        vis = {}

        r_OSi = self._compute_r_OSi(x)
        vis['lower_ball'] = self._compute_ball_visualization(
            r_OSi[0], self.p.r1, Quaternion(x[Q_1_W_IDX: Q_1_Z_IDX + 1]))
        vis['upper_ball'] = self._compute_ball_visualization(
            r_OSi[1], self.p.r2, Quaternion(x[Q_2_W_IDX: Q_2_Z_IDX + 1]))
        # vis['lever_arm'] = [[r_OSi[1][0], r_OSi[2][0]], [r_OSi[1][1], r_OSi[2][1]]]
        return vis

    def _x_dot(self, x, t):
        """computes the derivative of the state

        This function returns an numpy.array of the derivatives of the states, given the current state and inputs.

        Its signature is compatible with scipy.integrate.odeint's first callable argument.

        args:
            x (numpy.ndarray): initial state
            t: time [s]. Since this system is time invariant, this argument is unused.
        """
        # freeze system if state is irrecoverable
        if self.is_irrecoverable():
            return np.concatenate(
                [np.zeros(10), -100 * x[PSI_Y_DOT_IDX:OMEGA_2_Z_IDX + 1], np.zeros(2)])

        A = self._compute_acc_jacobian_matrix(x)
        b = self._compute_rhs(x)
        omega_dot = np.linalg.solve(A, b)

        omega_1 = self._get_lower_ball_omega(x)

        q1_dot = Quaternion(x[Q_1_W_IDX: Q_1_Z_IDX + 1]).q_dot(omega_1, frame='inertial')

        q2_dot = Quaternion(x[Q_2_W_IDX: Q_2_Z_IDX + 1]
                            ).q_dot(x[OMEGA_2_X_IDX:OMEGA_2_Z_IDX + 1], frame='inertial')

        return np.concatenate([x[PSI_Y_DOT_IDX:PSI_Z_DOT_IDX + 1], q1_dot,
                               q2_dot, omega_dot, self._get_lower_ball_vel(omega_1)])

    def _get_lower_ball_vel(self, omega_1):
        """computes the linear velocity (x/y) of the lower ball

        args:
            omega_1 (numpy.ndarray): angular velocity [rad/s] of lower ball
        """
        return self.p.r1 * np.array([omega_1[1], -omega_1[0]])

    def _get_lower_ball_omega(self, x):
        """computes the angular velocity (x/y/z) of the lower ball

        args:
            x (numpy.ndarray): current state
        """
        psi_z = x[PSI_Z_IDX]
        psi_y_dot = x[PSI_Y_DOT_IDX]
        psi_z_dot = x[PSI_Z_DOT_IDX]
        w_2x = x[OMEGA_2_X_IDX]
        w_2y = x[OMEGA_2_Y_IDX]
        w_2z = x[OMEGA_2_Z_IDX]

        w_1x = -(psi_y_dot * self.p.r1 * np.sin(psi_z) + psi_y_dot *
                 self.p.r2 * np.sin(psi_z) + self.p.r2 * w_2x) / self.p.r1
        w_1y = (psi_y_dot * self.p.r1 * np.cos(psi_z) + psi_y_dot *
                self.p.r2 * np.cos(psi_z) - self.p.r2 * w_2y) / self.p.r1
        w_1z = (psi_z_dot * self.p.r1 + psi_z_dot * self.p.r2 - self.p.r2 * w_2z) / self.p.r1

        return np.array([w_1x, w_1y, w_1z])

    def _compute_acc_jacobian_matrix(self, x):
        """computes angular acceleration matrix of rotational part of system dynamics (equal to jacobian matrix since dynamics are linear in angular accelerations)

        The non-linear rotational dynamics are of the form

        A * [psi_y_ddot, psi_z_ddot, omega_2_x_dot, omega_2_y_dot, omega_2_z_dot] = b

        where A = A(psi_y, psi_z) and b = b(psi_y, psi_z, psi_y_dot, psi_z_dot)

        This function computes matrix A.

        args:
            x (numpy.ndarray): current state

        Returns: 5x5 angular acceleration matrix
        """
        psi_y = x[PSI_Y_IDX]
        psi_z = x[PSI_Z_IDX]

        A = np.zeros([5, 5])

        # auto-generated symbolic expressions
        A[0, 0] = (self.p.r1 + self.p.r2)**2 * (self.p.r1**2 * (self.p.m1 + 2 * \
                   self.p.m2 * cos(psi_y) + 2 * self.p.m2) + self.p.theta1) / self.p.r1**2
        A[0, 1] = 0
        A[0, 2] = self.p.r2 * (self.p.m1 * self.p.r1**3 + self.p.m1 * self.p.r1**2 * self.p.r2 + self.p.m2 * self.p.r1**3 * cos(psi_y) + self.p.m2 * self.p.r1**3 + self.p.m2 *
                               self.p.r1**2 * self.p.r2 * cos(psi_y) + self.p.m2 * self.p.r1**2 * self.p.r2 + self.p.r1 * self.p.theta1 + self.p.r2 * self.p.theta1) * sin(psi_z) / self.p.r1**2
        A[0, 3] = -self.p.r2 * (self.p.m1 * self.p.r1**3 + self.p.m1 * self.p.r1**2 * self.p.r2 + self.p.m2 * self.p.r1**3 * cos(psi_y) + self.p.m2 * self.p.r1**3 + self.p.m2 *
                                self.p.r1**2 * self.p.r2 * cos(psi_y) + self.p.m2 * self.p.r1**2 * self.p.r2 + self.p.r1 * self.p.theta1 + self.p.r2 * self.p.theta1) * cos(psi_z) / self.p.r1**2
        A[0, 4] = 0
        A[1, 0] = A[0, 1]
        A[1, 1] = (self.p.r1 + self.p.r2)**2 * (self.p.m2 * self.p.r1 **
                                                2 * sin(psi_y)**2 + self.p.theta1) / self.p.r1**2
        A[1, 2] = self.p.m2 * self.p.r2 * (self.p.r1 + self.p.r2) * sin(psi_y) * cos(psi_z)
        A[1, 3] = self.p.m2 * self.p.r2 * (self.p.r1 + self.p.r2) * sin(psi_y) * sin(psi_z)
        A[1, 4] = -self.p.r2 * self.p.theta1 * (self.p.r1 + self.p.r2) / self.p.r1**2
        A[2, 0] = A[0, 2]
        A[2, 1] = A[1, 2]
        A[2, 2] = self.p.m1 * self.p.r2**2 + self.p.m2 * self.p.r2**2 + \
            self.p.theta2 + self.p.r2**2 * self.p.theta1 / self.p.r1**2
        A[2, 3] = 0
        A[2, 4] = 0
        A[3, 0] = A[0, 3]
        A[3, 1] = A[1, 3]
        A[3, 2] = A[2, 3]
        A[3, 3] = self.p.m1 * self.p.r2**2 + self.p.m2 * self.p.r2**2 + \
            self.p.theta2 + self.p.r2**2 * self.p.theta1 / self.p.r1**2
        A[3, 4] = 0
        A[4, 0] = A[0, 4]
        A[4, 1] = A[1, 4]
        A[4, 2] = A[2, 4]
        A[4, 3] = A[3, 4]
        A[4, 4] = self.p.theta2 + self.p.r2**2 * self.p.theta1 / self.p.r1**2

        return A

    def _compute_rhs(self, x):
        """computes state (and input) terms of system dynamics

        The non-linear rotational dynamics are of the form

        A * [psi_y_ddot, psi_z_ddot, omega_2_x_dot, omega_2_y_dot, omega_2_z_dot] = b

        where A = A(psi_y, psi_z) and b = b(psi_y, psi_z, psi_y_dot, psi_z_dot)

        This function computes vector b.

        args:
            x (numpy.ndarray): current state

        Returns: 5x1 array of state (and input) terms
        """
        b = np.zeros(5)

        psi_y = x[PSI_Y_IDX]
        psi_z = x[PSI_Z_IDX]
        psi_y_dot = x[PSI_Y_DOT_IDX]
        psi_z_dot = x[PSI_Z_DOT_IDX]

        # auto-generated symbolic expressions
        b[0] = self.p.m2 * (psi_y_dot**2 * self.p.r1**2 + 2 * psi_y_dot**2 * self.p.r1 * self.p.r2 + psi_y_dot**2 * self.p.r2**2 + psi_z_dot**2 * self.p.r1**2 * cos(psi_y) + psi_z_dot**2 * self.p.r1**2 + 2 * psi_z_dot **
                            2 * self.p.r1 * self.p.r2 * cos(psi_y) + 2 * psi_z_dot**2 * self.p.r1 * self.p.r2 + psi_z_dot**2 * self.p.r2**2 * cos(psi_y) + psi_z_dot**2 * self.p.r2**2 + self.p.g * self.p.r1 + self.p.g * self.p.r2) * sin(psi_y)
        b[1] = -psi_y_dot * psi_z_dot * self.p.m2 * \
            (self.p.r1 + self.p.r2)**2 * (sin(psi_y) + sin(2 * psi_y))
        b[2] = psi_y_dot**2 * self.p.m2 * self.p.r1 * self.p.r2 * sin(psi_y) * sin(psi_z) + psi_y_dot**2 * self.p.m2 * self.p.r2**2 * sin(psi_y) * sin(psi_z) - psi_y_dot * psi_z_dot * self.p.m1 * self.p.r1 * self.p.r2 * cos(psi_z) - psi_y_dot * psi_z_dot * self.p.m1 * self.p.r2**2 * cos(psi_z) - 2 * psi_y_dot * psi_z_dot * self.p.m2 * self.p.r1 * self.p.r2 * cos(psi_y) * cos(psi_z) - psi_y_dot * psi_z_dot * self.p.m2 * self.p.r1 * self.p.r2 * cos(
            psi_z) - 2 * psi_y_dot * psi_z_dot * self.p.m2 * self.p.r2**2 * cos(psi_y) * cos(psi_z) - psi_y_dot * psi_z_dot * self.p.m2 * self.p.r2**2 * cos(psi_z) - psi_y_dot * psi_z_dot * self.p.r2 * self.p.theta1 * cos(psi_z) / self.p.r1 - psi_y_dot * psi_z_dot * self.p.r2**2 * self.p.theta1 * cos(psi_z) / self.p.r1**2 + psi_z_dot**2 * self.p.m2 * self.p.r1 * self.p.r2 * sin(psi_y) * sin(psi_z) + psi_z_dot**2 * self.p.m2 * self.p.r2**2 * sin(psi_y) * sin(psi_z)
        b[3] = -psi_y_dot**2 * self.p.m2 * self.p.r1 * self.p.r2 * sin(psi_y) * cos(psi_z) - psi_y_dot**2 * self.p.m2 * self.p.r2**2 * sin(psi_y) * cos(psi_z) - psi_y_dot * psi_z_dot * self.p.m1 * self.p.r1 * self.p.r2 * sin(psi_z) - psi_y_dot * psi_z_dot * self.p.m1 * self.p.r2**2 * sin(psi_z) - 2 * psi_y_dot * psi_z_dot * self.p.m2 * self.p.r1 * self.p.r2 * sin(psi_z) * cos(psi_y) - psi_y_dot * psi_z_dot * self.p.m2 * self.p.r1 * self.p.r2 * sin(
            psi_z) - 2 * psi_y_dot * psi_z_dot * self.p.m2 * self.p.r2**2 * sin(psi_z) * cos(psi_y) - psi_y_dot * psi_z_dot * self.p.m2 * self.p.r2**2 * sin(psi_z) - psi_y_dot * psi_z_dot * self.p.r2 * self.p.theta1 * sin(psi_z) / self.p.r1 - psi_y_dot * psi_z_dot * self.p.r2**2 * self.p.theta1 * sin(psi_z) / self.p.r1**2 - psi_z_dot**2 * self.p.m2 * self.p.r1 * self.p.r2 * sin(psi_y) * cos(psi_z) - psi_z_dot**2 * self.p.m2 * self.p.r2**2 * sin(psi_y) * cos(psi_z)
        b[4] = 0

        return b

    def _compute_r_OSi(self, x):
        """computes center of mass locations of all bodies

        args:
            x (numpy.ndarray): current state

        Returns: list of x/y/z coordinates of center of mass of lower ball, upper ball, and lever arm.
        """
        pos_x = x[X_IDX]
        pos_y = x[Y_IDX]
        psi_y = x[PSI_Y_IDX]
        psi_z = x[PSI_Z_IDX]

        r_OS1 = np.array([pos_x, pos_y, self.p.r1])

        r12 = self.p.r1 + self.p.r2
        r_S1S2 = r12 * np.array([cos(psi_z) * sin(psi_y), sin(psi_z) * sin(psi_y), cos(psi_y)])

        r_OS2 = r_OS1 + r_S1S2

        return [r_OS1, r_OS2]

    def _compute_ball_visualization(self, center, radius, q_IB):
        """computes visualization points of a ball

        This function computes the points on the ball surface as well as a line that indicates where angle zero is.

        args:
            center (numpy.ndarray): center of the ball where x=center[0] and y=center[1] in [m]
            radius : ball radius [m]
            q_IB (Quaternion): orientation of the ball frame B wrt. inertial frame I

        Returns: list of x/y/z coordinates of ball surface and zero angle reference
        """
        circle_res = 20

        lon = np.linspace(0, 2 * np.pi, circle_res)
        lat = np.linspace(0, np.pi, circle_res / 2)
        x = radius * np.outer(np.cos(lon), np.sin(lat))
        y = radius * np.outer(np.sin(lon), np.sin(lat))
        z = radius * np.outer(np.ones(circle_res), np.cos(lat))

        R_IB = q_IB.rotation_matrix()

        x_rot = R_IB[0, 0] * x + R_IB[0, 1] * y + R_IB[0, 2] * z
        y_rot = R_IB[1, 0] * x + R_IB[1, 1] * y + R_IB[1, 2] * z
        z_rot = R_IB[2, 0] * x + R_IB[2, 1] * y + R_IB[2, 2] * z

        return [center[0] + x_rot, center[1] + y_rot, center[2] + z_rot]
