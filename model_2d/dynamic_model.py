"""Dynamic model of 2D version of Double Ball Balancer

This module contains the class DynamicModel for simulating the non-linear dynamics of the 2D Double Ball Balancer and a corresponding class ModelParam containing the relevant parameters. The control input is the angular velocity command for the motor's speed controller.

author: Christof Dubs
"""
import itertools
import numpy as np
from numpy import sin, cos
from scipy.integrate import odeint

from .definitions import *


class ModelParam(object):
    """Physical parameters of 2D Double Ball Balancer

    The Double Ball Balancer consists of 3 bodies:
        1: lower ball
        2: upper ball
        3: lever arm

    Physical parameters that multiple bodies have are indexed accordingly.

    Attributes:
        g: Gravitational constant [m/s^2]
        l : Arm length of lever [m] (distance from rotation axis to center of mass)
        m1: Mass of lower ball [kg]
        m2: Mass of upper ball [kg]
        m3: Mass of lever arm [kg]
        r1: Radius of lower ball [m]
        r2: Radius of upper ball [m]
        tau: time constant of speed controlled motor [s]
        theta1: Mass moment of inertia of lower ball wrt. its center of mass [kg*m^2]
        theta2: Mass moment of inertia of upper ball wrt. its center of mass [kg*m^2]
        theta3: Mass moment of inertia of lever arm wrt. its center of mass [kg*m^2]
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


class DynamicModel(object):
    """Simulation interface for the 2D Double Ball Balancer

    Attributes:
        p (ModelParam): physical parameters
        x (numpy.ndarray): state [beta, phi, psi, beta_dot, phi_dot, psi_dot]

    Functions that are not meant to be called from outside the class (private methods) are prefixed with a single underline.
    """

    def __init__(self, param, x0=np.zeros(STATE_SIZE)):
        """Initializes attributes to default values

        args:
            param (ModelParam): parameters of type ModelParam
            x0 (numpy.ndarray, optional): initial state. Set to equilibrium state if not specified
        """
        self.p = param
        if not param.is_valid():
            print('Warning: not all parameters set!')

        if not self.set_state(x0):
            self.x = np.zeros(STATE_SIZE)

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
        return True

    def simulate_step(self, delta_t, omega_cmd):
        """Simulate one time step

        Simulates the changes of the state over a time interval.

        args:
            delta_t: time step [s]
            omega_cmd: angular velocity command for lever motor [rad/s]
        """
        t = np.array([0, delta_t])
        self.x = odeint(self._x_dot, self.x, t, args=(omega_cmd,))[-1]

    def is_irrecoverable(
            self,
            x=None,
            contact_forces=None,
            omega_cmd=None,
            ignore_force_check=False):
        """Checks if system is recoverable

        args:
            x (numpy.ndarray, optional): state. If not specified, the internal state is checked
            contact_forces(list(numpy.ndarray), optional): contact forces [N]. If not specified, will be internally calculated
            omega_cmd (optional): motor speed command [rad/s] used for contact force calculation if contact_forces are not specified
            ignore_force_check (optional): If set to True, will skip the contact forces check

        Returns:
            bool: True if state is irrecoverable, False otherwise.
        """
        if x is None:
            x = self.x

        psi = x[PSI_IDX]

        # upper ball falling off the lower ball
        if np.abs(psi) > np.pi / 2:
            return True

        # upper ball touching the ground
        if self.p.r2 > self.p.r1:
            psi_crit = np.arccos((self.p.r2 - self.p.r1) /
                                 (self.p.r2 + self.p.r1))
            if np.abs(psi) > psi_crit:
                return True

        # lift off: contact force between lower and upper ball <= 0
        if not ignore_force_check:
            if contact_forces is None:
                contact_forces = self.compute_contact_forces(x, omega_cmd)

            if np.dot(contact_forces[1], self._compute_e_S1S2(x)) <= 0:
                return True

        return False

    def get_visualization(self, x=None, contact_forces=None, omega_cmd=None):
        """Get visualization of the system for plotting

        Usage example:
            v = model.get_visualization()
            plt.plot(*v['lower_ball'])
            plt.arrow(*vis['F1'])

        args:
            x (numpy.ndarray, optional): state. If not specified, the internal state is used
            contact_forces(list(numpy.ndarray), optional): contact forces [N]. If not specified, will be internally calculated
            omega_cmd (optional): motor speed command [rad/s] used for contact force calculation if contact_forces are not specified

        Returns:
            dict: dictionary with keys "lower_ball", "upper_ball" and "lever_arm". The value for each key is a list with two elements: a list of x coordinates, and a list of y coordinates.
        """
        if x is None:
            x = self.x
        if contact_forces is None:
            contact_forces = self.compute_contact_forces(x, omega_cmd)

        vis = {}
        beta = x[BETA_IDX]
        psi = x[PSI_IDX]

        # rolling constraint
        alpha = psi + (self.p.r2 / self.p.r1) * (psi - beta)

        r_OSi = self._compute_r_OSi(x)
        vis['lower_ball'] = self._compute_ball_visualization(
            r_OSi[0], self.p.r1, alpha)
        vis['upper_ball'] = self._compute_ball_visualization(
            r_OSi[1], self.p.r2, beta)
        vis['lever_arm'] = [np.array([r_OSi[1][i], r_OSi[2][i]]) for i in range(2)]

        force_scale = 0.05
        contact_pt_1 = np.array([r_OSi[0][0], 0])
        vis['F1'] = list(itertools.chain.from_iterable(
            [contact_pt_1, force_scale * contact_forces[0]]))

        contact_pt_2 = r_OSi[0] + self.p.r1 * self._compute_e_S1S2(x)
        vis['F12'] = list(itertools.chain.from_iterable(
            [contact_pt_2, force_scale * contact_forces[1]]))

        vis['F23'] = list(itertools.chain.from_iterable(
            [r_OSi[1], force_scale * contact_forces[2]]))

        return vis

    def compute_contact_forces(self, x=None, omega_cmd=None):
        """computes contact forces between bodies

        This function computes the contact forces between the rigid bodies.

        args:
            x (numpy.ndarray, optional): state. If not specified, the internal state is used
            omega_cmd(optional): motor speed command [rad/s]. Defaults to zero if not specified

        Returns: list of the 3 contact forces [F1, F12, F23] with:
        - F1: force from ground onto lower ball
        - F12: force from lower ball onto upper ball
        - F23: force from upper ball onto lever arm
        """
        if x is None:
            x = self.x
        if omega_cmd is None:
            print('Warning: no omega_cmd specified for contact force calculation; default to 0')
            omega_cmd = 0

        phi = x[PHI_IDX]
        psi = x[PSI_IDX]

        phi_dot = x[PHI_DOT_IDX]
        psi_dot = x[PSI_DOT_IDX]

        x_ddot = self._compute_omega_dot(x, omega_cmd)

        beta_dd = x_ddot[BETA_IDX]
        phi_dd = x_ddot[PHI_IDX]
        psi_dd = x_ddot[PSI_IDX]

        x0 = beta_dd * self.p.r2
        x1 = -self.p.r1 - self.p.r2
        x2 = psi_dot**2
        x3 = self.p.m2 * x2
        x4 = sin(psi)
        x5 = self.p.r1 + self.p.r2
        x6 = x4 * x5
        x7 = x5 * cos(psi)
        x8 = psi_dd * (x1 - x7)
        x9 = cos(phi)
        x10 = phi_dd * self.p.l * self.p.m3
        x11 = sin(phi)
        x12 = phi_dot**2 * self.p.l * self.p.m3
        x13 = self.p.m3 * x2
        x14 = self.p.m3 * x0 + self.p.m3 * x8 + x10 * x9 - x11 * x12 + x13 * x6
        x15 = self.p.m2 * x0 + self.p.m2 * x8 + x14 + x3 * x6
        x16 = psi_dd * x4 * x5
        x17 = self.p.g * self.p.m3 - self.p.m3 * x16 + x10 * x11 + x12 * x9 - x13 * x7
        x18 = self.p.g * self.p.m2 - self.p.m2 * x16 + x17 - x3 * x7

        F1 = np.zeros(2)
        F12 = np.zeros(2)
        F23 = np.zeros(2)

        F1[0] = psi_dd * self.p.m1 * x1 + self.p.m1 * x0 + x15
        F1[1] = self.p.g * self.p.m1 + x18
        F12[0] = x15
        F12[1] = x18
        F23[0] = x14
        F23[1] = x17

        return [F1, F12, F23]

    def _x_dot(self, x, t, u):
        """computes the derivative of the state

        This function returns an numpy.array of the derivatives of the states, given the current state and inputs.

        Its signature is compatible with scipy.integrate.odeint's first callable argument.

        args:
            x (numpy.ndarray): initial state
            t: time [s]. Since this system is time invariant, this argument is unused.
            u: input motor speed command for lever motor [rad/s]
        """
        # freeze system if state is irrecoverable
        if self.is_irrecoverable(ignore_force_check=True):
            return np.concatenate([x[3:], -100 * x[3:]])

        omega_dot = self._compute_omega_dot(x, u)
        return np.concatenate([x[3:], omega_dot])

    def _compute_omega_dot(self, x, omega_cmd):
        """computes angular accelerations of system dynamics

        The non-linear dynamics are of the form

        A * [beta_ddot, phi_ddot, psi_ddot] = b

        where A = A(phi, psi) and b=b(phi,psi,phi_dot,psi_dot,u)

        This function solves for the angular accelerations [beta_ddot, phi_ddot, psi_ddot]

        args:
            x (numpy.ndarray): current state
            omega_cmd: motor speed command [rad/s]


        Returns: array containing the time derivative of the angular velocities
        """
        phi = x[PHI_IDX]
        psi = x[PSI_IDX]

        beta_dot = x[BETA_DOT_IDX]
        phi_dot = x[PHI_DOT_IDX]
        psi_dot = x[PSI_DOT_IDX]

        A = np.zeros([3, 3])
        b = np.zeros(3)

        # auto-generated symbolic expressions
        x0 = self.p.r2**2
        x1 = cos(phi)
        x2 = self.p.l * self.p.m3 * x1
        x3 = self.p.r2 * x2
        x4 = self.p.theta1 / self.p.r1**2
        x5 = sin(phi)
        x6 = self.p.l**2 * self.p.m3
        x7 = self.p.l * self.p.m3 * x5
        x8 = sin(psi)
        x9 = self.p.r1 + self.p.r2
        x10 = x8 * x9
        x11 = cos(psi)
        x12 = -self.p.r1 - self.p.r2
        x13 = -x11 * x9 + x12
        x14 = -x10 * x7 + x13 * x2
        x15 = self.p.r2 * x9
        x16 = self.p.r2 * x13
        x17 = self.p.m1 * self.p.r2 * x12 + self.p.m2 * x16 + self.p.m3 * x16 - x15 * x4
        x18 = x9**2
        x19 = x18 * x8**2
        x20 = x13**2
        x21 = psi_dot**2
        x22 = phi_dot**2
        x23 = x21 * x8 * x9
        x24 = self.p.m3 * x23 - x22 * x7
        x25 = x11 * x21 * x9
        x26 = self.p.g * self.p.m3 - self.p.m3 * x25 + x2 * x22
        A[0, 0] = self.p.m1 * x0 + self.p.m2 * x0 + self.p.m3 * x0 + self.p.theta2 + x0 * x4 + x3
        A[0, 1] = self.p.theta3 + x1**2 * x6 + x3 + x5**2 * x6
        A[0, 2] = x14 + x17
        A[1, 0] = -1
        A[1, 1] = 1
        A[1, 2] = 0
        A[2, 0] = x17
        A[2, 1] = x14
        A[2, 2] = self.p.m1 * x12**2 + self.p.m2 * x19 + self.p.m2 * \
            x20 + self.p.m3 * x19 + self.p.m3 * x20 + x18 * x4
        b[0] = -self.p.l * x1 * x24 - self.p.l * x26 * \
            x5 - self.p.m2 * x15 * x21 * x8 - self.p.r2 * x24
        b[1] = (beta_dot + omega_cmd - phi_dot) / self.p.tau
        b[2] = -self.p.m2 * x13 * x23 + x10 * x26 + x10 * \
            (self.p.g * self.p.m2 - self.p.m2 * x25) - x13 * x24

        return np.linalg.solve(A, b)

    def _compute_r_OSi(self, x):
        """computes center of mass locations of all bodies

        args:
            x (numpy.ndarray): current state

        Returns: list of x/y coordinates of center of mass of lower ball, upper ball, and lever arm.
        """
        beta = x[BETA_IDX]
        phi = x[PHI_IDX]
        psi = x[PSI_IDX]

        # upper ball on lower ball rolling constraint
        alpha = psi + (self.p.r2 / self.p.r1) * (psi - beta)

        # lower ball on ground rolling constraint
        p_x = -self.p.r1 * alpha

        r_OS1 = np.array([p_x, self.p.r1])

        r_S1S2 = (self.p.r1 + self.p.r2) * self._compute_e_S1S2(x)
        r_OS2 = r_OS1 + r_S1S2

        r_S2S3 = self.p.l * np.array([np.sin(phi), - np.cos(phi)])
        r_OS3 = r_OS2 + r_S2S3
        return [r_OS1, r_OS2, r_OS3]

    def _compute_e_S1S2(self, x):
        """computes the unit vector pointing from lower ball center to upper ball center

        args:
            x (ModelState): current state
        returns:
            array containing unit direction pointing from lower ball center to upper ball center
        """
        psi = x[PSI_IDX]
        return np.array([-np.sin(psi), np.cos(psi)])

    def _compute_ball_visualization(self, center, radius, angle):
        """computes visualization points of a ball

        This function computes the points on the ball surface as well as a line that indicates where angle zero is.

        args:
            center (numpy.ndarray): center of the ball where x=center[0] and y=center[1] in [m]
            radius : ball radius [m]
            angle: rotation angle of the ball [rad]

        Returns: list of x/y coordinates of ball surface and zero angle reference
        """
        x_coord = [center[0]]
        y_coord = [center[1]]

        angles = np.linspace(angle, angle + 2 * np.pi, 100)

        x_coord.extend([center[0] - radius * np.sin(a) for a in angles])
        y_coord.extend([center[1] + radius * np.cos(a) for a in angles])

        return [x_coord, y_coord]
