"""Dynamic model of 2D version of Double Ball Balancer

This module contains the class DynamicModel for simulating the non-linear dynamics of the 2D Double Ball Balancer and a corresponding class ModelParam containing the relevant parameters. The control input is the angular velocity command for the motor's speed controller.

author: Christof Dubs
"""
import numpy as np
from numpy import sin, cos
from scipy.integrate import odeint


class ModelParam:
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
        self.m1 = 1
        self.m2 = 1
        self.m3 = 1
        self.r1 = 1
        self.r2 = 1
        self.tau = 0.001
        self.theta1 = 1
        self.theta2 = 1
        self.theta3 = 1

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

    STATE_SIZE = 6

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
            self.x = np.zeros(self.STATE_SIZE)

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
        if len(x0_flat) != self.STATE_SIZE:
            print(
                'called set_state with array of length {} instead of {}. Ignoring.'.format(
                    len(x0_flat), self.STATE_SIZE))
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

    def is_irrecoverable(self, x=None):
        """Checks if system is recoverable

        args:
            x0 (numpy.ndarray, optional): state. If not specified, the internal state is checked

        Returns:
            bool: True if state is irrecoverable, False otherwise.
        """
        if x is None:
            x = self.x

        psi = x[2]

        # upper ball falling off the lower ball
        if psi < -np.pi / 2 or psi > np.pi / 2:
            return True

        # upper ball touching the ground
        if self.p.r2 > self.p.r1:
            psi_crit = np.arccos((self.p.r2 - self.p.r1) /
                                 (self.p.r2 + self.p.r1))
            if np.abs(psi) > psi_crit:
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
            dict: dictionary with keys "lower_ball", "upper_ball" and "lever_arm". The value for each key is a list with two elements: a list of x coordinates, and a list of y coordinates.
        """
        if x is None:
            x = self.x

        vis = {}
        beta = x[0]
        phi = x[1]
        psi = x[2]

        # rolling constraint
        alpha = psi + (self.p.r2 / self.p.r1) * (psi - beta)

        r_OSi = self._compute_r_OSi(x)
        vis['lower_ball'] = self._compute_ball_visualization(
            r_OSi[0], self.p.r1, alpha)
        vis['upper_ball'] = self._compute_ball_visualization(
            r_OSi[1], self.p.r2, beta)
        vis['lever_arm'] = [[r_OSi[1][0], r_OSi[2][0]], [r_OSi[1][1], r_OSi[2][1]]]
        return vis

    def _x_dot(self, x, t, u):
        """computes the derivative of the state

        This function returns an numpy.array of the derivatives of the states, given the current state and inputs.

        Its signature is compatible with scipy.integrate.odeint's first callable argument.

        args:
            x (numpy.ndarray): initial state
            t: time [s]. Since this system is time invariant, this argument is unused.
            u: input torque command for lever motor [Nm]
        """
        # freeze system if state is irrecoverable
        if self.is_irrecoverable():
            return np.concatenate([x[3:], -100 * x[3:]])

        A = self._compute_acc_jacobian_matrix(x)
        b = self._compute_rhs(x, u)
        omega_dot = np.linalg.solve(A, b)
        return np.concatenate([x[3:], omega_dot])

    def _compute_acc_jacobian_matrix(self, x):
        """computes angular acceleration matrix of system dynamics (equal to jacobian matrix since dynamics are linear in angular accelerations)

        The non-linear dynamics are of the form

        A * [beta_ddot, phi_ddot, psi_ddot] = b

        where A = A(phi, psi) and b=b(phi,psi,phi_dot,psi_dot,u)

        This function computes matrix A.

        args:
            x (numpy.ndarray): current state

        Returns: 3x3 angular acceleration matrix
        """
        phi = x[1]
        psi = x[2]

        A = np.zeros([3, 3])

        # auto-generated symbolic expressions
        A[0, 0] = self.p.l * self.p.m3 * self.p.r2 * cos(phi) + self.p.m1 * self.p.r2**2 + self.p.m2 * \
            self.p.r2**2 + self.p.m3 * self.p.r2**2 + self.p.theta2 + self.p.r2**2 * self.p.theta1 / self.p.r1**2
        A[0, 1] = self.p.l**2 * self.p.m3 + self.p.l * \
            self.p.m3 * self.p.r2 * cos(phi) + self.p.theta3
        A[0, 2] = -(self.p.r1**2 * (self.p.l * self.p.m3 * (self.p.r1 * cos(phi) + self.p.r1 * cos(phi - psi) + self.p.r2 * cos(phi) + self.p.r2 * cos(phi - psi)) + self.p.m1 * self.p.r2 * (self.p.r1 + self.p.r2) + self.p.m2 * self.p.r2 * \
                    (self.p.r1 + self.p.r2 + (self.p.r1 + self.p.r2) * cos(psi)) + self.p.m3 * self.p.r2 * (self.p.r1 + self.p.r2 + (self.p.r1 + self.p.r2) * cos(psi))) + self.p.r2 * self.p.theta1 * (self.p.r1 + self.p.r2)) / self.p.r1**2
        A[1, 0] = -self.p.r2 * (self.p.r1**2 * (self.p.m1 * (self.p.r1 + self.p.r2) + self.p.m2 * (self.p.r1 + self.p.r2 + (self.p.r1 + self.p.r2) * cos(
            psi)) + self.p.m3 * (self.p.r1 + self.p.r2 + (self.p.r1 + self.p.r2) * cos(psi))) + self.p.theta1 * (self.p.r1 + self.p.r2)) / self.p.r1**2
        A[1, 1] = -self.p.l * self.p.m3 * ((self.p.r1 + self.p.r2) * sin(phi) * sin(
            psi) + (self.p.r1 + self.p.r2 + (self.p.r1 + self.p.r2) * cos(psi)) * cos(phi))
        A[1, 2] = (self.p.r1 + self.p.r2)**2 * (self.p.r1**2 * (self.p.m1 + 2 * self.p.m2 * cos(psi) + \
                   2 * self.p.m2 + 2 * self.p.m3 * cos(psi) + 2 * self.p.m3) + self.p.theta1) / self.p.r1**2
        A[2, 0] = -1
        A[2, 1] = 1
        A[2, 2] = 0

        return A

    def _compute_rhs(self, x, omega_cmd):
        """computes state and input terms of system dynamics

        The non-linear dynamics are of the form

        A * [beta_ddot, phi_ddot, psi_ddot] = b

        where A = A(phi, psi) and b=b(phi,psi,phi_dot,psi_dot,u)

        This function computes vector b.

        args:
            x (numpy.ndarray): current state

        Returns: 3x1 array of state and input terms
        """
        b = np.zeros(3)

        phi = x[1]
        psi = x[2]
        beta_dot = x[3]
        phi_dot = x[4]
        psi_dot = x[5]

        # auto-generated symbolic expressions
        b[0] = phi_dot**2 * self.p.l * self.p.m3 * self.p.r2 * sin(phi) + psi_dot**2 * self.p.l * self.p.m3 * self.p.r1 * sin(phi - psi) + psi_dot**2 * self.p.l * self.p.m3 * self.p.r2 * sin(phi - psi) - psi_dot**2 * self.p.m2 * self.p.r1 * self.p.r2 * sin(
            psi) - psi_dot**2 * self.p.m2 * self.p.r2**2 * sin(psi) - psi_dot**2 * self.p.m3 * self.p.r1 * self.p.r2 * sin(psi) - psi_dot**2 * self.p.m3 * self.p.r2**2 * sin(psi) - self.p.g * self.p.l * self.p.m3 * sin(phi)
        b[1] = -phi_dot**2 * self.p.l * self.p.m3 * self.p.r1 * sin(phi) - phi_dot**2 * self.p.l * self.p.m3 * self.p.r1 * sin(phi - psi) - phi_dot**2 * self.p.l * self.p.m3 * self.p.r2 * sin(phi) - phi_dot**2 * self.p.l * self.p.m3 * self.p.r2 * sin(phi - psi) + psi_dot**2 * self.p.m2 * self.p.r1**2 * sin(psi) + 2 * psi_dot**2 * self.p.m2 * self.p.r1 * self.p.r2 * sin(
            psi) + psi_dot**2 * self.p.m2 * self.p.r2**2 * sin(psi) + psi_dot**2 * self.p.m3 * self.p.r1**2 * sin(psi) + 2 * psi_dot**2 * self.p.m3 * self.p.r1 * self.p.r2 * sin(psi) + psi_dot**2 * self.p.m3 * self.p.r2**2 * sin(psi) + self.p.g * self.p.m2 * self.p.r1 * sin(psi) + self.p.g * self.p.m2 * self.p.r2 * sin(psi) + self.p.g * self.p.m3 * self.p.r1 * sin(psi) + self.p.g * self.p.m3 * self.p.r2 * sin(psi)
        b[2] = -(-beta_dot - omega_cmd + phi_dot) / self.p.tau

        return b

    def _compute_r_OSi(self, x):
        """computes center of mass locations of all bodies

        args:
            x (numpy.ndarray): current state

        Returns: list of x/y coordinates of center of mass of lower ball, upper ball, and lever arm.
        """
        beta = x[0]
        phi = x[1]
        psi = x[2]

        # upper ball on lower ball rolling constraint
        alpha = psi + (self.p.r2 / self.p.r1) * (psi - beta)

        # lower ball on ground rolling constraint
        x = -self.p.r1 * alpha

        r_OS1 = np.array([x, self.p.r1])

        r12 = self.p.r1 + self.p.r2
        r_S1S2 = np.array([-np.sin(psi) * r12, np.cos(psi) * r12])
        r_OS2 = r_OS1 + r_S1S2

        r_S2S3 = np.array([self.p.l * np.sin(phi), -self.p.l * np.cos(phi)])
        r_OS3 = r_OS2 + r_S2S3
        return [r_OS1, r_OS2, r_OS3]

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
