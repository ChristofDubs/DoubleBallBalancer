"""Dynamic model of 3D version of Double Ball Balancer

This module contains the class DynamicModel for simulating the non-linear dynamics of the 3D Double Ball Balancer and a corresponding class ModelParam containing the relevant parameters.

author: Christof Dubs
"""
import numpy as np
from numpy import sin, cos, tan
from scipy.integrate import odeint
from rotation import Quaternion


class ModelParam:
    """Physical parameters of 3D Double Ball Balancer

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
        self.theta3x = 1.0
        self.theta3y = 1.0
        self.theta3z = 1.0

    def is_valid(self,):
        """Checks validity of parameter configuration

        Returns:
            bool: True if valid, False if invalid.
        """
        return self.g > 0 and self.l > 0 and self.m1 > 0 and self.m2 > 0 and self.m3 > 0 and self.r1 > 0 and self.r2 > 0 and self.tau > 0 and self.theta1 > 0 and self.theta2 > 0 and self.theta3x > 0 and self.theta3y > 0 and self.theta3z > 0


# state size and indices
PSI_X_IDX = 0
PSI_Y_IDX = 1
Q_1_W_IDX = 2
Q_1_X_IDX = 3
Q_1_Y_IDX = 4
Q_1_Z_IDX = 5
Q_2_W_IDX = 6
Q_2_X_IDX = 7
Q_2_Y_IDX = 8
Q_2_Z_IDX = 9
Q_3_W_IDX = 10
Q_3_X_IDX = 11
Q_3_Y_IDX = 12
Q_3_Z_IDX = 13
OMEGA_1_Z_IDX = 14
PSI_X_DOT_IDX = 15
PSI_Y_DOT_IDX = 16
OMEGA_2_X_IDX = 17
OMEGA_2_Y_IDX = 18
OMEGA_2_Z_IDX = 19
OMEGA_3_X_IDX = 20
OMEGA_3_Y_IDX = 21
OMEGA_3_Z_IDX = 22
X_IDX = 23
Y_IDX = 24
STATE_SIZE = 25


class ModelState:
    """Class to represent state of 3D Double Ball Balancer

    The state is stored in this class as a numpy array, for ease of interfacing
    with scipy. Numerous properties / setters allow interfacing with this class
    without knowledge about the state index definitions.

    Attributes:
        x: array representing the full state
    """

    def __init__(self, x0=None, skip_checks=False):
        """Initializes attributes

        args:
            x0 (np.ndarray, optional): initial state. Set to default values if not specified
            skip_checks (bool, optional): if set to true and x0 is provided, x0 is set without checking it.
        """
        if skip_checks and x0 is not None:
            self.x = x0
            return

        if x0 is None or not self.set_state(x0):
            self.x = np.zeros(STATE_SIZE, dtype=np.float)
            self.x[Q_1_W_IDX] = 1
            self.x[Q_2_W_IDX] = 1
            self.x[Q_3_W_IDX] = 1

    def normalize_quaternions(self):
        """Normalize the rotation quaternions"""
        self.q1 *= 1.0 / np.linalg.norm(self.q1)
        self.q2 *= 1.0 / np.linalg.norm(self.q2)
        self.q3 *= 1.0 / np.linalg.norm(self.q3)

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

        q1_norm = np.linalg.norm(x0_flat[Q_1_W_IDX:Q_1_Z_IDX + 1])
        q2_norm = np.linalg.norm(x0_flat[Q_2_W_IDX:Q_2_Z_IDX + 1])
        q3_norm = np.linalg.norm(x0_flat[Q_3_W_IDX:Q_3_Z_IDX + 1])

        # quaternion check
        if q1_norm == 0 or q2_norm == 0 or q3_norm == 0:
            return false

        self.x = x0_flat
        self.normalize_quaternions()
        return True

    @property
    def psi_x(self):
        return self.x[PSI_X_IDX]

    @psi_x.setter
    def psi_x(self, value):
        self.x[PSI_X_IDX] = value

    @property
    def psi_x_dot(self):
        return self.x[PSI_X_DOT_IDX]

    @psi_x_dot.setter
    def psi_x_dot(self, value):
        self.x[PSI_X_DOT_IDX] = value

    @property
    def psi_y(self):
        return self.x[PSI_Y_IDX]

    @psi_y.setter
    def psi_y(self, value):
        self.x[PSI_Y_IDX] = value

    @property
    def psi_y_dot(self):
        return self.x[PSI_Y_DOT_IDX]

    @psi_y_dot.setter
    def psi_y_dot(self, value):
        self.x[PSI_Y_DOT_IDX] = value

    @property
    def q1(self):
        return self.x[Q_1_W_IDX:Q_1_Z_IDX + 1]

    @q1.setter
    def q1(self, value):
        if isinstance(value, Quaternion):
            self.x[Q_1_W_IDX:Q_1_Z_IDX + 1] = value.q
            return
        if isinstance(value, np.ndarray):
            self.x[Q_1_W_IDX:Q_1_Z_IDX + 1] = value
            return
        print('failed to set x')

    @property
    def q2(self):
        return self.x[Q_2_W_IDX:Q_2_Z_IDX + 1]

    @q2.setter
    def q2(self, value):
        if isinstance(value, Quaternion):
            self.x[Q_2_W_IDX:Q_2_Z_IDX + 1] = value.q
            return
        if isinstance(value, np.ndarray):
            self.x[Q_2_W_IDX:Q_2_Z_IDX + 1] = value
            return
        print('failed to set x')

    @property
    def q3(self):
        return self.x[Q_3_W_IDX:Q_3_Z_IDX + 1]

    @q3.setter
    def q3(self, value):
        if isinstance(value, Quaternion):
            self.x[Q_3_W_IDX:Q_3_Z_IDX + 1] = value.q
            return
        if isinstance(value, np.ndarray):
            self.x[Q_3_W_IDX:Q_3_Z_IDX + 1] = value
            return
        print('failed to set x')

    @property
    def psi(self):
        return self.x[PSI_X_IDX:PSI_Y_IDX + 1]

    @psi.setter
    def psi(self, value):
        self.x[PSI_X_IDX:PSI_Y_IDX + 1] = value

    @property
    def psi_dot(self):
        return self.x[PSI_X_DOT_IDX:PSI_Y_DOT_IDX + 1]

    @psi_dot.setter
    def psi_dot(self, value):
        self.x[PSI_X_DOT_IDX:PSI_Y_DOT_IDX + 1] = value

    @property
    def pos(self):
        return self.x[X_IDX:Y_IDX + 1]

    @pos.setter
    def pos(self, value):
        self.x[X_IDX:Y_IDX + 1] = value

    @property
    def omega(self):
        return self.x[OMEGA_1_Z_IDX:OMEGA_3_Z_IDX + 1]

    @omega.setter
    def omega(self, value):
        self.x[OMEGA_1_Z_IDX:OMEGA_3_Z_IDX + 1] = value

    @property
    def omega_1_z(self):
        return self.x[OMEGA_1_Z_IDX]

    @omega_1_z.setter
    def omega_1_z(self, value):
        self.x[OMEGA_1_Z_IDX] = value

    @property
    def omega_2(self):
        return self.x[OMEGA_2_X_IDX:OMEGA_2_Z_IDX + 1]

    @omega_2.setter
    def omega_2(self, value):
        self.x[OMEGA_2_X_IDX:OMEGA_2_Z_IDX + 1] = value

    @property
    def omega_3(self):
        return self.x[OMEGA_3_X_IDX:OMEGA_3_Z_IDX + 1]

    @omega_3.setter
    def omega_3(self, value):
        self.x[OMEGA_3_X_IDX:OMEGA_3_Z_IDX + 1] = value


class DynamicModel:
    """Simulation interface for the 2D Double Ball Balancer

    Attributes:
        p (ModelParam): physical parameters
        state (ModelState): 24-dimensional state

    Functions that are not meant to be called from outside the class (private methods) are prefixed with a single underline.
    """

    def __init__(self, param, x0=None):
        """Initializes attributes to default values

        args:
            param (ModelParam): parameters of type ModelParam
            x0 (ModelState, optional): initial state. Set to equilibrium state if not specified
        """
        self.p = param
        if not param.is_valid():
            print('Warning: not all parameters set!')

        if x0 is not None:
            if not isinstance(x0, ModelState):
                print('invalid type passed as initial state')
                self.state = ModelState()
            else:
                self.state = x0
        else:
            self.state = ModelState()

    def simulate_step(self, delta_t):
        """Simulate one time step

        Simulates the changes of the state over a time interval.

        args:
            delta_t: time step [s]
        """
        t = np.array([0, delta_t])
        self.state.x = odeint(self._x_dot, self.state.x, t)[-1]

        # normalize quaternions
        self.state.normalize_quaternions()

    def is_irrecoverable(self, state=None):
        """Checks if system is recoverable

        args:
            state (ModelState, optional): state. If not specified, the internal state is checked

        Returns:
            bool: True if state is irrecoverable, False otherwise.
        """
        if state is None:
            state = self.state

        arccos_psi = np.cos(state.psi_x) * np.cos(state.psi_y)

        # upper ball falling off the lower ball
        if arccos_psi < 0:
            return True

        # upper ball touching the ground
        if self.p.r2 > self.p.r1:
            arccos_psi_crit = (self.p.r2 - self.p.r1) / (self.p.r2 + self.p.r1)
            if arccos_psi < arccos_psi_crit:
                return True

        # todo: decide how to check / what to do if upper ball lifts off
        return False

    def get_visualization(self, state=None):
        """Get visualization of the system for plotting

        Usage example:
            v = model.get_visualization()
            plt.plot(*v['lower_ball'])

        args:
            state (ModelState, optional): state. If not specified, the internal state is checked

        Returns:
            dict: dictionary with keys "lower_ball", "upper_ball" and "lever_arm". The value for each key is a list with three elements: a list of x coordinates, a list of y coordinates and a list of z coordinates.
        """
        if state is None:
            state = self.state

        vis = {}

        r_OSi = self._compute_r_OSi(state)
        vis['lower_ball'] = self._compute_ball_visualization(
            r_OSi[0], self.p.r1, Quaternion(state.q1))
        vis['upper_ball'] = self._compute_ball_visualization(
            r_OSi[1], self.p.r2, Quaternion(state.q2))
        vis['lever_arm'] = [np.array([[r_OSi[1][i], r_OSi[2][i]]]) for i in range(3)]
        return vis

    def _x_dot(self, x, t):
        """computes the derivative of the state

        This function returns an numpy.array of the derivatives of the states, given the current state and inputs.

        Its signature is compatible with scipy.integrate.odeint's first callable argument.

        args:
            x (numpy.ndarray): state at which the state derivative function is evaluated
            t: time [s]. Since this system is time invariant, this argument is unused.
        """
        eval_state = ModelState(x, skip_checks=True)

        # freeze system if state is irrecoverable
        if self.is_irrecoverable(eval_state):
            return np.zeros(np.shape(eval_state.x))

        xdot = ModelState()

        A = self._compute_acc_jacobian_matrix(eval_state)
        b = self._compute_rhs(eval_state)
        omega_dot = np.linalg.solve(A, b)
        xdot.omega = omega_dot

        omega_1 = self._get_lower_ball_omega(eval_state)

        xdot.q1 = Quaternion(eval_state.q1).q_dot(omega_1, frame='inertial')

        xdot.q2 = Quaternion(eval_state.q2).q_dot(eval_state.omega_2, frame='inertial')

        xdot.q3 = Quaternion(eval_state.q3).q_dot(eval_state.omega_3, frame='body')

        xdot.psi = eval_state.psi_dot
        xdot.pos = self._get_lower_ball_vel(omega_1)

        return xdot.x

    def _get_lower_ball_vel(self, omega_1):
        """computes the linear velocity (x/y) of the lower ball

        args:
            omega_1 (numpy.ndarray): angular velocity [rad/s] of lower ball
        """
        return self.p.r1 * np.array([omega_1[1], -omega_1[0]])

    def _get_lower_ball_omega(self, state):
        """computes the angular velocity (x/y/z) of the lower ball

        args:
            state (ModelState): current state
        """
        psi_x = state.psi_x
        psi_y = state.psi_y
        psi_x_dot = state.psi_x_dot
        psi_y_dot = state.psi_y_dot
        w_1z = state.omega_1_z
        w_2x = state.omega_2[0]
        w_2y = state.omega_2[1]
        w_2z = state.omega_2[2]

        w_1x = (psi_x_dot * self.p.r1 + psi_x_dot * self.p.r2 + self.p.r1 * w_1z * sin(psi_y) - \
                self.p.r2 * w_2x * cos(psi_y) + self.p.r2 * w_2z * sin(psi_y)) / (self.p.r1 * cos(psi_y))
        w_1y = (-psi_x_dot * self.p.r1 * tan(psi_x) * tan(psi_y) - psi_x_dot * self.p.r2 * tan(psi_x) * tan(psi_y) + psi_y_dot * self.p.r1 + \
                psi_y_dot * self.p.r2 - self.p.r1 * w_1z * tan(psi_x) / cos(psi_y) - self.p.r2 * w_2y - self.p.r2 * w_2z * tan(psi_x) / cos(psi_y)) / self.p.r1

        return np.array([w_1x, w_1y, w_1z])

    def _compute_acc_jacobian_matrix(self, state):
        """computes angular acceleration matrix of rotational part of system dynamics (equal to jacobian matrix since dynamics are linear in angular accelerations)

        The non-linear rotational dynamics are of the form

        A * [omega_1_z_dot, psi_x_ddot, psi_y_ddot, omega_2_x_dot, omega_2_y_dot, omega_2_z_dot, omega_3_x_dot, omega_3_y_dot, omega_3_z_dot] = b

        where A = A(phi_x, phi_y, phi_z, psi_x, psi_y)

        This function computes matrix A.

        args:
            x (ModelState): current state

        Returns: 5x5 angular acceleration matrix
        """
        psi_x = state.psi_x
        psi_y = state.psi_y
        [phi_x, phi_y, phi_z] = Quaternion(state.q3).get_roll_pitch_yaw()

        A = np.zeros([9, 9])

        # auto-generated symbolic expressions
        A[0, 0] = -self.p.m1 * self.p.r1**2 + self.p.m1 * self.p.r1**2 / (cos(psi_x)**2 * cos(psi_y)**2) - self.p.m2 * self.p.r1**2 + self.p.m2 * self.p.r1**2 / (cos(
            psi_x)**2 * cos(psi_y)**2) - self.p.m3 * self.p.r1**2 + self.p.m3 * self.p.r1**2 / (cos(psi_x)**2 * cos(psi_y)**2) + self.p.theta1 / (cos(psi_x)**2 * cos(psi_y)**2)
        A[0, 1] = (self.p.m1 * self.p.r1**3 / cos(psi_x) + self.p.m1 * self.p.r1**2 * self.p.r2 / cos(psi_x) + self.p.m2 * self.p.r1**3 * cos(psi_y) + self.p.m2 * self.p.r1**3 / cos(psi_x) + self.p.m2 * self.p.r1**2 * self.p.r2 * cos(psi_y) + self.p.m2 * self.p.r1**2 * self.p.r2 / cos(psi_x) + self.p.m3 * self.p.r1 **
                   3 * cos(psi_y) + self.p.m3 * self.p.r1**3 / cos(psi_x) + self.p.m3 * self.p.r1**2 * self.p.r2 * cos(psi_y) + self.p.m3 * self.p.r1**2 * self.p.r2 / cos(psi_x) + self.p.r1 * self.p.theta1 / cos(psi_x) + self.p.r2 * self.p.theta1 / cos(psi_x)) * sin(psi_y) / (self.p.r1 * cos(psi_x) * cos(psi_y)**2)
        A[0, 2] = -(self.p.r1**2 * (self.p.m1 * (self.p.r1 + self.p.r2) + self.p.m2 * (self.p.r1 + self.p.r2 + (self.p.r1 + self.p.r2) * cos(psi_x) * cos(psi_y)) + self.p.m3 * \
                    (self.p.r1 + self.p.r2 + (self.p.r1 + self.p.r2) * cos(psi_x) * cos(psi_y))) + self.p.theta1 * (self.p.r1 + self.p.r2)) * tan(psi_x) / (self.p.r1 * cos(psi_y))
        A[0, 3] = -self.p.r2 * (self.p.r1**2 * (self.p.m1 + self.p.m2 +
                                                self.p.m3) + self.p.theta1) * tan(psi_y) / self.p.r1
        A[0, 4] = self.p.r2 * (self.p.r1**2 * (self.p.m1 + self.p.m2 + self.p.m3) +
                               self.p.theta1) * tan(psi_x) / (self.p.r1 * cos(psi_y))
        A[0, 5] = self.p.r2 * (-cos(psi_x)**2 * cos(psi_y)**2 + 1) * (self.p.m3 * self.p.r1**2 + self.p.r1**2 * (
            self.p.m1 + self.p.m2) + self.p.theta1) / (self.p.r1 * cos(psi_x)**2 * cos(psi_y)**2)
        A[0, 6] = -self.p.l * self.p.m3 * self.p.r1 * ((sin(phi_x) * sin(phi_y) * sin(phi_z) + cos(phi_x) * cos(phi_z)) * sin(
            psi_y) + (sin(phi_x) * sin(phi_y) * cos(phi_z) - sin(phi_z) * cos(phi_x)) * tan(psi_x)) / cos(psi_y)
        A[0, 7] = self.p.l * self.p.m3 * self.p.r1 * \
            (sin(phi_z) * tan(psi_y) + cos(phi_z) * tan(psi_x) / cos(psi_y)) * cos(phi_y)
        A[0, 8] = 0
        A[1, 0] = A[0, 1]
        A[1, 1] = (self.p.m3 * self.p.r1**2 * ((self.p.r1 + self.p.r2)**2 * (sin(psi_x) * sin(psi_y) + tan(psi_x) * tan(psi_y))**2 * cos(psi_y)**2 + (self.p.r1 + self.p.r2)**2 * sin(psi_x)**2 * cos(psi_y)**4 + (self.p.r1 + self.p.r2 + (self.p.r1 + self.p.r2) * cos(psi_x) * cos(psi_y))**2) + self.p.r1**2 * (self.p.r1 + self.p.r2)**2 * (self.p.m1 * sin(psi_y)
                                                                                                                                                                                                                                                                                                                                                 ** 2 * tan(psi_x)**2 + self.p.m1 - 2 * self.p.m2 * sin(psi_x)**2 * cos(psi_y)**3 / cos(psi_x) + self.p.m2 * sin(psi_y)**2 * tan(psi_x)**2 - self.p.m2 * sin(psi_y)**2 + 2 * self.p.m2 + 2 * self.p.m2 * cos(psi_y) / cos(psi_x)) + self.p.theta1 * (self.p.r1 + self.p.r2)**2 * (sin(psi_y)**2 * tan(psi_x)**2 + 1)) / (self.p.r1**2 * cos(psi_y)**2)
        A[1, 2] = -(self.p.r1 + self.p.r2)**2 * (self.p.m1 * self.p.r1**2 * tan(psi_x) * tan(psi_y) + 2 * self.p.m2 * self.p.r1**2 * sin(psi_x) * sin(psi_y) + self.p.m2 * self.p.r1**2 * tan(psi_x)
                                                 * tan(psi_y) + 2 * self.p.m3 * self.p.r1**2 * sin(psi_x) * sin(psi_y) + self.p.m3 * self.p.r1**2 * tan(psi_x) * tan(psi_y) + self.p.theta1 * tan(psi_x) * tan(psi_y)) / self.p.r1**2
        A[1, 3] = -self.p.r2 * (self.p.r1**2 * (self.p.m1 * (self.p.r1 + self.p.r2) + self.p.m2 * (self.p.r1 + self.p.r2 + (self.p.r1 + self.p.r2) * cos(psi_x) * cos(
            psi_y)) + self.p.m3 * (self.p.r1 + self.p.r2 + (self.p.r1 + self.p.r2) * cos(psi_x) * cos(psi_y))) + self.p.theta1 * (self.p.r1 + self.p.r2)) / (self.p.r1**2 * cos(psi_y))
        A[1, 4] = self.p.r2 * (self.p.r1**2 * (self.p.m1 * (self.p.r1 + self.p.r2) * tan(psi_x) * tan(psi_y) + self.p.m2 * (self.p.r1 * tan(psi_x) * tan(psi_y) + self.p.r2 * tan(psi_x) * tan(psi_y) + (self.p.r1 + self.p.r2) * sin(psi_x) * sin(
            psi_y)) + self.p.m3 * (self.p.r1 * tan(psi_x) * tan(psi_y) + self.p.r2 * tan(psi_x) * tan(psi_y) + (self.p.r1 + self.p.r2) * sin(psi_x) * sin(psi_y))) + self.p.theta1 * (self.p.r1 + self.p.r2) * tan(psi_x) * tan(psi_y)) / self.p.r1**2
        A[1, 5] = self.p.r2 * (self.p.m1 * self.p.r1**3 + self.p.m1 * self.p.r1**2 * self.p.r2 + self.p.m2 * self.p.r1**3 * cos(psi_x) * cos(psi_y) + self.p.m2 * self.p.r1**3 + self.p.m2 * self.p.r1**2 * self.p.r2 * cos(psi_x) * cos(psi_y) + self.p.m2 * self.p.r1**2 * self.p.r2 + self.p.m3 * \
                               self.p.r1**3 * cos(psi_x) * cos(psi_y) + self.p.m3 * self.p.r1**3 + self.p.m3 * self.p.r1**2 * self.p.r2 * cos(psi_x) * cos(psi_y) + self.p.m3 * self.p.r1**2 * self.p.r2 + self.p.r1 * self.p.theta1 + self.p.r2 * self.p.theta1) * sin(psi_y) / (self.p.r1**2 * cos(psi_x)**2 * cos(psi_y)**2)
        A[1, 6] = -self.p.l * self.p.m3 * ((self.p.r1 + self.p.r2) * sin(phi_x) * sin(psi_x) * cos(phi_y) * cos(psi_y)**2 + (sin(phi_x) * sin(phi_y) * sin(phi_z) + cos(phi_x) * cos(phi_z)) * (self.p.r1 + self.p.r2 + (self.p.r1 + self.p.r2) * cos(
            psi_x) * cos(psi_y)) + (sin(phi_x) * sin(phi_y) * cos(phi_z) - sin(phi_z) * cos(phi_x)) * (self.p.r1 * tan(psi_x) * tan(psi_y) + self.p.r2 * tan(psi_x) * tan(psi_y) + (self.p.r1 + self.p.r2) * sin(psi_x) * sin(psi_y)) * cos(psi_y)) / cos(psi_y)
        A[1, 7] = self.p.l * self.p.m3 * (-(self.p.r1 + self.p.r2) * sin(phi_y) * sin(psi_x) * cos(psi_y)**2 + (self.p.r1 + self.p.r2 + (self.p.r1 + self.p.r2) * cos(psi_x) * cos(psi_y)) * sin(phi_z) * cos(
            phi_y) + (self.p.r1 * tan(psi_x) * tan(psi_y) + self.p.r2 * tan(psi_x) * tan(psi_y) + (self.p.r1 + self.p.r2) * sin(psi_x) * sin(psi_y)) * cos(phi_y) * cos(phi_z) * cos(psi_y)) / cos(psi_y)
        A[1, 8] = 0
        A[2, 0] = A[0, 2]
        A[2, 1] = A[1, 2]
        A[2, 2] = (self.p.r1 + self.p.r2)**2 * (self.p.m1 * self.p.r1**2 + self.p.m2 * self.p.r1**2 * (cos(psi_x)**2 + 2 * cos(psi_x) * \
                   cos(psi_y) + 1) + self.p.m3 * self.p.r1**2 * (cos(psi_x)**2 + 2 * cos(psi_x) * cos(psi_y) + 1) + self.p.theta1) / self.p.r1**2
        A[2, 3] = 0
        A[2, 4] = -self.p.r2 * (self.p.r1**2 * (self.p.m1 * (self.p.r1 + self.p.r2) + self.p.m2 * (self.p.r1 + self.p.r2 + (self.p.r1 + self.p.r2) * cos(psi_x) * cos(
            psi_y)) + self.p.m3 * (self.p.r1 + self.p.r2 + (self.p.r1 + self.p.r2) * cos(psi_x) * cos(psi_y))) + self.p.theta1 * (self.p.r1 + self.p.r2)) / self.p.r1**2
        A[2, 5] = -self.p.r2 * (2 * self.p.r1**2 * (self.p.m1 * (self.p.r1 + self.p.r2) + self.p.m2 * (self.p.r1 + self.p.r2 + (self.p.r1 + self.p.r2) * cos(psi_x) * cos(psi_y)) + self.p.m3 * (self.p.r1 + self.p.r2 + (
            self.p.r1 + self.p.r2) * cos(psi_x) * cos(psi_y))) * cos(psi_x)**2 * tan(psi_x) + self.p.theta1 * (self.p.r1 + self.p.r2) * sin(2 * psi_x)) / (2 * self.p.r1**2 * cos(psi_x)**2 * cos(psi_y))
        A[2, 6] = -self.p.l * self.p.m3 * ((self.p.r1 + self.p.r2) * sin(phi_x) * sin(psi_y) * cos(phi_y) * cos(psi_x) - (sin(phi_x) * sin(
            phi_y) * cos(phi_z) - sin(phi_z) * cos(phi_x)) * (self.p.r1 + self.p.r2 + (self.p.r1 + self.p.r2) * cos(psi_x) * cos(psi_y)))
        A[2, 7] = -self.p.l * self.p.m3 * ((self.p.r1 + self.p.r2) * sin(phi_y) * sin(psi_y) * cos(psi_x) + (
            self.p.r1 + self.p.r2 + (self.p.r1 + self.p.r2) * cos(psi_x) * cos(psi_y)) * cos(phi_y) * cos(phi_z))
        A[2, 8] = 0
        A[3, 0] = A[0, 3]
        A[3, 1] = A[1, 3]
        A[3, 2] = A[2, 3]
        A[3, 3] = self.p.m1 * self.p.r2**2 + self.p.m2 * self.p.r2**2 + self.p.m3 * \
            self.p.r2**2 + self.p.theta2 + self.p.r2**2 * self.p.theta1 / self.p.r1**2
        A[3, 4] = 0
        A[3, 5] = -self.p.r2**2 * (self.p.m1 * self.p.r1**2 + self.p.m2 * self.p.r1 **
                                   2 + self.p.m3 * self.p.r1**2 + self.p.theta1) * tan(psi_y) / self.p.r1**2
        A[3, 6] = self.p.l * self.p.m3 * self.p.r2 * \
            (sin(phi_x) * sin(phi_y) * sin(phi_z) + cos(phi_x) * cos(phi_z))
        A[3, 7] = -self.p.l * self.p.m3 * self.p.r2 * sin(phi_z) * cos(phi_y)
        A[3, 8] = 0
        A[4, 0] = A[0, 4]
        A[4, 1] = A[1, 4]
        A[4, 2] = A[2, 4]
        A[4, 3] = A[3, 4]
        A[4, 4] = self.p.m1 * self.p.r2**2 + self.p.m2 * self.p.r2**2 + self.p.m3 * \
            self.p.r2**2 + self.p.theta2 + self.p.r2**2 * self.p.theta1 / self.p.r1**2
        A[4, 5] = self.p.r2**2 * (self.p.m1 * self.p.r1**2 + self.p.m2 * self.p.r1**2 + \
                                  self.p.m3 * self.p.r1**2 + self.p.theta1) * tan(psi_x) / (self.p.r1**2 * cos(psi_y))
        A[4, 6] = self.p.l * self.p.m3 * self.p.r2 * \
            (-sin(phi_x) * sin(phi_y) * cos(phi_z) + sin(phi_z) * cos(phi_x))
        A[4, 7] = self.p.l * self.p.m3 * self.p.r2 * cos(phi_y) * cos(phi_z)
        A[4, 8] = 0
        A[5, 0] = A[0, 5]
        A[5, 1] = A[1, 5]
        A[5, 2] = A[2, 5]
        A[5, 3] = A[3, 5]
        A[5, 4] = A[4, 5]
        A[5, 5] = -self.p.m1 * self.p.r2**2 + self.p.m1 * self.p.r2**2 / (cos(psi_x)**2 * cos(psi_y)**2) - self.p.m2 * self.p.r2**2 + self.p.m2 * self.p.r2**2 / (cos(psi_x)**2 * cos(psi_y)**2) - self.p.m3 * self.p.r2**2 + self.p.m3 * self.p.r2**2 / (
            cos(psi_x)**2 * cos(psi_y)**2) + self.p.theta2 - self.p.r2**2 * self.p.theta1 / self.p.r1**2 + self.p.r2**2 * self.p.theta1 / (self.p.r1**2 * cos(psi_x)**2 * cos(psi_y)**2)
        A[5, 6] = -self.p.l * self.p.m3 * self.p.r2 * ((sin(phi_x) * sin(phi_y) * sin(phi_z) + cos(phi_x) * cos(phi_z)) * sin(
            psi_y) + (sin(phi_x) * sin(phi_y) * cos(phi_z) - sin(phi_z) * cos(phi_x)) * tan(psi_x)) / cos(psi_y)
        A[5, 7] = self.p.l * self.p.m3 * self.p.r2 * \
            (sin(phi_z) * tan(psi_y) + cos(phi_z) * tan(psi_x) / cos(psi_y)) * cos(phi_y)
        A[5, 8] = 0
        A[6, 0] = A[0, 6]
        A[6, 1] = A[1, 6]
        A[6, 2] = A[2, 6]
        A[6, 3] = A[3, 6]
        A[6, 4] = A[4, 6]
        A[6, 5] = A[5, 6]
        A[6, 6] = self.p.l**2 * self.p.m3 + self.p.theta3x
        A[6, 7] = 0
        A[6, 8] = 0
        A[7, 0] = A[0, 7]
        A[7, 1] = A[1, 7]
        A[7, 2] = A[2, 7]
        A[7, 3] = A[3, 7]
        A[7, 4] = A[4, 7]
        A[7, 5] = A[5, 7]
        A[7, 6] = A[6, 7]
        A[7, 7] = self.p.l**2 * self.p.m3 + self.p.theta3y
        A[7, 8] = 0
        A[8, 0] = A[0, 8]
        A[8, 1] = A[1, 8]
        A[8, 2] = A[2, 8]
        A[8, 3] = A[3, 8]
        A[8, 4] = A[4, 8]
        A[8, 5] = A[5, 8]
        A[8, 6] = A[6, 8]
        A[8, 7] = A[7, 8]
        A[8, 8] = self.p.theta3z

        return A

    def _compute_rhs(self, state):
        """computes state (and input) terms of system dynamics

        The non-linear rotational dynamics are of the form

        A * [omega_1_z_dot, psi_x_ddot, psi_y_ddot, omega_2_x_dot, omega_2_y_dot, omega_2_z_dot, omega_3_x_dot, omega_3_y_dot, omega_3_z_dot] = b

        where A = A(phi_x, phi_y, phi_z, psi_x, psi_y)

        This function computes vector b.

        args:
            state (ModelState): current state

        Returns: 5x1 array of state (and input) terms
        """
        b = np.zeros(9)

        psi_x = state.psi_x
        psi_y = state.psi_y
        psi_x_dot = state.psi_x_dot
        psi_y_dot = state.psi_y_dot
        [phi_x, phi_y, phi_z] = Quaternion(state.q3).get_roll_pitch_yaw()
        w_1z = state.omega_1_z
        w_2x = state.omega_2[0]
        w_2y = state.omega_2[1]
        w_2z = state.omega_2[2]
        w_3x = state.omega_3[0]
        w_3y = state.omega_3[1]
        w_3z = state.omega_3[2]
        Tx = 0.0
        Ty = 0.0
        Tz = 0.0

        # auto-generated symbolic expressions
        b[0] = -(-self.p.m3 * self.p.r1**2 * (-(psi_x_dot * (self.p.r1 * w_1z + self.p.r2 * w_2z + (psi_x_dot * self.p.r1 * tan(psi_y) + psi_x_dot * self.p.r2 * tan(psi_y) + (self.p.r1 + self.p.r2) * (psi_x_dot * sin(psi_y) * cos(psi_x) + psi_y_dot * sin(psi_x) * cos(psi_y)) * cos(psi_x)**2) * cos(psi_y)) * cos(psi_y) + (psi_x_dot * psi_y_dot * self.p.r1 * sin(psi_x) * cos(psi_y)**3 + psi_x_dot * psi_y_dot * self.p.r1 * tan(psi_x) + psi_x_dot * psi_y_dot * self.p.r2 * sin(psi_x) * cos(psi_y)**3 + psi_x_dot * psi_y_dot * self.p.r2 * tan(psi_x) - psi_y_dot**2 * self.p.r1 * sin(psi_y)**3 * cos(psi_x) + psi_y_dot**2 * self.p.r1 * sin(psi_y) * cos(psi_x) - psi_y_dot**2 * self.p.r2 * sin(psi_y)**3 * cos(psi_x) + psi_y_dot**2 * self.p.r2 * sin(psi_y) * cos(psi_x) + psi_y_dot * self.p.r1 * w_1z * sin(psi_y) * tan(psi_x) + psi_y_dot * self.p.r2 * w_2z * sin(psi_y) * tan(psi_x) - self.p.l * w_3x**2 * sin(phi_x) * sin(phi_z) * cos(psi_y)**2 - self.p.l * w_3x**2 * sin(phi_y) * cos(phi_x) * cos(phi_z) * cos(psi_y)**2 + self.p.l * w_3x * w_3z * cos(phi_y) * cos(phi_z) * cos(psi_y)**2 - self.p.l * w_3y**2 * sin(phi_x) * sin(phi_z) * cos(psi_y)**2 - self.p.l * w_3y**2 * sin(phi_y) * cos(phi_x) * cos(phi_z) * cos(psi_y)**2 + self.p.l * w_3y * w_3z * sin(phi_x) * sin(phi_y) * cos(phi_z) * cos(psi_y)**2 - self.p.l * w_3y * w_3z * sin(phi_z) * cos(phi_x) * cos(psi_y)**2) * cos(psi_x)**2) * cos(phi_y) * tan(psi_x) + (-psi_y_dot * (psi_x_dot * self.p.r1 * sin(psi_y) + psi_x_dot * self.p.r2 * sin(psi_y) + self.p.r1 * w_1z + self.p.r2 * w_2z) * cos(phi_y) + self.p.l * (w_3x * (sin(phi_x) * sin(phi_y) * cos(phi_z) - sin(phi_z) * cos(phi_x)) - w_3y * cos(phi_y) * cos(phi_z)) * (w_3y * sin(phi_x) + w_3z * cos(phi_x)) * cos(psi_y)**2 + (psi_x_dot**2 * (self.p.r1 + self.p.r2) * sin(psi_x) - self.p.l * w_3x * (sin(phi_x) * cos(phi_z) - sin(phi_y) * sin(phi_z) * cos(phi_x)) * (w_3x + w_3y * sin(phi_x) * tan(phi_y) + w_3z * cos(phi_x) * tan(phi_y)) + self.p.l * (w_3y * cos(phi_x) - w_3z * sin(phi_x)) * (w_3x * sin(phi_x) * cos(phi_y) + w_3y * sin(phi_y)) * sin(phi_z)) * cos(phi_y) * cos(psi_y)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            ** 2) * sin(psi_y) * cos(psi_x)**2) + self.p.r1**2 * (self.p.m1 * (psi_y_dot * (psi_x_dot * self.p.r1 * sin(psi_y) + psi_x_dot * self.p.r2 * sin(psi_y) + self.p.r1 * w_1z + self.p.r2 * w_2z) * sin(psi_y) * cos(psi_x)**2 + (psi_x_dot * (psi_x_dot * (self.p.r1 + self.p.r2) * sin(psi_y) + self.p.r1 * w_1z + self.p.r2 * w_2z) * cos(psi_y) + psi_y_dot * (psi_x_dot * self.p.r1 + psi_x_dot * self.p.r2 + self.p.r1 * w_1z * sin(psi_y) + self.p.r2 * w_2z * sin(psi_y)) * sin(psi_x) * cos(psi_x)) * tan(psi_x)) + self.p.m2 * ((psi_x_dot * (self.p.r1 * w_1z + self.p.r2 * w_2z + (psi_x_dot * self.p.r1 * tan(psi_y) + psi_x_dot * self.p.r2 * tan(psi_y) + (self.p.r1 + self.p.r2) * (psi_x_dot * sin(psi_y) * cos(psi_x) + psi_y_dot * sin(psi_x) * cos(psi_y)) * cos(psi_x)**2) * cos(psi_y)) * cos(psi_y) + psi_y_dot * (psi_x_dot * self.p.r1 * sin(psi_x) * cos(psi_y)**3 + psi_x_dot * self.p.r1 * tan(psi_x) + psi_x_dot * self.p.r2 * sin(psi_x) * cos(psi_y)**3 + psi_x_dot * self.p.r2 * tan(psi_x) - psi_y_dot * self.p.r1 * sin(psi_y)**3 * cos(psi_x) + psi_y_dot * self.p.r1 * sin(psi_y) * cos(psi_x) - psi_y_dot * self.p.r2 * sin(psi_y)**3 * cos(psi_x) + psi_y_dot * self.p.r2 * sin(psi_y) * cos(psi_x) + self.p.r1 * w_1z * sin(psi_y) * tan(psi_x) + self.p.r2 * w_2z * sin(psi_y) * tan(psi_x)) * cos(psi_x)**2) * tan(psi_x) - (psi_x_dot**2 * (self.p.r1 + self.p.r2) * sin(psi_x) * cos(psi_y)**2 - psi_y_dot * (psi_x_dot * self.p.r1 * sin(psi_y) + psi_x_dot * self.p.r2 * sin(psi_y) + self.p.r1 * w_1z + self.p.r2 * w_2z)) * sin(psi_y) * cos(psi_x)**2)) * cos(phi_y) + self.p.theta1 * (psi_y_dot * (psi_x_dot * self.p.r1 * sin(psi_y) + psi_x_dot * self.p.r2 * sin(psi_y) + self.p.r1 * w_1z + self.p.r2 * w_2z) * sin(psi_y) * cos(psi_x)**2 + (psi_x_dot * (psi_x_dot * (self.p.r1 + self.p.r2) * sin(psi_y) + self.p.r1 * w_1z + self.p.r2 * w_2z) * cos(psi_y) + psi_y_dot * (psi_x_dot * self.p.r1 + psi_x_dot * self.p.r2 + self.p.r1 * w_1z * sin(psi_y) + self.p.r2 * w_2z * sin(psi_y)) * sin(psi_x) * cos(psi_x)) * tan(psi_x)) * cos(phi_y)) / (self.p.r1 * cos(phi_y) * cos(psi_x)**2 * cos(psi_y)**3)
        b[1] = -(-self.p.m3 * self.p.r1**2 * (-(psi_x_dot * (self.p.r1 * w_1z + self.p.r2 * w_2z + (psi_x_dot * self.p.r1 * tan(psi_y) + psi_x_dot * self.p.r2 * tan(psi_y) + (self.p.r1 + self.p.r2) * (psi_x_dot * sin(psi_y) * cos(psi_x) + psi_y_dot * sin(psi_x) * cos(psi_y)) * cos(psi_x)**2) * cos(psi_y)) * cos(psi_y) + (psi_x_dot * psi_y_dot * self.p.r1 * sin(psi_x) * cos(psi_y)**3 + psi_x_dot * psi_y_dot * self.p.r1 * tan(psi_x) + psi_x_dot * psi_y_dot * self.p.r2 * sin(psi_x) * cos(psi_y)**3 + psi_x_dot * psi_y_dot * self.p.r2 * tan(psi_x) - psi_y_dot**2 * self.p.r1 * sin(psi_y)**3 * cos(psi_x) + psi_y_dot**2 * self.p.r1 * sin(psi_y) * cos(psi_x) - psi_y_dot**2 * self.p.r2 * sin(psi_y)**3 * cos(psi_x) + psi_y_dot**2 * self.p.r2 * sin(psi_y) * cos(psi_x) + psi_y_dot * self.p.r1 * w_1z * sin(psi_y) * tan(psi_x) + psi_y_dot * self.p.r2 * w_2z * sin(psi_y) * tan(psi_x) - self.p.l * w_3x**2 * sin(phi_x) * sin(phi_z) * cos(psi_y)**2 - self.p.l * w_3x**2 * sin(phi_y) * cos(phi_x) * cos(phi_z) * cos(psi_y)**2 + self.p.l * w_3x * w_3z * cos(phi_y) * cos(phi_z) * cos(psi_y)**2 - self.p.l * w_3y**2 * sin(phi_x) * sin(phi_z) * cos(psi_y)**2 - self.p.l * w_3y**2 * sin(phi_y) * cos(phi_x) * cos(phi_z) * cos(psi_y)**2 + self.p.l * w_3y * w_3z * sin(phi_x) * sin(phi_y) * cos(phi_z) * cos(psi_y)**2 - self.p.l * w_3y * w_3z * sin(phi_z) * cos(phi_x) * cos(psi_y)**2) * cos(psi_x)**2) * (self.p.r1 * tan(psi_x) * tan(psi_y) + self.p.r2 * tan(psi_x) * tan(psi_y) + (self.p.r1 + self.p.r2) * sin(psi_x) * sin(psi_y)) * cos(phi_y) * cos(psi_y) + ((self.p.r1 + self.p.r2) * (-psi_x_dot * (self.p.r1 + self.p.r2) * (psi_x_dot * cos(psi_x) * cos(psi_y) - psi_y_dot * sin(psi_x) * sin(psi_y)) + psi_y_dot * (self.p.r1 + self.p.r2) * (psi_x_dot * sin(psi_x) * sin(psi_y) - psi_y_dot * cos(psi_x) * cos(psi_y)) + self.p.g + self.p.l * w_3x * (w_3x + w_3y * sin(phi_x) * tan(phi_y) + w_3z * cos(phi_x) * tan(phi_y)) * cos(phi_x) * cos(phi_y) - self.p.l * (w_3y * cos(phi_x) - w_3z * sin(phi_x)) * (w_3x * sin(phi_x) * sin(phi_y) - w_3y * cos(phi_y))) * sin(psi_x) * cos(phi_y) * cos(psi_y)**4 + (self.p.r1 + self.p.r2 + (self.p.r1 + self.p.r2) * cos(psi_x) * cos(psi_y)) * (-psi_y_dot * (psi_x_dot * self.p.r1 * sin(psi_y) + psi_x_dot * self.p.r2 * sin(psi_y) + self.p.r1 * w_1z + self.p.r2 * w_2z) * cos(phi_y) + self.p.l * (w_3x * (sin(phi_x) * sin(phi_y) * cos(phi_z) - sin(phi_z) * cos(phi_x)) - w_3y * cos(phi_y) * cos(phi_z)) * (w_3y * sin(phi_x) + w_3z * cos(phi_x)) * cos(psi_y)**2 + (psi_x_dot**2 * (self.p.r1 + self.p.r2) * sin(psi_x) - self.p.l * w_3x * (sin(phi_x) * cos(phi_z) - sin(phi_y) * sin(phi_z) * cos(phi_x)) * (w_3x + w_3y * sin(phi_x) * tan(phi_y) + w_3z * cos(phi_x) * tan(phi_y)) + self.p.l * (w_3y * cos(phi_x) - w_3z * sin(phi_x))
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         * (w_3x * sin(phi_x) * cos(phi_y) + w_3y * sin(phi_y)) * sin(phi_z)) * cos(phi_y) * cos(psi_y)**2)) * cos(psi_x)**2) + self.p.r1**2 * (self.p.m1 * (self.p.r1 + self.p.r2) * (psi_y_dot * (psi_x_dot * self.p.r1 * sin(psi_y) + psi_x_dot * self.p.r2 * sin(psi_y) + self.p.r1 * w_1z + self.p.r2 * w_2z) * cos(psi_x)**2 + (psi_x_dot * (psi_x_dot * (self.p.r1 + self.p.r2) * sin(psi_y) + self.p.r1 * w_1z + self.p.r2 * w_2z) * cos(psi_y) + psi_y_dot * (psi_x_dot * self.p.r1 + psi_x_dot * self.p.r2 + self.p.r1 * w_1z * sin(psi_y) + self.p.r2 * w_2z * sin(psi_y)) * sin(psi_x) * cos(psi_x)) * cos(psi_y) * tan(psi_x) * tan(psi_y)) - self.p.m2 * (-(psi_x_dot * (self.p.r1 * w_1z + self.p.r2 * w_2z + (psi_x_dot * self.p.r1 * tan(psi_y) + psi_x_dot * self.p.r2 * tan(psi_y) + (self.p.r1 + self.p.r2) * (psi_x_dot * sin(psi_y) * cos(psi_x) + psi_y_dot * sin(psi_x) * cos(psi_y)) * cos(psi_x)**2) * cos(psi_y)) * cos(psi_y) + psi_y_dot * (psi_x_dot * self.p.r1 * sin(psi_x) * cos(psi_y)**3 + psi_x_dot * self.p.r1 * tan(psi_x) + psi_x_dot * self.p.r2 * sin(psi_x) * cos(psi_y)**3 + psi_x_dot * self.p.r2 * tan(psi_x) - psi_y_dot * self.p.r1 * sin(psi_y)**3 * cos(psi_x) + psi_y_dot * self.p.r1 * sin(psi_y) * cos(psi_x) - psi_y_dot * self.p.r2 * sin(psi_y)**3 * cos(psi_x) + psi_y_dot * self.p.r2 * sin(psi_y) * cos(psi_x) + self.p.r1 * w_1z * sin(psi_y) * tan(psi_x) + self.p.r2 * w_2z * sin(psi_y) * tan(psi_x)) * cos(psi_x)**2) * (self.p.r1 * tan(psi_x) * tan(psi_y) + self.p.r2 * tan(psi_x) * tan(psi_y) + (self.p.r1 + self.p.r2) * sin(psi_x) * sin(psi_y)) * cos(psi_y) + ((self.p.r1 + self.p.r2) * (-psi_x_dot * (self.p.r1 + self.p.r2) * (psi_x_dot * cos(psi_x) * cos(psi_y) - psi_y_dot * sin(psi_x) * sin(psi_y)) + psi_y_dot * (self.p.r1 + self.p.r2) * (psi_x_dot * sin(psi_x) * sin(psi_y) - psi_y_dot * cos(psi_x) * cos(psi_y)) + self.p.g) * sin(psi_x) * cos(psi_y)**4 + (psi_x_dot**2 * (self.p.r1 + self.p.r2) * sin(psi_x) * cos(psi_y)**2 - psi_y_dot * (psi_x_dot * self.p.r1 * sin(psi_y) + psi_x_dot * self.p.r2 * sin(psi_y) + self.p.r1 * w_1z + self.p.r2 * w_2z)) * (self.p.r1 + self.p.r2 + (self.p.r1 + self.p.r2) * cos(psi_x) * cos(psi_y))) * cos(psi_x)**2)) * cos(phi_y) + self.p.theta1 * (self.p.r1 + self.p.r2) * (psi_y_dot * (psi_x_dot * self.p.r1 * sin(psi_y) + psi_x_dot * self.p.r2 * sin(psi_y) + self.p.r1 * w_1z + self.p.r2 * w_2z) * cos(psi_x)**2 + (psi_x_dot * (psi_x_dot * (self.p.r1 + self.p.r2) * sin(psi_y) + self.p.r1 * w_1z + self.p.r2 * w_2z) * cos(psi_y) + psi_y_dot * (psi_x_dot * self.p.r1 + psi_x_dot * self.p.r2 + self.p.r1 * w_1z * sin(psi_y) + self.p.r2 * w_2z * sin(psi_y)) * sin(psi_x) * cos(psi_x)) * cos(psi_y) * tan(psi_x) * tan(psi_y)) * cos(phi_y)) / (self.p.r1**2 * cos(phi_y) * cos(psi_x)**2 * cos(psi_y)**3)
        b[2] = (self.p.m3 * self.p.r1**2 * ((self.p.r1 + self.p.r2) * (-psi_x_dot * (self.p.r1 + self.p.r2) * (psi_x_dot * cos(psi_x) * cos(psi_y) - psi_y_dot * sin(psi_x) * sin(psi_y)) + psi_y_dot * (self.p.r1 + self.p.r2) * (psi_x_dot * sin(psi_x) * sin(psi_y) - psi_y_dot * cos(psi_x) * cos(psi_y)) + self.p.g + self.p.l * w_3x * (w_3x + w_3y * sin(phi_x) * tan(phi_y) + w_3z * cos(phi_x) * tan(phi_y)) * cos(phi_x) * cos(phi_y) - self.p.l * (w_3y * cos(phi_x) - w_3z * sin(phi_x)) * (w_3x * sin(phi_x) * sin(phi_y) - w_3y * cos(phi_y))) * sin(psi_y) * cos(psi_x)**3 * cos(psi_y)**2 + (psi_x_dot * (self.p.r1 * w_1z + self.p.r2 * w_2z + (psi_x_dot * self.p.r1 * tan(psi_y) + psi_x_dot * self.p.r2 * tan(psi_y) + (self.p.r1 + self.p.r2) * (psi_x_dot * sin(psi_y) * cos(psi_x) + psi_y_dot * sin(psi_x) * cos(psi_y)) * cos(psi_x)**2) * cos(psi_y)) * cos(psi_y) + (psi_x_dot * psi_y_dot * self.p.r1 * sin(psi_x) * cos(psi_y)**3 + psi_x_dot * psi_y_dot * self.p.r1 * tan(psi_x) + psi_x_dot * psi_y_dot * self.p.r2 * sin(psi_x) * cos(psi_y)**3 + psi_x_dot * psi_y_dot * self.p.r2 * tan(psi_x) - psi_y_dot**2 * self.p.r1 * sin(psi_y)**3 * cos(psi_x) + psi_y_dot**2 * self.p.r1 * sin(psi_y) * cos(psi_x) - psi_y_dot**2 * self.p.r2 * sin(psi_y)**3 * cos(psi_x) + psi_y_dot**2 * self.p.r2 * sin(psi_y) * cos(psi_x) + psi_y_dot * self.p.r1 * w_1z * sin(psi_y) * tan(psi_x) + psi_y_dot * self.p.r2 * w_2z * sin(psi_y) * tan(psi_x) - self.p.l * w_3x**2 * sin(phi_x) * sin(phi_z) * cos(psi_y)**2 - self.p.l * w_3x**2 * sin(phi_y) * cos(phi_x) * cos(phi_z) * cos(psi_y)**2 + self.p.l * w_3x * w_3z * cos(phi_y) * cos(phi_z) * cos(psi_y)**2 - self.p.l * w_3y**2 * sin(phi_x) * sin(phi_z) * cos(psi_y)**2 - self.p.l * w_3y**2 * sin(phi_y) * cos(phi_x) * cos(phi_z) * cos(psi_y)**2 + self.p.l * w_3y * w_3z * sin(phi_x) * sin(phi_y) * cos(phi_z) * cos(psi_y)**2 - self.p.l * w_3y * w_3z * sin(phi_z) * cos(phi_x) * cos(psi_y)**2) * cos(psi_x)**2) * (self.p.r1 + \
                self.p.r2 + (self.p.r1 + self.p.r2) * cos(psi_x) * cos(psi_y))) + self.p.r1**2 * (self.p.m1 * (self.p.r1 + self.p.r2) * (psi_x_dot * (psi_x_dot * (self.p.r1 + self.p.r2) * sin(psi_y) + self.p.r1 * w_1z + self.p.r2 * w_2z) * cos(psi_y) + psi_y_dot * (psi_x_dot * self.p.r1 + psi_x_dot * self.p.r2 + self.p.r1 * w_1z * sin(psi_y) + self.p.r2 * w_2z * sin(psi_y)) * sin(psi_x) * cos(psi_x)) + self.p.m2 * ((self.p.r1 + self.p.r2) * (-psi_x_dot * (self.p.r1 + self.p.r2) * (psi_x_dot * cos(psi_x) * cos(psi_y) - psi_y_dot * sin(psi_x) * sin(psi_y)) + psi_y_dot * (self.p.r1 + self.p.r2) * (psi_x_dot * sin(psi_x) * sin(psi_y) - psi_y_dot * cos(psi_x) * cos(psi_y)) + self.p.g) * sin(psi_y) * cos(psi_x)**3 * cos(psi_y)**2 + (psi_x_dot * (self.p.r1 * w_1z + self.p.r2 * w_2z + (psi_x_dot * self.p.r1 * tan(psi_y) + psi_x_dot * self.p.r2 * tan(psi_y) + (self.p.r1 + self.p.r2) * (psi_x_dot * sin(psi_y) * cos(psi_x) + psi_y_dot * sin(psi_x) * cos(psi_y)) * cos(psi_x)**2) * cos(psi_y)) * cos(psi_y) + psi_y_dot * (psi_x_dot * self.p.r1 * sin(psi_x) * cos(psi_y)**3 + psi_x_dot * self.p.r1 * tan(psi_x) + psi_x_dot * self.p.r2 * sin(psi_x) * cos(psi_y)**3 + psi_x_dot * self.p.r2 * tan(psi_x) - psi_y_dot * self.p.r1 * sin(psi_y)**3 * cos(psi_x) + psi_y_dot * self.p.r1 * sin(psi_y) * cos(psi_x) - psi_y_dot * self.p.r2 * sin(psi_y)**3 * cos(psi_x) + psi_y_dot * self.p.r2 * sin(psi_y) * cos(psi_x) + self.p.r1 * w_1z * sin(psi_y) * tan(psi_x) + self.p.r2 * w_2z * sin(psi_y) * tan(psi_x)) * cos(psi_x)**2) * (self.p.r1 + self.p.r2 + (self.p.r1 + self.p.r2) * cos(psi_x) * cos(psi_y)))) + self.p.theta1 * (self.p.r1 + self.p.r2) * (psi_x_dot * (psi_x_dot * (self.p.r1 + self.p.r2) * sin(psi_y) + self.p.r1 * w_1z + self.p.r2 * w_2z) * cos(psi_y) + psi_y_dot * (psi_x_dot * self.p.r1 + psi_x_dot * self.p.r2 + self.p.r1 * w_1z * sin(psi_y) + self.p.r2 * w_2z * sin(psi_y)) * sin(psi_x) * cos(psi_x))) / (self.p.r1**2 * cos(psi_x)**2 * cos(psi_y)**2)
        b[3] = -(-psi_y_dot * self.p.r2 * self.p.theta1 * (psi_x_dot * self.p.r1 * sin(psi_y) + psi_x_dot * self.p.r2 * sin(psi_y) + self.p.r1 * w_1z + self.p.r2 * w_2z) + self.p.r1**2 * self.p.r2 * (-psi_y_dot * self.p.m1 * (psi_x_dot * self.p.r1 * sin(psi_y) + psi_x_dot * self.p.r2 * sin(psi_y) + self.p.r1 * w_1z + self.p.r2 * w_2z) + self.p.m2 * (psi_x_dot**2 * (self.p.r1 + self.p.r2) * sin(psi_x) * cos(psi_y)**2 - psi_y_dot * (psi_x_dot * self.p.r1 * sin(psi_y) + psi_x_dot * self.p.r2 * sin(psi_y) + self.p.r1 * w_1z + self.p.r2 * w_2z)) - self.p.m3 * (psi_x_dot * psi_y_dot * self.p.r1 * sin(psi_y) + psi_x_dot * psi_y_dot * self.p.r2 * sin(psi_y) + psi_y_dot * self.p.r1 * w_1z + psi_y_dot * self.p.r2 * w_2z + \
                 (-psi_x_dot**2 * self.p.r1 * sin(psi_x) - psi_x_dot**2 * self.p.r2 * sin(psi_x) + self.p.l * w_3x**2 * sin(phi_x) * cos(phi_z) - self.p.l * w_3x**2 * sin(phi_y) * sin(phi_z) * cos(phi_x) + self.p.l * w_3x * w_3z * sin(phi_z) * cos(phi_y) + self.p.l * w_3y**2 * sin(phi_x) * cos(phi_z) - self.p.l * w_3y**2 * sin(phi_y) * sin(phi_z) * cos(phi_x) + self.p.l * w_3y * w_3z * sin(phi_x) * sin(phi_y) * sin(phi_z) + self.p.l * w_3y * w_3z * cos(phi_x) * cos(phi_z)) * cos(psi_y)**2)) + self.p.r1**2 * (Tx * cos(phi_y) * cos(phi_z) + Ty * (sin(phi_x) * sin(phi_y) * cos(phi_z) - sin(phi_z) * cos(phi_x)) + Tz * (sin(phi_x) * sin(phi_z) + sin(phi_y) * cos(phi_x) * cos(phi_z))) * cos(psi_y)**2) / (self.p.r1**2 * cos(psi_y)**2)
        b[4] = -(self.p.m3 * self.p.r1**2 * self.p.r2 * (psi_x_dot * (self.p.r1 * w_1z + self.p.r2 * w_2z + (psi_x_dot * self.p.r1 * tan(psi_y) + psi_x_dot * self.p.r2 * tan(psi_y) + (self.p.r1 + self.p.r2) * (psi_x_dot * sin(psi_y) * cos(psi_x) + psi_y_dot * sin(psi_x) * cos(psi_y)) * cos(psi_x)**2) * cos(psi_y)) * cos(psi_y) + (psi_x_dot * psi_y_dot * self.p.r1 * sin(psi_x) * cos(psi_y)**3 + psi_x_dot * psi_y_dot * self.p.r1 * tan(psi_x) + psi_x_dot * psi_y_dot * self.p.r2 * sin(psi_x) * cos(psi_y)**3 + psi_x_dot * psi_y_dot * self.p.r2 * tan(psi_x) - psi_y_dot**2 * self.p.r1 * sin(psi_y)**3 * cos(psi_x) + psi_y_dot**2 * self.p.r1 * sin(psi_y) * cos(psi_x) - psi_y_dot**2 * self.p.r2 * sin(psi_y)**3 * cos(psi_x) + psi_y_dot**2 * self.p.r2 * sin(psi_y) * cos(psi_x) + psi_y_dot * self.p.r1 * w_1z * sin(psi_y) * tan(psi_x) + psi_y_dot * self.p.r2 * w_2z * sin(psi_y) * tan(psi_x) - self.p.l * w_3x**2 * sin(phi_x) * sin(phi_z) * cos(psi_y)**2 - self.p.l * w_3x**2 * sin(phi_y) * cos(phi_x) * cos(phi_z) * cos(psi_y)**2 + self.p.l * w_3x * w_3z * cos(phi_y) * cos(phi_z) * cos(psi_y)**2 - self.p.l * w_3y**2 * sin(phi_x) * sin(phi_z) * cos(psi_y)**2 - self.p.l * w_3y**2 * sin(phi_y) * cos(phi_x) * cos(phi_z) * cos(psi_y)**2 + self.p.l * w_3y * w_3z * sin(phi_x) * sin(phi_y) * cos(phi_z) * cos(psi_y)**2 - self.p.l * w_3y * w_3z * sin(phi_z) * cos(phi_x) * cos(psi_y)**2) * cos(psi_x)**2) + self.p.r1**2 * self.p.r2 * (self.p.m1 * (psi_x_dot * (psi_x_dot * (self.p.r1 + self.p.r2) * sin(psi_y) + self.p.r1 * w_1z + self.p.r2 * w_2z)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   * cos(psi_y) + psi_y_dot * (psi_x_dot * self.p.r1 + psi_x_dot * self.p.r2 + self.p.r1 * w_1z * sin(psi_y) + self.p.r2 * w_2z * sin(psi_y)) * sin(psi_x) * cos(psi_x)) + self.p.m2 * (psi_x_dot * (self.p.r1 * w_1z + self.p.r2 * w_2z + (psi_x_dot * self.p.r1 * tan(psi_y) + psi_x_dot * self.p.r2 * tan(psi_y) + (self.p.r1 + self.p.r2) * (psi_x_dot * sin(psi_y) * cos(psi_x) + psi_y_dot * sin(psi_x) * cos(psi_y)) * cos(psi_x)**2) * cos(psi_y)) * cos(psi_y) + psi_y_dot * (psi_x_dot * self.p.r1 * sin(psi_x) * cos(psi_y)**3 + psi_x_dot * self.p.r1 * tan(psi_x) + psi_x_dot * self.p.r2 * sin(psi_x) * cos(psi_y)**3 + psi_x_dot * self.p.r2 * tan(psi_x) - psi_y_dot * self.p.r1 * sin(psi_y)**3 * cos(psi_x) + psi_y_dot * self.p.r1 * sin(psi_y) * cos(psi_x) - psi_y_dot * self.p.r2 * sin(psi_y)**3 * cos(psi_x) + psi_y_dot * self.p.r2 * sin(psi_y) * cos(psi_x) + self.p.r1 * w_1z * sin(psi_y) * tan(psi_x) + self.p.r2 * w_2z * sin(psi_y) * tan(psi_x)) * cos(psi_x)**2)) + self.p.r1**2 * (Tx * sin(phi_z) * cos(phi_y) + Ty * (sin(phi_x) * sin(phi_y) * sin(phi_z) + cos(phi_x) * cos(phi_z)) - Tz * (sin(phi_x) * cos(phi_z) - sin(phi_y) * sin(phi_z) * cos(phi_x))) * cos(psi_x)**2 * cos(psi_y)**2 + self.p.r2 * self.p.theta1 * (psi_x_dot * (psi_x_dot * (self.p.r1 + self.p.r2) * sin(psi_y) + self.p.r1 * w_1z + self.p.r2 * w_2z) * cos(psi_y) + psi_y_dot * (psi_x_dot * self.p.r1 + psi_x_dot * self.p.r2 + self.p.r1 * w_1z * sin(psi_y) + self.p.r2 * w_2z * sin(psi_y)) * sin(psi_x) * cos(psi_x))) / (self.p.r1**2 * cos(psi_x)**2 * cos(psi_y)**2)
        b[5] = -(-self.p.m3 * self.p.r1**2 * self.p.r2 * (-(psi_x_dot * (self.p.r1 * w_1z + self.p.r2 * w_2z + (psi_x_dot * self.p.r1 * tan(psi_y) + psi_x_dot * self.p.r2 * tan(psi_y) + (self.p.r1 + self.p.r2) * (psi_x_dot * sin(psi_y) * cos(psi_x) + psi_y_dot * sin(psi_x) * cos(psi_y)) * cos(psi_x)**2) * cos(psi_y)) * cos(psi_y) + (psi_x_dot * psi_y_dot * self.p.r1 * sin(psi_x) * cos(psi_y)**3 + psi_x_dot * psi_y_dot * self.p.r1 * tan(psi_x) + psi_x_dot * psi_y_dot * self.p.r2 * sin(psi_x) * cos(psi_y)**3 + psi_x_dot * psi_y_dot * self.p.r2 * tan(psi_x) - psi_y_dot**2 * self.p.r1 * sin(psi_y)**3 * cos(psi_x) + psi_y_dot**2 * self.p.r1 * sin(psi_y) * cos(psi_x) - psi_y_dot**2 * self.p.r2 * sin(psi_y)**3 * cos(psi_x) + psi_y_dot**2 * self.p.r2 * sin(psi_y) * cos(psi_x) + psi_y_dot * self.p.r1 * w_1z * sin(psi_y) * tan(psi_x) + psi_y_dot * self.p.r2 * w_2z * sin(psi_y) * tan(psi_x) - self.p.l * w_3x**2 * sin(phi_x) * sin(phi_z) * cos(psi_y)**2 - self.p.l * w_3x**2 * sin(phi_y) * cos(phi_x) * cos(phi_z) * cos(psi_y)**2 + self.p.l * w_3x * w_3z * cos(phi_y) * cos(phi_z) * cos(psi_y)**2 - self.p.l * w_3y**2 * sin(phi_x) * sin(phi_z) * cos(psi_y)**2 - self.p.l * w_3y**2 * sin(phi_y) * cos(phi_x) * cos(phi_z) * cos(psi_y)**2 + self.p.l * w_3y * w_3z * sin(phi_x) * sin(phi_y) * cos(phi_z) * cos(psi_y)**2 - self.p.l * w_3y * w_3z * sin(phi_z) * cos(phi_x) * cos(psi_y)**2) * cos(psi_x)**2) * cos(phi_y) * tan(psi_x) + (-psi_y_dot * (psi_x_dot * self.p.r1 * sin(psi_y) + psi_x_dot * self.p.r2 * sin(psi_y) + self.p.r1 * w_1z + self.p.r2 * w_2z) * cos(phi_y) + self.p.l * (w_3x * (sin(phi_x) * sin(phi_y) * cos(phi_z) - sin(phi_z) * cos(phi_x)) - w_3y * cos(phi_y) * cos(phi_z)) * (w_3y * sin(phi_x) + w_3z * cos(phi_x)) * cos(psi_y)**2 + (psi_x_dot**2 * (self.p.r1 + self.p.r2) * sin(psi_x) - self.p.l * w_3x * (sin(phi_x) * cos(phi_z) - sin(phi_y) * sin(phi_z) * cos(phi_x)) * (w_3x + w_3y * sin(phi_x) * tan(phi_y) + w_3z * cos(phi_x) * tan(phi_y)) + self.p.l * (w_3y * cos(phi_x) - w_3z * sin(phi_x)) * (w_3x * sin(phi_x) * cos(phi_y) + w_3y * sin(phi_y)) * sin(phi_z)) * cos(phi_y) * cos(psi_y)**2) * sin(psi_y) * cos(psi_x)**2) + self.p.r1**2 * self.p.r2 * (self.p.m1 * \
                 (psi_y_dot * (psi_x_dot * self.p.r1 * sin(psi_y) + psi_x_dot * self.p.r2 * sin(psi_y) + self.p.r1 * w_1z + self.p.r2 * w_2z) * sin(psi_y) * cos(psi_x)**2 + (psi_x_dot * (psi_x_dot * (self.p.r1 + self.p.r2) * sin(psi_y) + self.p.r1 * w_1z + self.p.r2 * w_2z) * cos(psi_y) + psi_y_dot * (psi_x_dot * self.p.r1 + psi_x_dot * self.p.r2 + self.p.r1 * w_1z * sin(psi_y) + self.p.r2 * w_2z * sin(psi_y)) * sin(psi_x) * cos(psi_x)) * tan(psi_x)) + self.p.m2 * ((psi_x_dot * (self.p.r1 * w_1z + self.p.r2 * w_2z + (psi_x_dot * self.p.r1 * tan(psi_y) + psi_x_dot * self.p.r2 * tan(psi_y) + (self.p.r1 + self.p.r2) * (psi_x_dot * sin(psi_y) * cos(psi_x) + psi_y_dot * sin(psi_x) * cos(psi_y)) * cos(psi_x)**2) * cos(psi_y)) * cos(psi_y) + psi_y_dot * (psi_x_dot * self.p.r1 * sin(psi_x) * cos(psi_y)**3 + psi_x_dot * self.p.r1 * tan(psi_x) + psi_x_dot * self.p.r2 * sin(psi_x) * cos(psi_y)**3 + psi_x_dot * self.p.r2 * tan(psi_x) - psi_y_dot * self.p.r1 * sin(psi_y)**3 * cos(psi_x) + psi_y_dot * self.p.r1 * sin(psi_y) * cos(psi_x) - psi_y_dot * self.p.r2 * sin(psi_y)**3 * cos(psi_x) + psi_y_dot * self.p.r2 * sin(psi_y) * cos(psi_x) + self.p.r1 * w_1z * sin(psi_y) * tan(psi_x) + self.p.r2 * w_2z * sin(psi_y) * tan(psi_x)) * cos(psi_x)**2) * tan(psi_x) - (psi_x_dot**2 * (self.p.r1 + self.p.r2) * sin(psi_x) * cos(psi_y)**2 - psi_y_dot * (psi_x_dot * self.p.r1 * sin(psi_y) + psi_x_dot * self.p.r2 * sin(psi_y) + self.p.r1 * w_1z + self.p.r2 * w_2z)) * sin(psi_y) * cos(psi_x)**2)) * cos(phi_y) + self.p.r1**2 * (-Tx * sin(phi_y) + Ty * sin(phi_x) * cos(phi_y) + Tz * cos(phi_x) * cos(phi_y)) * cos(phi_y) * cos(psi_x)**2 * cos(psi_y)**3 + self.p.r2 * self.p.theta1 * (psi_y_dot * (psi_x_dot * self.p.r1 * sin(psi_y) + psi_x_dot * self.p.r2 * sin(psi_y) + self.p.r1 * w_1z + self.p.r2 * w_2z) * sin(psi_y) * cos(psi_x)**2 + (psi_x_dot * (psi_x_dot * (self.p.r1 + self.p.r2) * sin(psi_y) + self.p.r1 * w_1z + self.p.r2 * w_2z) * cos(psi_y) + psi_y_dot * (psi_x_dot * self.p.r1 + psi_x_dot * self.p.r2 + self.p.r1 * w_1z * sin(psi_y) + self.p.r2 * w_2z * sin(psi_y)) * sin(psi_x) * cos(psi_x)) * tan(psi_x)) * cos(phi_y)) / (self.p.r1**2 * cos(phi_y) * cos(psi_x)**2 * cos(psi_y)**3)
        b[6] = -(-self.p.l * self.p.m3 * (-((sin(phi_x) * sin(phi_y) * sin(phi_z) + cos(phi_x) * cos(phi_z)) * (-psi_y_dot * (psi_x_dot * self.p.r1 * sin(psi_y) + psi_x_dot * self.p.r2 * sin(psi_y) + self.p.r1 * w_1z + self.p.r2 * w_2z) * cos(phi_y) + self.p.l * (w_3x * (sin(phi_x) * sin(phi_y) * cos(phi_z) - sin(phi_z) * cos(phi_x)) - w_3y * cos(phi_y) * cos(phi_z)) * (w_3y * sin(phi_x) + w_3z * cos(phi_x)) * cos(psi_y)**2 + (psi_x_dot**2 * (self.p.r1 + self.p.r2) * sin(psi_x) - self.p.l * w_3x * (sin(phi_x) * cos(phi_z) - sin(phi_y) * sin(phi_z) * cos(phi_x)) * (w_3x + w_3y * sin(phi_x) * tan(phi_y) + w_3z * cos(phi_x) * tan(phi_y)) + self.p.l * (w_3y * cos(phi_x) - w_3z * sin(phi_x)) * (w_3x * sin(phi_x) * cos(phi_y) + w_3y * sin(phi_y)) * sin(phi_z)) * cos(phi_y) * cos(psi_y)**2) + (-psi_x_dot * (self.p.r1 + self.p.r2) * (psi_x_dot * cos(psi_x) * cos(psi_y) - psi_y_dot * sin(psi_x) * sin(psi_y)) + psi_y_dot * (self.p.r1 + self.p.r2) * (psi_x_dot * sin(psi_x) * sin(psi_y) - psi_y_dot * cos(psi_x) * cos(psi_y)) + self.p.g + self.p.l * w_3x * (w_3x + w_3y * sin(phi_x) * tan(phi_y) + w_3z * cos(phi_x) * tan(phi_y)) * cos(phi_x) * cos(phi_y) - self.p.l * (w_3y * cos(phi_x) - w_3z * sin(phi_x)) * (w_3x * sin(phi_x) * sin(phi_y) - w_3y * cos(phi_y))) * sin(phi_x) * cos(phi_y)**2 * cos(psi_y)**2) * cos(psi_x)**2 + (psi_x_dot * (self.p.r1 * w_1z + self.p.r2 * w_2z + (psi_x_dot * self.p.r1 * tan(psi_y) + psi_x_dot * self.p.r2 * tan(psi_y) + \
                 (self.p.r1 + self.p.r2) * (psi_x_dot * sin(psi_y) * cos(psi_x) + psi_y_dot * sin(psi_x) * cos(psi_y)) * cos(psi_x)**2) * cos(psi_y)) * cos(psi_y) + (psi_x_dot * psi_y_dot * self.p.r1 * sin(psi_x) * cos(psi_y)**3 + psi_x_dot * psi_y_dot * self.p.r1 * tan(psi_x) + psi_x_dot * psi_y_dot * self.p.r2 * sin(psi_x) * cos(psi_y)**3 + psi_x_dot * psi_y_dot * self.p.r2 * tan(psi_x) - psi_y_dot**2 * self.p.r1 * sin(psi_y)**3 * cos(psi_x) + psi_y_dot**2 * self.p.r1 * sin(psi_y) * cos(psi_x) - psi_y_dot**2 * self.p.r2 * sin(psi_y)**3 * cos(psi_x) + psi_y_dot**2 * self.p.r2 * sin(psi_y) * cos(psi_x) + psi_y_dot * self.p.r1 * w_1z * sin(psi_y) * tan(psi_x) + psi_y_dot * self.p.r2 * w_2z * sin(psi_y) * tan(psi_x) - self.p.l * w_3x**2 * sin(phi_x) * sin(phi_z) * cos(psi_y)**2 - self.p.l * w_3x**2 * sin(phi_y) * cos(phi_x) * cos(phi_z) * cos(psi_y)**2 + self.p.l * w_3x * w_3z * cos(phi_y) * cos(phi_z) * cos(psi_y)**2 - self.p.l * w_3y**2 * sin(phi_x) * sin(phi_z) * cos(psi_y)**2 - self.p.l * w_3y**2 * sin(phi_y) * cos(phi_x) * cos(phi_z) * cos(psi_y)**2 + self.p.l * w_3y * w_3z * sin(phi_x) * sin(phi_y) * cos(phi_z) * cos(psi_y)**2 - self.p.l * w_3y * w_3z * sin(phi_z) * cos(phi_x) * cos(psi_y)**2) * cos(psi_x)**2) * (sin(phi_x) * sin(phi_y) * cos(phi_z) - sin(phi_z) * cos(phi_x)) * cos(phi_y)) + (-Tx - self.p.theta3y * w_3y * w_3z + self.p.theta3z * w_3y * w_3z) * cos(phi_y) * cos(psi_x)**2 * cos(psi_y)**2) / (cos(phi_y) * cos(psi_x)**2 * cos(psi_y)**2)
        b[7] = -(self.p.l * self.p.m3 * ((-(-psi_y_dot * (psi_x_dot * self.p.r1 * sin(psi_y) + psi_x_dot * self.p.r2 * sin(psi_y) + self.p.r1 * w_1z + self.p.r2 * w_2z) * cos(phi_y) + self.p.l * (w_3x * (sin(phi_x) * sin(phi_y) * cos(phi_z) - sin(phi_z) * cos(phi_x)) - w_3y * cos(phi_y) * cos(phi_z)) * (w_3y * sin(phi_x) + w_3z * cos(phi_x)) * cos(psi_y)**2 + (psi_x_dot**2 * (self.p.r1 + self.p.r2) * sin(psi_x) - self.p.l * w_3x * (sin(phi_x) * cos(phi_z) - sin(phi_y) * sin(phi_z) * cos(phi_x)) * (w_3x + w_3y * sin(phi_x) * tan(phi_y) + w_3z * cos(phi_x) * tan(phi_y)) + self.p.l * (w_3y * cos(phi_x) - w_3z * sin(phi_x)) * (w_3x * sin(phi_x) * cos(phi_y) + w_3y * sin(phi_y)) * sin(phi_z)) * cos(phi_y) * cos(psi_y)**2) * sin(phi_z) + (-psi_x_dot * (self.p.r1 + self.p.r2) * (psi_x_dot * cos(psi_x) * cos(psi_y) - psi_y_dot * sin(psi_x) * sin(psi_y)) + psi_y_dot * (self.p.r1 + self.p.r2) * (psi_x_dot * sin(psi_x) * sin(psi_y) - psi_y_dot * cos(psi_x) * cos(psi_y)) + self.p.g + self.p.l * w_3x * (w_3x + w_3y * sin(phi_x) * tan(phi_y) + w_3z * cos(phi_x) * tan(phi_y)) * cos(phi_x) * cos(phi_y) - self.p.l * (w_3y * cos(phi_x) - w_3z * sin(phi_x)) * (w_3x * sin(phi_x) * sin(phi_y) - w_3y * cos(phi_y))) * sin(phi_y) * cos(psi_y)**2) * cos(psi_x)**2 + (psi_x_dot * (self.p.r1 * w_1z + self.p.r2 * w_2z + (psi_x_dot * self.p.r1 * tan(psi_y) + psi_x_dot * self.p.r2 * tan(psi_y) + \
                 (self.p.r1 + self.p.r2) * (psi_x_dot * sin(psi_y) * cos(psi_x) + psi_y_dot * sin(psi_x) * cos(psi_y)) * cos(psi_x)**2) * cos(psi_y)) * cos(psi_y) + (psi_x_dot * psi_y_dot * self.p.r1 * sin(psi_x) * cos(psi_y)**3 + psi_x_dot * psi_y_dot * self.p.r1 * tan(psi_x) + psi_x_dot * psi_y_dot * self.p.r2 * sin(psi_x) * cos(psi_y)**3 + psi_x_dot * psi_y_dot * self.p.r2 * tan(psi_x) - psi_y_dot**2 * self.p.r1 * sin(psi_y)**3 * cos(psi_x) + psi_y_dot**2 * self.p.r1 * sin(psi_y) * cos(psi_x) - psi_y_dot**2 * self.p.r2 * sin(psi_y)**3 * cos(psi_x) + psi_y_dot**2 * self.p.r2 * sin(psi_y) * cos(psi_x) + psi_y_dot * self.p.r1 * w_1z * sin(psi_y) * tan(psi_x) + psi_y_dot * self.p.r2 * w_2z * sin(psi_y) * tan(psi_x) - self.p.l * w_3x**2 * sin(phi_x) * sin(phi_z) * cos(psi_y)**2 - self.p.l * w_3x**2 * sin(phi_y) * cos(phi_x) * cos(phi_z) * cos(psi_y)**2 + self.p.l * w_3x * w_3z * cos(phi_y) * cos(phi_z) * cos(psi_y)**2 - self.p.l * w_3y**2 * sin(phi_x) * sin(phi_z) * cos(psi_y)**2 - self.p.l * w_3y**2 * sin(phi_y) * cos(phi_x) * cos(phi_z) * cos(psi_y)**2 + self.p.l * w_3y * w_3z * sin(phi_x) * sin(phi_y) * cos(phi_z) * cos(psi_y)**2 - self.p.l * w_3y * w_3z * sin(phi_z) * cos(phi_x) * cos(psi_y)**2) * cos(psi_x)**2) * cos(phi_y) * cos(phi_z)) + (-Ty + self.p.theta3x * w_3x * w_3z - self.p.theta3z * w_3x * w_3z) * cos(psi_x)**2 * cos(psi_y)**2) / (cos(psi_x)**2 * cos(psi_y)**2)
        b[8] = Tz + self.p.theta3x * w_3x * w_3y - self.p.theta3y * w_3x * w_3y

        return b

    def _compute_r_OSi(self, state):
        """computes center of mass locations of all bodies

        args:
            state (ModelState): current state

        Returns: list of x/y/z coordinates of center of mass of lower ball, upper ball, and lever arm.
        """
        pos_x = state.pos[0]
        pos_y = state.pos[1]
        psi_x = state.psi_x
        psi_y = state.psi_y
        [phi_x, phi_y, phi_z] = Quaternion(state.q3).get_roll_pitch_yaw()

        r_OS1 = np.array([pos_x, pos_y, self.p.r1])

        r12 = self.p.r1 + self.p.r2
        r_S1S2 = r12 * np.array([sin(psi_y) * cos(psi_x), -sin(psi_x), cos(psi_x) * cos(psi_y)])

        r_OS2 = r_OS1 + r_S1S2

        r_S2S3 = -self.p.l * np.array([sin(phi_x) * sin(phi_z) + sin(phi_y) * cos(phi_x) * cos(phi_z), -sin(
            phi_x) * cos(phi_z) + sin(phi_y) * sin(phi_z) * cos(phi_x), cos(phi_x) * cos(phi_y)])

        r_OS3 = r_OS2 + r_S2S3

        return [r_OS1, r_OS2, r_OS3]

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
