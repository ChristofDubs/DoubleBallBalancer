"""Dynamic model of 3D version of Double Ball Balancer

This module contains the class DynamicModel for simulating the non-linear dynamics of the 3D Double Ball Balancer and a corresponding class ModelParam containing the relevant parameters.

author: Christof Dubs
"""
import numpy as np
from numpy import sin, cos
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
        self.theta3 = 1.0

    def is_valid(self,):
        """Checks validity of parameter configuration

        Returns:
            bool: True if valid, False if invalid.
        """
        return self.g > 0 and self.l > 0 and self.m1 > 0 and self.m2 > 0 and self.m3 > 0 and self.r1 > 0 and self.r2 > 0 and self.tau > 0 and self.theta1 > 0 and self.theta2 > 0 and self.theta3 > 0


# state size and indices
PSI_Y_IDX = 0
PSI_Z_IDX = 1
Q_1_W_IDX = 2
Q_1_X_IDX = 3
Q_1_Y_IDX = 4
Q_1_Z_IDX = 5
Q_2_W_IDX = 6
Q_2_X_IDX = 7
Q_2_Y_IDX = 8
Q_2_Z_IDX = 9
PSI_Y_DOT_IDX = 10
PSI_Z_DOT_IDX = 11
OMEGA_2_X_IDX = 12
OMEGA_2_Y_IDX = 13
OMEGA_2_Z_IDX = 14
X_IDX = 15
Y_IDX = 16
STATE_SIZE = 17


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

    def normalize_quaternions(self):
        """Normalize the rotation quaternions"""
        self.q1 *= 1.0 / np.linalg.norm(self.q1)
        self.q2 *= 1.0 / np.linalg.norm(self.q2)

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

        # quaternion check
        if q1_norm == 0 or q2_norm == 0:
            return false

        self.x = x0_flat
        self.normalize_quaternions()
        return True

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
    def psi_z(self):
        return self.x[PSI_Z_IDX]

    @psi_z.setter
    def psi_z(self, value):
        self.x[PSI_Z_IDX] = value

    @property
    def psi_z_dot(self):
        return self.x[PSI_Z_DOT_IDX]

    @psi_z_dot.setter
    def psi_z_dot(self, value):
        self.x[PSI_Z_DOT_IDX] = value

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
    def psi(self):
        return self.x[PSI_Y_IDX:PSI_Z_IDX + 1]

    @psi.setter
    def psi(self, value):
        self.x[PSI_Y_IDX:PSI_Z_IDX + 1] = value

    @property
    def psi_dot(self):
        return self.x[PSI_Y_DOT_IDX:PSI_Z_DOT_IDX + 1]

    @psi_dot.setter
    def psi_dot(self, value):
        self.x[PSI_Y_DOT_IDX:PSI_Z_DOT_IDX + 1] = value

    @property
    def pos(self):
        return self.x[X_IDX:Y_IDX + 1]

    @pos.setter
    def pos(self, value):
        self.x[X_IDX:Y_IDX + 1] = value

    @property
    def omega(self):
        return self.x[PSI_Y_DOT_IDX:OMEGA_2_Z_IDX + 1]

    @omega.setter
    def omega(self, value):
        self.x[PSI_Y_DOT_IDX:OMEGA_2_Z_IDX + 1] = value

    @property
    def omega_2(self):
        return self.x[OMEGA_2_X_IDX:OMEGA_2_Z_IDX + 1]

    @omega_2.setter
    def omega_2(self, value):
        self.x[OMEGA_2_X_IDX:OMEGA_2_Z_IDX + 1] = value


class DynamicModel:
    """Simulation interface for the 2D Double Ball Balancer

    Attributes:
        p (ModelParam): physical parameters
        state (ModelState): 17-dimensional state

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

        psi_y = state.psi_y

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
        # vis['lever_arm'] = [[r_OSi[1][0], r_OSi[2][0]], [r_OSi[1][1], r_OSi[2][1]]]
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
        psi_z = state.psi_z
        psi_y_dot = state.psi_y_dot
        psi_z_dot = state.psi_z_dot
        w_2x = state.omega_2[0]
        w_2y = state.omega_2[1]
        w_2z = state.omega_2[2]

        w_1x = -(psi_y_dot * self.p.r1 * np.sin(psi_z) + psi_y_dot *
                 self.p.r2 * np.sin(psi_z) + self.p.r2 * w_2x) / self.p.r1
        w_1y = (psi_y_dot * self.p.r1 * np.cos(psi_z) + psi_y_dot *
                self.p.r2 * np.cos(psi_z) - self.p.r2 * w_2y) / self.p.r1
        w_1z = (psi_z_dot * self.p.r1 + psi_z_dot * self.p.r2 - self.p.r2 * w_2z) / self.p.r1

        return np.array([w_1x, w_1y, w_1z])

    def _compute_acc_jacobian_matrix(self, state):
        """computes angular acceleration matrix of rotational part of system dynamics (equal to jacobian matrix since dynamics are linear in angular accelerations)

        The non-linear rotational dynamics are of the form

        A * [psi_y_ddot, psi_z_ddot, omega_2_x_dot, omega_2_y_dot, omega_2_z_dot] = b

        where A = A(psi_y, psi_z) and b = b(psi_y, psi_z, psi_y_dot, psi_z_dot)

        This function computes matrix A.

        args:
            x (ModelState): current state

        Returns: 5x5 angular acceleration matrix
        """
        psi_y = state.psi_y
        psi_z = state.psi_z

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

    def _compute_rhs(self, state):
        """computes state (and input) terms of system dynamics

        The non-linear rotational dynamics are of the form

        A * [psi_y_ddot, psi_z_ddot, omega_2_x_dot, omega_2_y_dot, omega_2_z_dot] = b

        where A = A(psi_y, psi_z) and b = b(psi_y, psi_z, psi_y_dot, psi_z_dot)

        This function computes vector b.

        args:
            state (ModelState): current state

        Returns: 5x1 array of state (and input) terms
        """
        b = np.zeros(5)

        psi_y = state.psi_y
        psi_z = state.psi_z
        psi_y_dot = state.psi_y_dot
        psi_z_dot = state.psi_z_dot

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

    def _compute_r_OSi(self, state):
        """computes center of mass locations of all bodies

        args:
            state (ModelState): current state

        Returns: list of x/y/z coordinates of center of mass of lower ball, upper ball, and lever arm.
        """
        pos_x = state.pos[0]
        pos_y = state.pos[1]
        psi_y = state.psi_y
        psi_z = state.psi_z

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
