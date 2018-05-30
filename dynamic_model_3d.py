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

        xdot.omega = self._compute_omega_dot(eval_state)

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

        omega_1 = np.zeros(3)

        x0 = 1 / self.p.r1
        x1 = cos(psi_y)
        x2 = 1 / x1
        x3 = psi_x_dot * self.p.r1
        x4 = psi_x_dot * self.p.r2
        x5 = sin(psi_y)
        x6 = self.p.r1 * w_1z
        x7 = self.p.r2 * w_2z
        x8 = tan(psi_x)
        x9 = x8 * tan(psi_y)
        x10 = x2 * x8
        omega_1[0] = x0 * x2 * (-self.p.r2 * w_2x * x1 + x3 + x4 + x5 * x6 + x5 * x7)
        omega_1[1] = x0 * (psi_y_dot * self.p.r1 + psi_y_dot * self.p.r2 -
                           self.p.r2 * w_2y - x10 * x6 - x10 * x7 - x3 * x9 - x4 * x9)
        omega_1[2] = w_1z

        return omega_1

    def _compute_omega_dot(self, state):
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

        A = np.zeros([9, 9])
        b = np.zeros(9)

        # auto-generated symbolic expressions
        x0 = sin(psi_y)
        x1 = x0**2
        x2 = cos(psi_y)
        x3 = x2**2
        x4 = 1 / x3
        x5 = self.p.theta1 * x4
        x6 = x1 * x5
        x7 = tan(psi_x)
        x8 = x7**2
        x9 = x5 * x8
        x10 = self.p.r1**2
        x11 = self.p.m1 * x10 * x4
        x12 = self.p.m2 * x10 * x4
        x13 = self.p.m3 * x10 * x4
        x14 = self.p.r1 + self.p.r2
        x15 = x0 * x4
        x16 = 1 / self.p.r1
        x17 = x14 * x16
        x18 = self.p.m1 * self.p.r1
        x19 = 1 / x2
        x20 = tan(psi_y)
        x21 = x20 * x7
        x22 = -self.p.r1 * x21 - self.p.r2 * x21
        x23 = x19 * x22 * x7
        x24 = cos(psi_x)
        x25 = x14 * x24
        x26 = x14 * x19
        x27 = -x25 - x26
        x28 = self.p.r1 * x0 * x19 * x27
        x29 = self.p.theta1 * x16
        x30 = self.p.m2 * self.p.r1 * x19 * x7
        x31 = sin(psi_x)
        x32 = x0 * x14 * x31
        x33 = x22 - x32
        x34 = self.p.r1 * x19 * x7
        x35 = self.p.m1 * self.p.r1 * x14 * x15 - self.p.m2 * x28 - self.p.m3 * x28 - self.p.m3 * \
            x33 * x34 + self.p.theta1 * x0 * x17 * x4 - x18 * x23 - x23 * x29 - x30 * x33
        x36 = x14 * x19 * x7
        x37 = x2 * x25
        x38 = x14 + x37
        x39 = self.p.m3 * x38
        x40 = -x18 * x36 - x29 * x36 - x30 * x38 - x34 * x39
        x41 = self.p.m1 * self.p.r1 * self.p.r2 * x19
        x42 = self.p.m2 * self.p.r1 * self.p.r2 * x19
        x43 = self.p.m3 * self.p.r1 * self.p.r2 * x19
        x44 = self.p.r2 * self.p.theta1 * x16 * x19
        x45 = x0 * x44
        x46 = -x0 * x41 - x0 * x42 - x0 * x43 - x45
        x47 = x44 * x7
        x48 = x41 * x7 + x42 * x7 + x43 * x7 + x47
        x49 = self.p.m1 * self.p.r1 * self.p.r2 * x4
        x50 = self.p.m2 * self.p.r1 * self.p.r2 * x4
        x51 = self.p.m3 * self.p.r1 * self.p.r2 * x4
        x52 = self.p.r2 * x16
        x53 = x1 * x49 + x1 * x50 + x1 * x51 + x49 * x8 + x50 * x8 + x51 * x8 + x52 * x6 + x52 * x9
        x54 = self.p.r1 * x0 * x19
        x55 = cos(phi_x)
        x56 = cos(phi_z)
        x57 = x55 * x56
        x58 = sin(phi_y)
        x59 = sin(phi_x)
        x60 = sin(phi_z)
        x61 = x59 * x60
        x62 = x58 * x61
        x63 = x57 + x62
        x64 = self.p.l * self.p.m3 * x63
        x65 = x55 * x60
        x66 = x56 * x59
        x67 = x58 * x66
        x68 = -x65 + x67
        x69 = self.p.l * self.p.m3 * x68
        x70 = -x34 * x69 - x54 * x64
        x71 = cos(phi_y)
        x72 = self.p.l * self.p.m3 * self.p.r1 * x19 * x71
        x73 = x0 * x60 * x72 + x56 * x7 * x72
        x74 = x14**2
        x75 = self.p.m1 * x74
        x76 = 1 / x10
        x77 = self.p.theta1 * x76
        x78 = x74 * x77
        x79 = self.p.m2 * x74
        x80 = x3 * x31**2
        x81 = self.p.m3 * x74
        x82 = x22**2
        x83 = x27**2
        x84 = x33**2
        x85 = x0 * x2 * x24 * x31
        x86 = self.p.m1 * x22
        x87 = self.p.theta1 * x22 * x76
        x88 = self.p.m2 * x33 * x38 + x14 * x86 + x14 * x87 + x33 * x39 + x79 * x85 + x81 * x85
        x89 = self.p.r2 * x14
        x90 = self.p.m1 * x89
        x91 = x19 * x90
        x92 = x77 * x89
        x93 = x19 * x92
        x94 = self.p.r2 * x27
        x95 = self.p.m2 * x94
        x96 = self.p.m3 * x94
        x97 = -x91 - x93 + x95 + x96
        x98 = self.p.r2 * x86
        x99 = self.p.r2 * x87
        x100 = self.p.r2 * x33
        x101 = self.p.m2 * x100
        x102 = self.p.m3 * x100
        x103 = -x101 - x102 - x98 - x99
        x104 = x19 * x7
        x105 = x0 * x19
        x106 = -x101 * x104 - x102 * x104 - x104 * x98 - x104 * \
            x99 - x105 * x95 - x105 * x96 + x15 * x90 + x15 * x92
        x107 = self.p.l * self.p.m3 * x59 * x71
        x108 = x14 * x2 * x31
        x109 = -x107 * x108 + x27 * x64 + x33 * x69
        x110 = self.p.l * self.p.m3 * x58
        x111 = self.p.l * self.p.m3
        x112 = -x108 * x110 - x111 * x27 * x60 * x71 - x111 * x33 * x56 * x71
        x113 = x38**2
        x114 = x1 * x24**2
        x115 = self.p.r2 * x38
        x116 = self.p.m2 * x115
        x117 = self.p.m3 * x115
        x118 = -x116 - x117 - x90 - x92
        x119 = -x104 * x116 - x104 * x117 - x7 * x91 - x7 * x93
        x120 = x0 * x14 * x24
        x121 = self.p.l * self.p.m3 * x38
        x122 = -x107 * x120 + x121 * x68
        x123 = x56 * x71
        x124 = -x110 * x120 - x121 * x123
        x125 = self.p.r2**2
        x126 = self.p.m1 * x125 + self.p.m2 * x125 + self.p.m3 * x125 + self.p.theta2 + x125 * x77
        x127 = self.p.m1 * x125 * x19
        x128 = self.p.m2 * x125 * x19
        x129 = self.p.m3 * x125 * x19
        x130 = self.p.theta1 * x125 * x19 * x76
        x131 = -x0 * x127 - x0 * x128 - x0 * x129 - x0 * x130
        x132 = self.p.l * self.p.m3 * self.p.r2
        x133 = x132 * x63
        x134 = self.p.l * self.p.m3 * self.p.r2 * x71
        x135 = x134 * x60
        x136 = -x135
        x137 = x127 * x7 + x128 * x7 + x129 * x7 + x130 * x7
        x138 = x132 * x68
        x139 = -x138
        x140 = x134 * x56
        x141 = self.p.m1 * x125 * x4
        x142 = self.p.m2 * x125 * x4
        x143 = self.p.m3 * x125 * x4
        x144 = self.p.theta1 * x125 * x4 * x76
        x145 = -x104 * x138 - x105 * x133
        x146 = x104 * x140 + x105 * x135
        x147 = x59**2
        x148 = self.p.l**2
        x149 = self.p.m3 * x148 * x71**2
        x150 = self.p.m3 * x148
        x151 = x60 * x71
        x152 = -self.p.m3 * x123 * x148 * x68 - self.p.m3 * x148 * x151 * x63 + x150 * x58 * x59 * x71
        x153 = self.p.r1 * w_1z
        x154 = self.p.r2 * w_2x
        x155 = self.p.r2 * w_2z
        x156 = x19 * (x0 * x154 + x153 * x2 + x155 * x2)
        x157 = psi_x_dot * self.p.r1
        x158 = psi_x_dot * self.p.r2
        x159 = x0 * x153
        x160 = x0 * x155
        x161 = x15 * (-x154 * x2 + x157 + x158 + x159 + x160)
        x162 = psi_y_dot * (-self.p.m1 * x156 - self.p.m1 * x161)
        x163 = x156 * x16 + x16 * x161
        x164 = psi_y_dot * self.p.theta1 * x163
        x165 = psi_x_dot**2 * x14 * x31
        x166 = psi_y_dot * (-x156 - x161)
        x167 = self.p.m2 * x165 + self.p.m2 * x166
        x168 = x7 * (x20**2 + 1)
        x169 = x4 * x7
        x170 = -x157 * x168 - x158 * x168 - x159 * x169 - x160 * x169
        x171 = psi_y_dot * x170
        x172 = x8 + 1
        x173 = x172 * x20
        x174 = x172 * x19
        x175 = -x153 * x174 - x155 * x174 - x157 * x173 - x158 * x173
        x176 = psi_x_dot * x175
        x177 = self.p.m1 * x171 + self.p.m1 * x176
        x178 = x16 * x171 + x16 * x176
        x179 = self.p.theta1 * x178
        x180 = psi_y_dot * (-psi_x_dot * x108 - psi_y_dot * x120 + x170)
        x181 = psi_x_dot * (-psi_x_dot * x120 - psi_y_dot * x108 + x175)
        x182 = self.p.m2 * x180 + self.p.m2 * x181
        x183 = x58 * x65
        x184 = 1 / x55
        x185 = w_3y * x184 * x59 + w_3z
        x186 = x55 * x71
        x187 = 1 / (x147 * x184 * x71 + x186)
        x188 = x185 * x187
        x189 = self.p.l * self.p.m3 * w_3x * (w_3x + x188 * x58)
        x190 = self.p.l * w_3y * x71
        x191 = self.p.l * w_3x
        x192 = self.p.m3 * x185 * x187
        x193 = self.p.l * w_3y * x58
        x194 = self.p.l * w_3x * x71
        x195 = x59 * x71
        x196 = self.p.m3 * x184 * (w_3y - x188 * x195)
        x197 = self.p.m3 * x165 + self.p.m3 * x166 + x189 * \
            (x183 - x66) + x192 * (-x190 * x56 + x191 * x68) + x196 * (x193 * x60 + x194 * x61)
        x198 = x57 * x58
        x199 = -x57 - x62
        x200 = self.p.m3 * x180 + self.p.m3 * x181 + x189 * \
            (x198 + x61) + x192 * (x190 * x60 + x191 * x199) + x196 * (x193 * x56 + x194 * x66)
        x201 = psi_x_dot * (-psi_x_dot * x37 + psi_y_dot * x32)
        x202 = psi_y_dot * (psi_x_dot * x32 - psi_y_dot * x37)
        x203 = self.p.g * self.p.m2 + self.p.m2 * x201 + self.p.m2 * x202
        x204 = psi_y_dot * x163
        x205 = self.p.g * self.p.m3 + self.p.m3 * x201 + self.p.m3 * \
            x202 + x186 * x189 + x196 * (x190 - x191 * x58 * x59)
        x206 = Tx * x71
        x207 = self.p.r2 * x162
        x208 = self.p.r2 * x167
        x209 = self.p.r2 * x197
        x210 = self.p.r2 * x177
        x211 = self.p.r2 * x182
        x212 = self.p.r2 * x200
        x213 = self.p.theta3z * w_3z
        x214 = self.p.theta3y * w_3y
        x215 = self.p.l * x205
        x216 = self.p.l * x197
        x217 = self.p.l * x200
        x218 = self.p.theta3x * w_3x
        A[0, 0] = self.p.theta1 + x1 * x11 + x1 * x12 + x1 * \
            x13 + x11 * x8 + x12 * x8 + x13 * x8 + x6 + x9
        A[0, 1] = x35
        A[0, 2] = x40
        A[0, 3] = x46
        A[0, 4] = x48
        A[0, 5] = x53
        A[0, 6] = x70
        A[0, 7] = x73
        A[0, 8] = 0
        A[1, 0] = x35
        A[1, 1] = self.p.m1 * x82 + self.p.m2 * x83 + self.p.m2 * x84 + self.p.m3 * \
            x83 + self.p.m3 * x84 + x4 * x75 + x4 * x78 + x77 * x82 + x79 * x80 + x80 * x81
        A[1, 2] = x88
        A[1, 3] = x97
        A[1, 4] = x103
        A[1, 5] = x106
        A[1, 6] = x109
        A[1, 7] = x112
        A[1, 8] = 0
        A[2, 0] = x40
        A[2, 1] = x88
        A[2, 2] = self.p.m2 * x113 + self.p.m3 * x113 + x114 * x79 + x114 * x81 + x75 + x78
        A[2, 3] = 0
        A[2, 4] = x118
        A[2, 5] = x119
        A[2, 6] = x122
        A[2, 7] = x124
        A[2, 8] = 0
        A[3, 0] = x46
        A[3, 1] = x97
        A[3, 2] = 0
        A[3, 3] = x126
        A[3, 4] = 0
        A[3, 5] = x131
        A[3, 6] = x133
        A[3, 7] = x136
        A[3, 8] = 0
        A[4, 0] = x48
        A[4, 1] = x103
        A[4, 2] = x118
        A[4, 3] = 0
        A[4, 4] = x126
        A[4, 5] = x137
        A[4, 6] = x139
        A[4, 7] = x140
        A[4, 8] = 0
        A[5, 0] = x53
        A[5, 1] = x106
        A[5, 2] = x119
        A[5, 3] = x131
        A[5, 4] = x137
        A[5, 5] = self.p.theta2 + x1 * x141 + x1 * x142 + x1 * x143 + \
            x1 * x144 + x141 * x8 + x142 * x8 + x143 * x8 + x144 * x8
        A[5, 6] = x145
        A[5, 7] = x146
        A[5, 8] = 0
        A[6, 0] = x70
        A[6, 1] = x109
        A[6, 2] = x122
        A[6, 3] = x133
        A[6, 4] = x139
        A[6, 5] = x145
        A[6, 6] = self.p.theta3x + x147 * x149 + x150 * x63**2 + x150 * x68**2
        A[6, 7] = x152
        A[6, 8] = 0
        A[7, 0] = x73
        A[7, 1] = x112
        A[7, 2] = x124
        A[7, 3] = x136
        A[7, 4] = x140
        A[7, 5] = x146
        A[7, 6] = x152
        A[7, 7] = self.p.theta3y + x149 * x56**2 + x149 * x60**2 + x150 * x58**2
        A[7, 8] = 0
        A[8, 0] = 0
        A[8, 1] = 0
        A[8, 2] = 0
        A[8, 3] = 0
        A[8, 4] = 0
        A[8, 5] = 0
        A[8, 6] = 0
        A[8, 7] = 0
        A[8, 8] = self.p.theta3z
        b[0] = x104 * x179 - x105 * x164 + x162 * x54 + x167 * \
            x54 + x177 * x34 + x182 * x34 + x197 * x54 + x200 * x34
        b[1] = -self.p.theta1 * x14 * x16 * x19 * x204 + x108 * x203 + x108 * x205 + x162 * \
            x26 - x167 * x27 - x177 * x22 - x178 * x22 * x29 - x182 * x33 - x197 * x27 - x200 * x33
        b[2] = x120 * x203 + x120 * x205 - x14 * x177 - x17 * x179 - x182 * x38 - x200 * x38
        b[3] = Ty * (x65 - x67) + Tz * (-x198 - x61) + x164 * x52 - x206 * x56 - x207 - x208 - x209
        b[4] = Ty * x199 + Tz * (-x183 + x66) + x179 * x52 - x206 * x60 + x210 + x211 + x212
        b[5] = Tx * x58 - Ty * x195 - Tz * x186 + x104 * x210 + x104 * x211 + x104 * \
            x212 + x105 * x207 + x105 * x208 + x105 * x209 + x178 * x47 - x204 * x45
        b[6] = Tx - w_3y * x213 + w_3z * x214 - x195 * x215 - x216 * x63 - x217 * x68
        b[7] = Ty + w_3x * x213 - w_3z * x218 + x123 * x217 + x151 * x216 - x215 * x58
        b[8] = Tz - w_3x * x214 + w_3y * x218

        omega_dot = np.linalg.solve(A, b)

        return omega_dot

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

        r_S1S2 = np.zeros(3)
        r_S2S3 = np.zeros(3)

        x0 = self.p.r1 + self.p.r2
        x1 = x0 * cos(psi_x)
        x2 = sin(phi_x)
        x3 = sin(phi_z)
        x4 = cos(phi_z)
        x5 = cos(phi_x)
        x6 = x5 * sin(phi_y)
        r_S1S2[0] = x1 * sin(psi_y)
        r_S1S2[1] = -x0 * sin(psi_x)
        r_S1S2[2] = x1 * cos(psi_y)
        r_S2S3[0] = -self.p.l * (x2 * x3 + x4 * x6)
        r_S2S3[1] = -self.p.l * (-x2 * x4 + x3 * x6)
        r_S2S3[2] = -self.p.l * x5 * cos(phi_y)

        r_OS2 = r_OS1 + r_S1S2
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
