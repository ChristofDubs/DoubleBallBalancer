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
        theta1: Mass moment of inertia of lower ball wrt. its center of mass around all axes [kg*m^2]
        theta2: Mass moment of inertia of upper ball wrt. its center of mass around all axes [kg*m^2]
        theta3x: Mass moment of inertia of lever arm wrt. its center of mass around x axis [kg*m^2]
        theta3y: Mass moment of inertia of lever arm wrt. its center of mass around y axis [kg*m^2]
        theta3z: Mass moment of inertia of lever arm wrt. its center of mass around z axis [kg*m^2]
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
        x (numpy.ndarray): array representing the full state
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
    def phi(self):
        return Quaternion(self.q3).get_roll_pitch_yaw()

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
    """Simulation interface for the 3D Double Ball Balancer

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

    def simulate_step(self, delta_t, omega_cmd):
        """Simulate one time step

        Simulates the changes of the state over a time interval.

        args:
            delta_t: time step [s]
            omega_cmd (np.ndarray): motor speed commands [rad/s]
        """
        t = np.array([0, delta_t])
        self.state.x = odeint(self._x_dot, self.state.x, t, args=(omega_cmd,))[-1]

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

    def _x_dot(self, x, t, omega_cmd):
        """computes the derivative of the state

        This function returns an numpy.array of the derivatives of the states, given the current state and inputs.

        Its signature is compatible with scipy.integrate.odeint's first callable argument.

        args:
            x (numpy.ndarray): state at which the state derivative function is evaluated
            t: time [s]. Since this system is time invariant, this argument is unused.
            omega_cmd (np.ndarray): motor speed commands [rad/s]
        returns:
            array containing the time derivatives of all states
        """
        eval_state = ModelState(x, skip_checks=True)

        # freeze system if state is irrecoverable
        if self.is_irrecoverable(eval_state):
            return np.zeros(np.shape(eval_state.x))

        xdot = ModelState()

        xdot.omega = self._compute_omega_dot(eval_state, omega_cmd)

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
        returns:
            array with x and y velocity of the lower ball [m/s]
        """
        return self.p.r1 * np.array([omega_1[1], -omega_1[0]])

    def _get_lower_ball_omega(self, state):
        """computes the angular velocity (x/y/z) of the lower ball

        args:
            state (ModelState): current state
        returns:
            array containing angular velocity of lower ball [rad/s]
        """
        [psi_x, psi_y] = state.psi
        [psi_x_dot, psi_y_dot] = state.psi_dot
        w_1z = state.omega_1_z
        [w_2x, w_2y, w_2z] = state.omega_2

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

    def _compute_omega_dot(self, state, omega_cmd):
        """computes angular acceleration matrix of rotational part of system dynamics (equal to jacobian matrix since dynamics are linear in angular accelerations)

        The non-linear rotational dynamics are of the form

        A * [omega_1_z_dot, psi_x_ddot, psi_y_ddot, omega_2_x_dot, omega_2_y_dot, omega_2_z_dot, omega_3_x_dot, omega_3_y_dot, omega_3_z_dot] = b

        where A = A(phi_x, phi_y, phi_z, psi_x, psi_y) and b(state, inputs).

        args:
            state (ModelState): current state
            omega_cmd (np.ndarray): motor speed commands [rad/s]

        Returns: array containing the time derivative of the angular velocity state [rad/s^2]
        """
        [psi_x, psi_y] = state.psi
        [psi_x_dot, psi_y_dot] = state.psi_dot
        [phi_x, phi_y, phi_z] = state.phi
        [phi_mx, phi_my, phi_mz] = (Quaternion(state.q2).inverse() *
                                    Quaternion(state.q3)).get_roll_pitch_yaw()
        w_1z = state.omega_1_z
        [w_2x, w_2y, w_2z] = state.omega_2
        [w_3x, w_3y, w_3z] = state.omega_3
        [omega_x_cmd, omega_y_cmd] = omega_cmd

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
        x24 = self.p.r1 * x0 * x19
        x25 = cos(psi_x)
        x26 = x14 * x25
        x27 = x14 * x19
        x28 = -x26 - x27
        x29 = self.p.theta1 * x16
        x30 = self.p.m2 * self.p.r1 * x19 * x7
        x31 = sin(psi_x)
        x32 = x0 * x14 * x31
        x33 = x22 - x32
        x34 = self.p.r1 * x19 * x7
        x35 = self.p.m1 * self.p.r1 * x14 * x15 - self.p.m2 * x24 * x28 - self.p.m3 * x24 * x28 - \
            self.p.m3 * x33 * x34 + self.p.theta1 * x0 * x17 * x4 - x18 * x23 - x23 * x29 - x30 * x33
        x36 = x14 * x19 * x7
        x37 = x2 * x26
        x38 = x14 + x37
        x39 = self.p.m3 * x38
        x40 = -x18 * x36 - x29 * x36 - x30 * x38 - x34 * x39
        x41 = self.p.m1 * self.p.r1 * self.p.r2 * x19
        x42 = x0 * x41
        x43 = self.p.m2 * self.p.r1 * self.p.r2 * x19
        x44 = x0 * x43
        x45 = self.p.m3 * self.p.r1 * self.p.r2 * x19
        x46 = x0 * x45
        x47 = self.p.r2 * self.p.theta1 * x16 * x19
        x48 = x0 * x47
        x49 = x41 * x7
        x50 = x43 * x7
        x51 = x45 * x7
        x52 = x47 * x7
        x53 = self.p.m1 * self.p.r1 * self.p.r2 * x4
        x54 = x1 * x53
        x55 = x53 * x8
        x56 = self.p.m2 * self.p.r1 * self.p.r2 * x4
        x57 = x1 * x56
        x58 = x56 * x8
        x59 = self.p.m3 * self.p.r1 * self.p.r2 * x4
        x60 = x1 * x59
        x61 = x59 * x8
        x62 = self.p.r2 * x16
        x63 = x6 * x62
        x64 = x62 * x9
        x65 = self.p.l * self.p.m3 * self.p.r1 * x0 * x19
        x66 = cos(phi_x)
        x67 = cos(phi_z)
        x68 = x66 * x67
        x69 = sin(phi_y)
        x70 = sin(phi_x)
        x71 = sin(phi_z)
        x72 = x70 * x71
        x73 = x69 * x72
        x74 = x68 + x73
        x75 = self.p.l * self.p.m3 * self.p.r1 * x19 * x7
        x76 = x66 * x71
        x77 = x67 * x70
        x78 = x69 * x77 - x76
        x79 = -x65 * x74 - x75 * x78
        x80 = cos(phi_y)
        x81 = x71 * x80
        x82 = x67 * x80
        x83 = x65 * x81 + x75 * x82
        x84 = x14**2
        x85 = self.p.m1 * x84
        x86 = 1 / x10
        x87 = self.p.theta1 * x86
        x88 = x84 * x87
        x89 = self.p.m2 * x84
        x90 = x3 * x31**2
        x91 = self.p.m3 * x84
        x92 = x22**2
        x93 = x28**2
        x94 = x33**2
        x95 = x0 * x2 * x25 * x31
        x96 = self.p.m1 * x22
        x97 = self.p.theta1 * x22 * x86
        x98 = self.p.m2 * x33 * x38 + x14 * x96 + x14 * x97 + x33 * x39 + x89 * x95 + x91 * x95
        x99 = self.p.r2 * x14
        x100 = self.p.m1 * x99
        x101 = x100 * x19
        x102 = x87 * x99
        x103 = x102 * x19
        x104 = self.p.r2 * x28
        x105 = self.p.m2 * x104
        x106 = self.p.m3 * x104
        x107 = self.p.r2 * x96
        x108 = self.p.r2 * x97
        x109 = self.p.r2 * x33
        x110 = self.p.m2 * x109
        x111 = self.p.m3 * x109
        x112 = x100 * x15
        x113 = x102 * x15
        x114 = x19 * x7
        x115 = x107 * x114
        x116 = x0 * x19
        x117 = x105 * x116
        x118 = x106 * x116
        x119 = x108 * x114
        x120 = x110 * x114
        x121 = x111 * x114
        x122 = self.p.l * self.p.m3 * x70 * x80
        x123 = x14 * x2 * x31
        x124 = self.p.l * self.p.m3 * x28
        x125 = self.p.l * self.p.m3 * x33
        x126 = -x122 * x123 + x124 * x74 + x125 * x78
        x127 = self.p.l * self.p.m3 * x69
        x128 = -x123 * x127 - x124 * x81 - x125 * x82
        x129 = x38**2
        x130 = x1 * x25**2
        x131 = self.p.r2 * x38
        x132 = self.p.m2 * x131
        x133 = self.p.m3 * x131
        x134 = x101 * x7
        x135 = x103 * x7
        x136 = x114 * x132
        x137 = x114 * x133
        x138 = x0 * x14 * x25
        x139 = self.p.l * self.p.m3 * x38
        x140 = -x122 * x138 + x139 * x78
        x141 = -x127 * x138 - x139 * x82
        x142 = x42 + x44 + x46 + x48
        x143 = -x49 - x50 - x51 - x52
        x144 = -x54 - x55 - x57 - x58 - x60 - x61 - x63 - x64
        x145 = x101 + x103 - x105 - x106
        x146 = x107 + x108 + x110 + x111
        x147 = -x112 - x113 + x115 + x117 + x118 + x119 + x120 + x121
        x148 = x100 + x102 + x132 + x133
        x149 = x134 + x135 + x136 + x137
        x150 = self.p.l * self.p.m3 * self.p.r2
        x151 = x150 * x74
        x152 = -self.p.theta2
        x153 = self.p.r2**2
        x154 = -self.p.m1 * x153 - self.p.m2 * x153 - self.p.m3 * x153 + x152 - x153 * x87
        x155 = self.p.m1 * x153 * x19
        x156 = self.p.m2 * x153 * x19
        x157 = self.p.m3 * x153 * x19
        x158 = self.p.theta1 * x153 * x19 * x86
        x159 = x0 * x155 + x0 * x156 + x0 * x157 + x0 * x158
        x160 = x150 * x78
        x161 = -x155 * x7 - x156 * x7 - x157 * x7 - x158 * x7
        x162 = x116 * x151
        x163 = x114 * x160
        x164 = self.p.m1 * x153 * x4
        x165 = self.p.m2 * x153 * x4
        x166 = self.p.m3 * x153 * x4
        x167 = self.p.theta1 * x153 * x4 * x86
        x168 = -x1 * x164 - x1 * x165 - x1 * x166 - x1 * x167 + \
            x152 - x164 * x8 - x165 * x8 - x166 * x8 - x167 * x8
        x169 = x150 * x82
        x170 = x150 * x81
        x171 = x169 * x74 - x170 * x78
        x172 = x70**2
        x173 = self.p.l**2
        x174 = self.p.m3 * x173 * x80**2
        x175 = self.p.m3 * x173
        x176 = x162 + x163
        x177 = x116 * x170
        x178 = x114 * x169
        x179 = -x177 - x178
        x180 = -x175 * x67 * x78 * x80 + x175 * x69 * x70 * x80 - x175 * x71 * x74 * x80
        x181 = x70 * x80
        x182 = x68 * x69 + x72
        x183 = x69 * x76 - x77
        x184 = x66 * x80
        x185 = 1.0 * x76
        x186 = 1.0 * x77
        x187 = x186 * x69
        x188 = 1.0 * x68
        x189 = 1.0 * x72
        x190 = x189 * x69
        x191 = 1.0 * x70
        x192 = x188 * x69
        x193 = x185 * x69
        x194 = self.p.r1 * w_1z
        x195 = self.p.r2 * w_2x
        x196 = self.p.r2 * w_2z
        x197 = x19 * (x0 * x195 + x194 * x2 + x196 * x2)
        x198 = psi_x_dot * self.p.r1
        x199 = psi_x_dot * self.p.r2
        x200 = x0 * x194
        x201 = x0 * x196
        x202 = x15 * (-x195 * x2 + x198 + x199 + x200 + x201)
        x203 = psi_y_dot * (-self.p.m1 * x197 - self.p.m1 * x202)
        x204 = x16 * x197 + x16 * x202
        x205 = psi_y_dot * self.p.theta1 * x204
        x206 = psi_x_dot**2 * x14 * x31
        x207 = psi_y_dot * (-x197 - x202)
        x208 = self.p.m2 * x206 + self.p.m2 * x207
        x209 = x7 * (x20**2 + 1)
        x210 = x4 * x7
        x211 = -x198 * x209 - x199 * x209 - x200 * x210 - x201 * x210
        x212 = psi_y_dot * x211
        x213 = x8 + 1
        x214 = x20 * x213
        x215 = x19 * x213
        x216 = -x194 * x215 - x196 * x215 - x198 * x214 - x199 * x214
        x217 = psi_x_dot * x216
        x218 = self.p.m1 * x212 + self.p.m1 * x217
        x219 = x16 * x212 + x16 * x217
        x220 = self.p.theta1 * x219
        x221 = psi_y_dot * (-psi_x_dot * x123 - psi_y_dot * x138 + x211)
        x222 = psi_x_dot * (-psi_x_dot * x138 - psi_y_dot * x123 + x216)
        x223 = self.p.m2 * x221 + self.p.m2 * x222
        x224 = w_3y * x70
        x225 = 1 / x66
        x226 = w_3z + x224 * x225
        x227 = 1 / (x172 * x225 * x80 + x184)
        x228 = x226 * x227
        x229 = self.p.l * self.p.m3 * w_3x * (w_3x + x228 * x69)
        x230 = self.p.l * w_3y
        x231 = self.p.l * w_3x
        x232 = self.p.m3 * x226 * x227
        x233 = x69 * x71
        x234 = self.p.l * w_3x * x80
        x235 = self.p.m3 * x225 * (w_3y - x181 * x228)
        x236 = self.p.m3 * x206 + self.p.m3 * x207 + x183 * x229 + x232 * \
            (-x230 * x82 + x231 * x78) + x235 * (x230 * x233 + x234 * x72)
        x237 = x67 * x69
        x238 = self.p.m3 * x221 + self.p.m3 * x222 + x182 * x229 + x232 * \
            (x230 * x81 + x231 * (-x68 - x73)) + x235 * (x230 * x237 + x234 * x77)
        x239 = psi_x_dot * (-psi_x_dot * x37 + psi_y_dot * x32)
        x240 = psi_y_dot * (psi_x_dot * x32 - psi_y_dot * x37)
        x241 = self.p.g * self.p.m2 + self.p.m2 * x239 + self.p.m2 * x240
        x242 = psi_y_dot * x204
        x243 = self.p.g * self.p.m3 + self.p.m3 * x239 + self.p.m3 * \
            x240 + x184 * x229 + x235 * (x230 * x80 - x231 * x69 * x70)
        x244 = self.p.theta3z * w_3z
        x245 = self.p.theta3y * w_3y
        x246 = self.p.l * x243
        x247 = self.p.l * x236
        x248 = self.p.l * x238
        x249 = self.p.r2 * x203
        x250 = self.p.r2 * x208
        x251 = self.p.r2 * x236
        x252 = x205 * x62 - x249 - x250 - x251
        x253 = self.p.r2 * x218
        x254 = self.p.r2 * x223
        x255 = self.p.r2 * x238
        x256 = x220 * x62 + x253 + x254 + x255
        x257 = x114 * x253 + x114 * x254 + x114 * x255 + x116 * \
            x249 + x116 * x250 + x116 * x251 + x219 * x52 - x242 * x48
        x258 = self.p.theta3x * w_3x
        x259 = 1 / self.p.tau
        x260 = 2 * phi_x
        x261 = w_2z * x80
        x262 = cos(phi_mx)
        x263 = cos(phi_my)
        x264 = sin(phi_mx)
        x265 = 1 / x262
        x266 = 1 / (x262 * x263 + x263 * x264**2 * x265)
        x267 = x261 * x66
        x268 = x264 * x265
        x269 = -w_2x * x78 - w_2y * x74 + w_3y - x261 * x70
        x270 = -w_2x * x182 - w_2y * x183 + w_3z - x267 + x268 * x269
        x271 = x266 * x270
        x272 = x263 * x264 * x271
        x273 = x269 - x272
        x274 = 1.0 * w_2z * x69
        x275 = 1.0 * omega_y_cmd * x259
        x276 = w_2x * w_3x * x71
        x277 = 1.0 * w_2x * x67 * x80
        x278 = w_2y * w_3x * x67
        x279 = 1.0 * w_2y * x71 * x80
        x280 = 1.0 * w_3x
        x281 = w_2x * w_3x
        x282 = w_2y * w_3x
        x283 = 1.0 * x259
        x284 = x273 * x283
        x285 = x262 * x263 * x266 * x270
        x286 = sin(phi_my)
        x287 = -1.0 * w_2x * x67 * x80 - 1.0 * w_2y * x71 * x80 + 1.0 * x266 * x270 * x286 + x274 + x280
        x288 = x271 * x286 * (-1.0 * w_2x * x78 - 1.0 * w_2y * x74 - 1.0 *
                              w_2z * x70 * x80 + 1.0 * w_3y - 1.0 * x263 * x264 * x266 * x270)
        x289 = 1.0 * x66
        A[0, 0] = self.p.theta1 + x1 * x11 + x1 * x12 + x1 * \
            x13 + x11 * x8 + x12 * x8 + x13 * x8 + x6 + x9
        A[0, 1] = x35
        A[0, 2] = x40
        A[0, 3] = -x42 - x44 - x46 - x48
        A[0, 4] = x49 + x50 + x51 + x52
        A[0, 5] = x54 + x55 + x57 + x58 + x60 + x61 + x63 + x64
        A[0, 6] = x79
        A[0, 7] = x83
        A[0, 8] = 0
        A[1, 0] = x35
        A[1, 1] = self.p.m1 * x92 + self.p.m2 * x93 + self.p.m2 * x94 + self.p.m3 * \
            x93 + self.p.m3 * x94 + x4 * x85 + x4 * x88 + x87 * x92 + x89 * x90 + x90 * x91
        A[1, 2] = x98
        A[1, 3] = -x101 - x103 + x105 + x106
        A[1, 4] = -x107 - x108 - x110 - x111
        A[1, 5] = x112 + x113 - x115 - x117 - x118 - x119 - x120 - x121
        A[1, 6] = x126
        A[1, 7] = x128
        A[1, 8] = 0
        A[2, 0] = x40
        A[2, 1] = x98
        A[2, 2] = self.p.m2 * x129 + self.p.m3 * x129 + x130 * x89 + x130 * x91 + x85 + x88
        A[2, 3] = 0
        A[2, 4] = -x100 - x102 - x132 - x133
        A[2, 5] = -x134 - x135 - x136 - x137
        A[2, 6] = x140
        A[2, 7] = x141
        A[2, 8] = 0
        A[3, 0] = -x142 * x82 - x143 * x81 + x144 * x69 + x79
        A[3, 1] = x126 - x145 * x82 - x146 * x81 + x147 * x69
        A[3, 2] = x140 - x148 * x81 + x149 * x69
        A[3, 3] = x151 - x154 * x82 + x159 * x69
        A[3, 4] = -x154 * x81 - x160 + x161 * x69
        A[3, 5] = -x159 * x82 - x161 * x81 - x162 - x163 + x168 * x69
        A[3, 6] = self.p.theta3x + x171 + x172 * x174 + x175 * x74**2 + x175 * x78**2 + x176 * x69
        A[3, 7] = x179 * x69 + x180
        A[3, 8] = 0
        A[4, 0] = -x142 * x78 - x143 * x74 - x144 * x181 + x83
        A[4, 1] = x128 - x145 * x78 - x146 * x74 - x147 * x181
        A[4, 2] = x141 - x148 * x74 - x149 * x181
        A[4, 3] = -x154 * x78 - x159 * x181 - x170
        A[4, 4] = -x154 * x74 - x161 * x181 + x169
        A[4, 5] = -x159 * x78 - x161 * x74 - x168 * x181 + x177 + x178
        A[4, 6] = -x176 * x181 + x180
        A[4, 7] = self.p.theta3y + x171 + x174 * x67**2 + \
            x174 * x71**2 + x175 * x69**2 - x179 * x181
        A[4, 8] = 0
        A[5, 0] = -x142 * x182 - x143 * x183 - x144 * x184
        A[5, 1] = -x145 * x182 - x146 * x183 - x147 * x184
        A[5, 2] = -x148 * x183 - x149 * x184
        A[5, 3] = -x154 * x182 - x159 * x184
        A[5, 4] = -x154 * x183 - x161 * x184
        A[5, 5] = -x159 * x182 - x161 * x183 - x168 * x184
        A[5, 6] = x151 * x182 - x160 * x183 - x176 * x184
        A[5, 7] = x169 * x183 - x170 * x182 - x179 * x184
        A[5, 8] = self.p.theta3z
        A[6, 0] = 0
        A[6, 1] = 0
        A[6, 2] = 0
        A[6, 3] = x82
        A[6, 4] = x81
        A[6, 5] = -x69
        A[6, 6] = -1
        A[6, 7] = 0
        A[6, 8] = 0
        A[7, 0] = 0
        A[7, 1] = 0
        A[7, 2] = 0
        A[7, 3] = -x185 + x187
        A[7, 4] = x188 + x190
        A[7, 5] = x191 * x80
        A[7, 6] = 0
        A[7, 7] = -1
        A[7, 8] = 0
        A[8, 0] = 0
        A[8, 1] = 0
        A[8, 2] = 0
        A[8, 3] = x189 + x192
        A[8, 4] = -x186 + x193
        A[8, 5] = 1.0 * x184
        A[8, 6] = 0
        A[8, 7] = 0
        A[8, 8] = -1
        b[0] = x114 * x220 - x116 * x205 + x203 * x24 + x208 * \
            x24 + x218 * x34 + x223 * x34 + x236 * x24 + x238 * x34
        b[1] = -self.p.theta1 * x14 * x16 * x19 * x242 + x123 * x241 + x123 * x243 + x203 * \
            x27 - x208 * x28 - x218 * x22 - x219 * x22 * x29 - x223 * x33 - x236 * x28 - x238 * x33
        b[2] = x138 * x241 + x138 * x243 - x14 * x218 - x17 * x220 - x223 * x38 - x238 * x38
        b[3] = -w_3y * x244 + w_3z * x245 - x181 * x246 - x247 * \
            x74 - x248 * x78 + x252 * x82 + x256 * x81 - x257 * x69
        b[4] = w_3x * x244 - w_3z * x258 + x181 * x257 - x246 * \
            x69 + x247 * x81 + x248 * x82 + x252 * x78 + x256 * x74
        b[5] = -w_3x * x245 + w_3y * x258 + x182 * x252 + x183 * x256 + x184 * x257
        b[6] = -x225 * x259 * (-self.p.tau * x263 * x265 * x271 * x273 * x66 - self.p.tau * x66 * (w_2x * x71 - w_2y * x67) * (w_3z * x66 + x224) - self.p.tau * (
            w_2x * x237 + w_2y * x233 + x261) * (w_3y * cos(x260) + w_3y - w_3z * sin(x260)) / 2 + x66 * (omega_x_cmd + w_2x * x82 + w_2y * x81 - w_2z * x69 - w_3x))
        b[7] = -w_3z * x274 + w_3z * x277 + w_3z * x279 - x191 * x276 + x191 * x278 - x192 * x281 - x193 * \
            x282 - x262 * x275 - x267 * x280 + x268 * x273 * x287 + x268 * x288 + x272 * x283 + x284 - x285 * x287
        b[8] = w_2z * w_3x * x191 * x80 + w_3y * x274 - w_3y * x277 - w_3y * x279 + x187 * x281 + x190 * x282 + \
            x264 * x275 - x268 * x284 + x272 * x287 + x273 * x287 - x276 * x289 + x278 * x289 + x283 * x285 + x288

        omega_dot = np.linalg.solve(A, b)

        return omega_dot

    def _compute_r_OSi(self, state):
        """computes center of mass locations of all bodies

        args:
            state (ModelState): current state

        Returns: list of x/y/z coordinates of center of mass of lower ball, upper ball, and lever arm.
        """
        [pos_x, pos_y] = state.pos
        [psi_x, psi_y] = state.psi
        [phi_x, phi_y, phi_z] = state.phi

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
            center (numpy.ndarray): center of the ball [m]
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
