"""Dynamic model of 3D version of Double Ball Balancer

This module contains the class DynamicModel for simulating the non-linear dynamics of the 3D Double Ball Balancer and a corresponding class ModelParam containing the relevant parameters.

author: Christof Dubs
"""
import numpy as np
from numpy import sin, cos, tan
from scipy.integrate import odeint
from rotation import Quaternion, quat_from_angle_vector


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
PHI_X_IDX = 10
PHI_Y_IDX = 11
OMEGA_1_Z_IDX = 12
PSI_X_DOT_IDX = 13
PSI_Y_DOT_IDX = 14
OMEGA_2_X_IDX = 15
OMEGA_2_Y_IDX = 16
OMEGA_2_Z_IDX = 17
PHI_X_DOT_IDX = 18
PHI_Y_DOT_IDX = 19
X_IDX = 20
Y_IDX = 21
STATE_SIZE = 22


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
    def phi_x(self):
        return self.x[PHI_X_IDX]

    @phi_x.setter
    def phi_x(self, value):
        self.x[PHI_X_IDX] = value

    @property
    def phi_x_dot(self):
        return self.x[PHI_X_DOT_IDX]

    @phi_x_dot.setter
    def phi_x_dot(self, value):
        self.x[PHI_X_DOT_IDX] = value

    @property
    def phi_y(self):
        return self.x[PHI_Y_IDX]

    @phi_y.setter
    def phi_y(self, value):
        self.x[PHI_Y_IDX] = value

    @property
    def phi_y_dot(self):
        return self.x[PHI_Y_DOT_IDX]

    @phi_y_dot.setter
    def phi_y_dot(self, value):
        self.x[PHI_Y_DOT_IDX] = value

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
        return Quaternion(self.q2) * quat_from_angle_vector(
            np.array([0, self.phi_y, 0])) * quat_from_angle_vector(np.array([self.phi_x, 0, 0]))

    @property
    def beta(self):
        return Quaternion(self.q2).get_roll_pitch_yaw()

    @property
    def phi(self):
        return self.x[PHI_X_IDX:PHI_Y_IDX + 1]

    @phi.setter
    def phi(self, value):
        self.x[PHI_X_IDX:PHI_Y_IDX + 1] = value

    @property
    def phi_dot(self):
        return self.x[PHI_X_DOT_IDX:PHI_Y_DOT_IDX + 1]

    @phi_dot.setter
    def phi_dot(self, value):
        self.x[PHI_X_DOT_IDX:PHI_Y_DOT_IDX + 1] = value

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
        return self.x[OMEGA_1_Z_IDX:PHI_Y_DOT_IDX + 1]

    @omega.setter
    def omega(self, value):
        self.x[OMEGA_1_Z_IDX:PHI_Y_DOT_IDX + 1] = value

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

        xdot.q2 = Quaternion(eval_state.q2).q_dot(eval_state.omega_2, frame='body')

        xdot.phi = eval_state.phi_dot
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
        [beta_x, beta_y, beta_z] = state.beta
        [psi_x, psi_y] = state.psi
        [psi_x_dot, psi_y_dot] = state.psi_dot
        w_1z = state.omega_1_z
        [w_2x, w_2y, w_2z] = state.omega_2

        omega_1 = np.zeros(3)

        x0 = 1 / self.p.r1
        x1 = tan(psi_y)
        x2 = self.p.r1 * x1
        x3 = 1 / cos(psi_y)
        x4 = psi_x_dot * x3
        x5 = sin(beta_z)
        x6 = cos(beta_x)
        x7 = self.p.r2 * w_2y * x6
        x8 = self.p.r2 * x1
        x9 = sin(beta_y)
        x10 = w_2x * x9
        x11 = cos(beta_z)
        x12 = cos(beta_y)
        x13 = self.p.r2 * w_2x * x12
        x14 = sin(beta_x)
        x15 = self.p.r2 * w_2z * x14
        x16 = w_2y * x14
        x17 = self.p.r2 * x1 * x12
        x18 = w_2z * x6
        x19 = self.p.r2 * w_2y * x14 * x9
        x20 = self.p.r2 * w_2z * x6 * x9
        x21 = tan(psi_x)
        x22 = psi_x_dot * x21
        x23 = self.p.r2 * x12 * x21 * x3
        omega_1[0] = x0 * (self.p.r1 * x4 + self.p.r2 * x4 + w_1z * x2 - x10 * x8 - \
                           x11 * x13 - x11 * x19 - x11 * x20 - x15 * x5 + x16 * x17 + x17 * x18 + x5 * x7)
        omega_1[1] = x0 * (psi_y_dot * self.p.r1 + psi_y_dot * self.p.r2 - self.p.r1 * w_1z * x21 * x3 + self.p.r2 * x10 *
                           x21 * x3 + x11 * x15 - x11 * x7 - x13 * x5 - x16 * x23 - x18 * x23 - x19 * x5 - x2 * x22 - x20 * x5 - x22 * x8)
        omega_1[2] = w_1z

        return omega_1

    def _compute_omega_dot(self, state, omega_cmd):
        """computes angular acceleration matrix of rotational part of system dynamics (equal to jacobian matrix since dynamics are linear in angular accelerations)

        The non-linear rotational dynamics are of the form

        A * [omega_1_z_dot, psi_x_ddot, psi_y_ddot, omega_2_x_dot, omega_2_y_dot, omega_2_z_dot, phi_x_dot, phi_y_dot] = b

        where A = A(phi_x, phi_y, phi_x_dot, phi_y_dot, psi_x, psi_y) and b(state, inputs).

        args:
            state (ModelState): current state
            omega_cmd (np.ndarray): motor speed commands [rad/s]

        Returns: array containing the time derivative of the angular velocity state [rad/s^2]
        """
        [beta_x, beta_y, beta_z] = state.beta
        [phi_x, phi_y] = state.phi
        [phi_x_dot, phi_y_dot] = state.phi_dot
        [psi_x, psi_y] = state.psi
        [psi_x_dot, psi_y_dot] = state.psi_dot
        w_1z = state.omega_1_z
        [w_2x, w_2y, w_2z] = state.omega_2
        [omega_x_cmd, omega_y_cmd] = omega_cmd

        A = np.zeros([8, 8])
        b = np.zeros(8)

        # auto-generated symbolic expressions
        x0 = tan(psi_y)
        x1 = x0**2
        x2 = self.p.r1**2
        x3 = x1 * x2
        x4 = cos(psi_y)
        x5 = x4**2
        x6 = 1 / x5
        x7 = tan(psi_x)
        x8 = x7**2
        x9 = x2 * x6 * x8
        x10 = 1 / self.p.r1
        x11 = self.p.theta1 * x0 * x10
        x12 = 1 / x4
        x13 = self.p.r1 * x12
        x14 = self.p.r2 * x12
        x15 = x13 + x14
        x16 = self.p.m1 * self.p.r1 * x0
        x17 = -x13 - x14
        x18 = self.p.m1 * self.p.r1 * x12
        x19 = x0 * x7
        x20 = -self.p.r1 * x19 - self.p.r2 * x19
        x21 = x20 * x7
        x22 = self.p.theta1 * x10 * x12
        x23 = cos(psi_x)
        x24 = self.p.r1 + self.p.r2
        x25 = x23 * x24
        x26 = x17 - x25
        x27 = self.p.r1 * x0 * x26
        x28 = self.p.m2 * self.p.r1 * x12
        x29 = sin(psi_x)
        x30 = sin(psi_y)
        x31 = x24 * x29 * x30
        x32 = x20 - x31
        x33 = x32 * x7
        x34 = -self.p.m2 * x27 - self.p.m3 * self.p.r1 * x12 * x33 - self.p.m3 * \
            x27 + x11 * x15 - x16 * x17 - x18 * x21 - x21 * x22 - x28 * x33
        x35 = x24 * x7
        x36 = x25 * x4
        x37 = x24 + x36
        x38 = self.p.r1 * x12 * x37 * x7
        x39 = -self.p.m2 * x38 - self.p.m3 * x38 - x18 * x35 - x22 * x35
        x40 = sin(beta_y)
        x41 = self.p.r2 * x40
        x42 = x0 * x41
        x43 = cos(beta_z)
        x44 = cos(beta_y)
        x45 = self.p.r2 * x44
        x46 = x43 * x45
        x47 = x42 + x46
        x48 = self.p.r1 * x0 * x47
        x49 = -x42 - x46
        x50 = sin(beta_z)
        x51 = x45 * x50
        x52 = self.p.r2 * x12 * x7
        x53 = x40 * x52
        x54 = -x51 + x53
        x55 = x54 * x7
        x56 = self.p.m3 * self.p.r1 * x0
        x57 = sin(phi_x)
        x58 = sin(phi_y)
        x59 = x57 * x58
        x60 = cos(phi_y)
        x61 = x44 * x60
        x62 = x50 * x61
        x63 = sin(beta_x)
        x64 = x43 * x63
        x65 = cos(beta_x)
        x66 = x50 * x65
        x67 = x40 * x66
        x68 = -x64 + x67
        x69 = x58 * x68
        x70 = x62 - x69
        x71 = self.p.l * x70
        x72 = x44 * x57
        x73 = x58 * x72
        x74 = x50 * x73
        x75 = cos(phi_x)
        x76 = x43 * x65
        x77 = x50 * x63
        x78 = x40 * x77
        x79 = x76 + x78
        x80 = x60 * x68
        x81 = x57 * x80 + x74 + x75 * x79
        x82 = self.p.l * x81
        x83 = x47 - x59 * x71 + x60 * x82
        x84 = self.p.m3 * self.p.r1 * x12 * x7
        x85 = x43 * x61
        x86 = x40 * x76 + x77
        x87 = x58 * x86
        x88 = x85 - x87
        x89 = self.p.l * x88
        x90 = x40 * x64
        x91 = -x66 + x90
        x92 = x60 * x86
        x93 = x43 * x73 + x57 * x92 + x75 * x91
        x94 = self.p.l * x93
        x95 = x54 - x59 * x89 + x60 * x94
        x96 = -self.p.m1 * x48 - self.p.m2 * x48 + x11 * x49 - \
            x18 * x55 - x22 * x55 - x28 * x55 - x56 * x83 - x84 * x95
        x97 = self.p.r2 * x66
        x98 = self.p.r2 * x64
        x99 = x40 * x98
        x100 = self.p.r2 * x0 * x44
        x101 = x100 * x63
        x102 = -x101 - x97 + x99
        x103 = self.p.m2 * self.p.r1 * x0
        x104 = x101 + x97 - x99
        x105 = self.p.r2 * x76
        x106 = self.p.r2 * x77
        x107 = self.p.r2 * x12 * x44 * x7
        x108 = x107 * x63
        x109 = -x105 - x106 * x40 - x108
        x110 = x109 * x7
        x111 = self.p.l * x75
        x112 = x111 * x70
        x113 = x102 - x112
        x114 = x111 * x88
        x115 = x109 - x114
        x116 = -x102 * x103 - x102 * x16 + x104 * x11 - x110 * \
            x18 - x110 * x22 - x110 * x28 - x113 * x56 - x115 * x84
        x117 = x41 * x76
        x118 = x44 * x65
        x119 = self.p.r2 * x0 * x118
        x120 = x106 + x117 - x119
        x121 = -x106 - x117 + x119
        x122 = x41 * x66
        x123 = x118 * x52
        x124 = -x122 - x123 + x98
        x125 = x124 * x7
        x126 = x57 * x60
        x127 = x120 - x126 * x71 - x58 * x82
        x128 = x124 - x126 * x89 - x58 * x94
        x129 = -x103 * x120 + x11 * x121 - x120 * x16 - x125 * \
            x18 - x125 * x22 - x125 * x28 - x127 * x56 - x128 * x84
        x130 = x24**2
        x131 = self.p.m2 * x130
        x132 = x29**2 * x5
        x133 = self.p.m3 * x130
        x134 = x20**2
        x135 = 1 / x2
        x136 = self.p.theta1 * x135
        x137 = x26**2
        x138 = x32**2
        x139 = x23 * x29 * x30 * x4
        x140 = self.p.m1 * x24
        x141 = self.p.theta1 * x135 * x24
        x142 = self.p.m2 * x37
        x143 = self.p.m3 * x37
        x144 = x131 * x139 + x133 * x139 + x140 * x20 + x141 * x20 + x142 * x32 + x143 * x32
        x145 = self.p.m1 * x47
        x146 = self.p.theta1 * x135 * x15
        x147 = x20 * x54
        x148 = self.p.m2 * x47
        x149 = self.p.m2 * x54
        x150 = x24 * x29 * x4
        x151 = x40 * x60
        x152 = x118 * x58
        x153 = -x151 - x152
        x154 = self.p.l * x153
        x155 = x63 * x75
        x156 = x40 * x58
        x157 = x156 * x57
        x158 = x118 * x60
        x159 = x158 * x57
        x160 = x155 * x44 - x157 + x159
        x161 = self.p.l * x160
        x162 = -x154 * x59 + x161 * x60
        x163 = self.p.m3 * x162
        x164 = self.p.m3 * x26
        x165 = self.p.m3 * x32
        x166 = self.p.m1 * x147 + x136 * x147 + x145 * x17 + x146 * x49 + \
            x148 * x26 + x149 * x32 - x150 * x163 + x164 * x83 + x165 * x95
        x167 = self.p.l * self.p.m3 * x153 * x75
        x168 = x150 * x167
        x169 = self.p.m1 * x17
        x170 = self.p.m1 * x20
        x171 = self.p.m2 * x26
        x172 = self.p.theta1 * x135 * x20
        x173 = self.p.m2 * x32
        x174 = x102 * x169 + x102 * x171 + x104 * x146 + x109 * x170 + \
            x109 * x172 + x109 * x173 + x113 * x164 + x115 * x165 + x168
        x175 = -x126 * x154 - x161 * x58
        x176 = self.p.m3 * x175
        x177 = x120 * x169 + x120 * x171 + x121 * x146 + x124 * x170 + \
            x124 * x172 + x124 * x173 + x127 * x164 + x128 * x165 - x150 * x176
        x178 = self.p.l * self.p.m3 * x160
        x179 = x37**2
        x180 = x23**2 * x30**2
        x181 = x23 * x24 * x30
        x182 = x140 * x54 + x141 * x54 + x142 * x54 + x143 * x95 - x163 * x181
        x183 = x167 * x181
        x184 = x109 * x140 + x109 * x141 + x109 * x142 + x115 * x143 + x183
        x185 = x124 * x140 + x124 * x141 + x124 * x142 + x128 * x143 - x176 * x181
        x186 = x60**2
        x187 = x58**2
        x188 = x57**2
        x189 = self.p.theta3y * x188
        x190 = x75**2
        x191 = self.p.theta3z * x190
        x192 = x47**2
        x193 = x54**2
        x194 = self.p.theta3y * x57 * x75
        x195 = self.p.theta3z * x57 * x75
        x196 = -x162 * x167 + x194 * x58 - x195 * x58
        x197 = self.p.theta1 * x135 * x49
        x198 = self.p.m1 * x54
        x199 = self.p.theta1 * x135 * x54
        x200 = self.p.m3 * x113
        x201 = self.p.m3 * x115
        x202 = x102 * x145 + x102 * x148 + x104 * x197 + x109 * x149 + \
            x109 * x198 + x109 * x199 + x196 + x200 * x83 + x201 * x95
        x203 = self.p.theta3x * x58
        x204 = x58 * x60
        x205 = self.p.m3 * x127 * x83 + self.p.m3 * x128 * x95 + x120 * x145 + x120 * x148 + x121 * x197 + \
            x124 * x149 + x124 * x198 + x124 * x199 + x163 * x175 + x189 * x204 + x191 * x204 - x203 * x60
        x206 = self.p.l * self.p.m3 * x81
        x207 = self.p.l * self.p.m3 * x93
        x208 = self.p.l * self.p.m3 * x70 * x75
        x209 = self.p.l * self.p.m3 * x75 * x88
        x210 = self.p.l**2 * self.p.m3
        x211 = self.p.theta3y * x190 + self.p.theta3z * x188 + x153**2 * x190 * x210
        x212 = x102**2
        x213 = x109**2
        x214 = -x167 * x175 + x194 * x60 - x195 * x60
        x215 = x102 * x120
        x216 = x109 * x124
        x217 = self.p.m1 * x215 + self.p.m1 * x216 + self.p.m2 * x215 + self.p.m2 * \
            x216 + x104 * x121 * x136 + x127 * x200 + x128 * x201 + x136 * x216 + x214
        x218 = x120**2
        x219 = x124**2
        x220 = self.p.r1 * x0
        x221 = x1 + 1
        x222 = self.p.r1 * x221
        x223 = w_1z * x222
        x224 = psi_x_dot * x30 * x6
        x225 = self.p.r1 * x224
        x226 = self.p.r2 * x224
        x227 = self.p.r2 * w_2x * x40
        x228 = x221 * x227
        x229 = self.p.r2 * w_2y * x44 * x63
        x230 = x221 * x229
        x231 = self.p.r2 * w_2z * x44 * x65
        x232 = x221 * x231
        x233 = psi_y_dot * (-x223 - x225 - x226 + x228 - x230 - x232)
        x234 = w_2y * x63
        x235 = 1 / x65
        x236 = w_2z + x234 * x235
        x237 = 1 / (x118 + x235 * x44 * x63**2)
        x238 = self.p.m1 * x236 * x237
        x239 = w_2z * x98
        x240 = w_2x * x51
        x241 = w_2y * x105
        x242 = w_2y * x106
        x243 = x242 * x40
        x244 = w_2z * x122
        x245 = x239 - x240 - x241 - x243 - x244
        x246 = x236 * x237
        x247 = w_2x + x246 * x40
        x248 = w_2z * x97
        x249 = w_2y * x117
        x250 = w_2z * x101
        x251 = w_2y * x119
        x252 = x239 * x40
        x253 = x242 + x248 + x249 + x250 - x251 - x252
        x254 = x247 * x253
        x255 = w_2y - x246 * x44 * x63
        x256 = w_2x * x100
        x257 = x227 * x43
        x258 = x234 * x42
        x259 = w_2y * x98
        x260 = x259 * x44
        x261 = w_2z * x65
        x262 = x261 * x42
        x263 = x231 * x43
        x264 = x256 - x257 + x258 + x260 + x262 + x263
        x265 = x235 * x255 * x264
        x266 = self.p.m1 * x233 + self.p.m1 * x254 + self.p.m1 * x265 + x238 * x245
        x267 = psi_y_dot * x10
        x268 = x10 * x236 * x237
        x269 = x10 * x247
        x270 = x10 * x235 * x255
        x271 = x267 * (x223 + x225 + x226 - x228 + x230 + x232) + x268 * (-x239 + x240 + x241 + x243 + x244) + \
            x269 * (-x242 - x248 - x249 - x250 + x251 + x252) + x270 * (-x256 + x257 - x258 - x260 - x262 - x263)
        x272 = psi_x_dot**2 * x24 * x29
        x273 = self.p.m2 * x236 * x237
        x274 = self.p.m2 * x233 + self.p.m2 * x254 + self.p.m2 * x265 + self.p.m2 * x272 + x245 * x273
        x275 = self.p.r1 * x12 * x7
        x276 = w_2z * x106
        x277 = -w_2x * x46 + w_2y * x97 - w_2z * x117 - x259 * x40 - x276
        x278 = psi_x_dot * x7
        x279 = x30 * x6 * x7
        x280 = -self.p.r1 * w_1z * x279 - self.p.r2 * x221 * x278 - \
            x222 * x278 + x227 * x279 - x229 * x279 - x231 * x279
        x281 = x8 + 1
        x282 = psi_x_dot * x0 * x281
        x283 = self.p.r2 * x12 * x281
        x284 = -self.p.r1 * x282 - self.p.r2 * x282 - w_1z * x13 * x281 + \
            w_2x * x283 * x40 - w_2y * x283 * x44 * x63 - w_2z * x283 * x44 * x65
        x285 = psi_x_dot * x284
        x286 = -w_2y * x122 - w_2y * x123 + w_2z * x105 + w_2z * x108 + x259 + x276 * x40
        x287 = x247 * x286
        x288 = w_2x * x107 + x227 * x50 + x234 * x53 - x242 * x44 - x248 * x44 + x261 * x53
        x289 = x235 * x255 * x288
        x290 = psi_y_dot * self.p.m1 * x280 + self.p.m1 * x285 + \
            self.p.m1 * x287 + self.p.m1 * x289 + x238 * x277
        x291 = x10 * x285 + x267 * x280 + x268 * x277 + x269 * x286 + x270 * x288
        x292 = psi_y_dot * (-psi_x_dot * x150 - psi_y_dot * x181 + x280)
        x293 = psi_x_dot * (-psi_x_dot * x181 - psi_y_dot * x150 + x284)
        x294 = self.p.m2 * x287 + self.p.m2 * x289 + self.p.m2 * x292 + self.p.m2 * x293 + x273 * x277
        x295 = phi_x_dot * self.p.m3
        x296 = w_2x * x58
        x297 = w_2z * x60
        x298 = -phi_y_dot * x57 - w_2y * x57 + x296 * x75 + x297 * x75
        x299 = w_2x * x60
        x300 = w_2z * x58
        x301 = phi_x_dot + x299 - x300
        x302 = self.p.l * x301
        x303 = x44 * x58
        x304 = x303 * x50
        x305 = x68 * x75
        x306 = self.p.m3 * x247
        x307 = -x76 - x78
        x308 = phi_y_dot * x75
        x309 = w_2y * x75
        x310 = x296 * x57
        x311 = x297 * x57
        x312 = x308 + x309 + x310 + x311
        x313 = self.p.l * x312 * x58
        x314 = self.p.m3 * x235 * x255
        x315 = self.p.l * x312
        x316 = x44 * x75
        x317 = x44 * x57 * x60
        x318 = self.p.m3 * x236 * x237
        x319 = phi_y_dot * self.p.m3
        x320 = x299 * x57 - x300 * x57
        x321 = -x296 - x297
        x322 = self.p.m3 * x233 + self.p.m3 * x272 + x295 * (-x298 * x71 + x302 * (x304 * x75 + x305 * x60 - x57 * x79)) + x306 * (x253 + x302 * (x126 * x307 + x305) + x307 * x313) + x314 * (
            x264 + x302 * (-x157 * x50 + x316 * x77 + x317 * x66) - x315 * (-x151 * x50 - x303 * x66)) + x318 * (x245 + x302 * x93 - x315 * x88) + x319 * (x302 * (x57 * x62 - x57 * x69) - x315 * (-x304 - x80) - x320 * x71 + x321 * x82)
        x323 = x303 * x43
        x324 = x75 * x86
        x325 = x66 - x90
        x326 = x64 - x67
        x327 = self.p.m3 * x292 + self.p.m3 * x293 + x295 * (-x298 * x89 + x302 * (x323 * x75 + x324 * x60 - x57 * x91)) + x306 * (x286 + x302 * (x126 * x325 + x324) + x313 * x325) + x314 * (x288 + x302 * (-x157 * x43 + x159 * x43 + \
                                                             x316 * x64) - x315 * (-x151 * x43 - x152 * x43)) + x318 * (x277 + x302 * (x126 * x326 + x307 * x75 - x74) - x315 * (-x326 * x58 - x62)) + x319 * (x302 * (x57 * x85 - x57 * x87) - x315 * (-x323 - x92) - x320 * x89 + x321 * x94)
        x328 = psi_x_dot * (-psi_x_dot * x36 + psi_y_dot * x31)
        x329 = psi_y_dot * (psi_x_dot * x31 - psi_y_dot * x36)
        x330 = self.p.g * self.p.m2 + self.p.m2 * x328 + self.p.m2 * x329
        x331 = self.p.theta1 * x10 * x271
        x332 = self.p.theta1 * x10 * x291
        x333 = x118 * x75
        x334 = x151 * x57
        x335 = self.p.g * self.p.m3 + self.p.m3 * x328 + self.p.m3 * x329 + x295 * (-x154 * x298 + x302 * (-x156 * x75 + x333 * x60 - x63 * x72)) + x306 * (x302 * (-x317 * x63 + x333) - x315 * x44 * x58 * x63) + x314 * (
            x302 * (-x155 * x40 - x334 * x65 - x73) - x315 * (x156 * x65 - x61)) + x319 * (-x154 * x320 + x161 * x321 + x302 * (-x152 * x57 - x334) - x315 * (x156 - x158))
        x336 = x298 * x312
        x337 = phi_y_dot * self.p.theta3x * x321 - self.p.theta3y * x336 + self.p.theta3z * x336
        x338 = x301 * x312
        x339 = -self.p.theta3x * x338 + self.p.theta3y * x338 + self.p.theta3z * \
            (phi_x_dot * (-x308 - x309 - x310 - x311) + phi_y_dot * (x299 * x75 - x300 * x75))
        x340 = x339 * x75
        x341 = x298 * x301
        x342 = self.p.theta3x * x341 + self.p.theta3y * \
            (phi_x_dot * x298 + phi_y_dot * x320) - self.p.theta3z * x341
        x343 = 1 / self.p.tau
        A[0, 0] = self.p.m1 * x3 + self.p.m1 * x9 + self.p.m2 * x3 + self.p.m2 * x9 + self.p.m3 * \
            x3 + self.p.m3 * x9 + self.p.theta1 * x1 + self.p.theta1 * x6 * x8 + self.p.theta1
        A[0, 1] = x34
        A[0, 2] = x39
        A[0, 3] = x96
        A[0, 4] = x116
        A[0, 5] = x129
        A[0, 6] = -x56 * x82 - x84 * x94
        A[0, 7] = x112 * x56 + x114 * x84
        A[1, 0] = x34
        A[1, 1] = self.p.m1 * x134 + self.p.m1 * x17**2 + self.p.m2 * x137 + self.p.m2 * x138 + \
            self.p.m3 * x137 + self.p.m3 * x138 + x131 * x132 + x132 * x133 + x134 * x136 + x136 * x15**2
        A[1, 2] = x144
        A[1, 3] = x166
        A[1, 4] = x174
        A[1, 5] = x177
        A[1, 6] = -x150 * x178 + x164 * x82 + x165 * x94
        A[1, 7] = -x112 * x164 - x114 * x165 + x168
        A[2, 0] = x39
        A[2, 1] = x144
        A[2, 2] = self.p.m1 * x130 + self.p.m2 * x179 + self.p.m3 * \
            x179 + x130 * x136 + x131 * x180 + x133 * x180
        A[2, 3] = x182
        A[2, 4] = x184
        A[2, 5] = x185
        A[2, 6] = x143 * x94 - x178 * x181
        A[2, 7] = -x114 * x143 + x183
        A[3, 0] = x96
        A[3, 1] = x166
        A[3, 2] = x182
        A[3, 3] = self.p.m1 * x192 + self.p.m1 * x193 + self.p.m2 * x192 + self.p.m2 * x193 + self.p.m3 * x162**2 + self.p.m3 * \
            x83**2 + self.p.m3 * x95**2 + self.p.theta2 + self.p.theta3x * x186 + x136 * x193 + x136 * x49**2 + x187 * x189 + x187 * x191
        A[3, 4] = x202
        A[3, 5] = x205
        A[3, 6] = self.p.theta3x * x60 + x162 * x178 + x206 * x83 + x207 * x95
        A[3, 7] = x196 - x208 * x83 - x209 * x95
        A[4, 0] = x116
        A[4, 1] = x174
        A[4, 2] = x184
        A[4, 3] = x202
        A[4, 4] = self.p.m1 * x212 + self.p.m1 * x213 + self.p.m2 * x212 + self.p.m2 * x213 + \
            self.p.m3 * x113**2 + self.p.m3 * x115**2 + self.p.theta2 + x104**2 * x136 + x136 * x213 + x211
        A[4, 5] = x217
        A[4, 6] = -x153 * x160 * x210 * x75 + x200 * x82 + x201 * x94
        A[4, 7] = -x112 * x200 - x114 * x201 + x211
        A[5, 0] = x129
        A[5, 1] = x177
        A[5, 2] = x185
        A[5, 3] = x205
        A[5, 4] = x217
        A[5, 5] = self.p.m1 * x218 + self.p.m1 * x219 + self.p.m2 * x218 + self.p.m2 * x219 + self.p.m3 * x127**2 + self.p.m3 * \
            x128**2 + self.p.m3 * x175**2 + self.p.theta2 + self.p.theta3x * x187 + x121**2 * x136 + x136 * x219 + x186 * x189 + x186 * x191
        A[5, 6] = x127 * x206 + x128 * x207 + x175 * x178 - x203
        A[5, 7] = -x127 * x208 - x128 * x209 + x214
        A[6, 0] = 0
        A[6, 1] = 0
        A[6, 2] = 0
        A[6, 3] = 0
        A[6, 4] = 0
        A[6, 5] = 0
        A[6, 6] = 1
        A[6, 7] = 0
        A[7, 0] = 0
        A[7, 1] = 0
        A[7, 2] = 0
        A[7, 3] = 0
        A[7, 4] = 0
        A[7, 5] = 0
        A[7, 6] = 0
        A[7, 7] = 1
        b[0] = -self.p.theta1 * x0 * x271 + self.p.theta1 * x12 * x291 * x7 + x220 * \
            x266 + x220 * x274 + x220 * x322 + x275 * x290 + x275 * x294 + x275 * x327
        b[1] = -x15 * x331 + x150 * x330 + x150 * x335 - x17 * x266 - x20 * \
            x290 - x20 * x332 - x26 * x274 - x26 * x322 - x294 * x32 - x32 * x327
        b[2] = x181 * x330 + x181 * x335 - x24 * x290 - x24 * x332 - x294 * x37 - x327 * x37
        b[3] = -x162 * x335 - x266 * x47 - x274 * x47 - x290 * x54 - x294 * x54 - x322 * \
            x83 - x327 * x95 - x331 * x49 - x332 * x54 - x337 * x60 - x340 * x58 - x342 * x59
        b[4] = -x102 * x266 - x102 * x274 - x104 * x331 - x109 * x290 - x109 * x294 - x109 * \
            x332 - x113 * x322 - x115 * x327 + x154 * x335 * x75 + x339 * x57 - x342 * x75
        b[5] = -x120 * x266 - x120 * x274 - x121 * x331 - x124 * x290 - x124 * x294 - x124 * \
            x332 - x126 * x342 - x127 * x322 - x128 * x327 - x175 * x335 + x337 * x58 - x340 * x60
        b[6] = x343 * (omega_x_cmd - phi_x_dot)
        b[7] = x343 * (omega_y_cmd - phi_y_dot)

        omega_dot = np.linalg.solve(A, b)

        return omega_dot

    def _compute_r_OSi(self, state):
        """computes center of mass locations of all bodies

        args:
            state (ModelState): current state

        Returns: list of x/y/z coordinates of center of mass of lower ball, upper ball, and lever arm.
        """
        [beta_x, beta_y, beta_z] = state.beta
        [pos_x, pos_y] = state.pos
        [psi_x, psi_y] = state.psi
        [phi_x, phi_y] = state.phi

        r_OS1 = np.array([pos_x, pos_y, self.p.r1])

        r_S1S2 = np.zeros(3)
        r_S2S3 = np.zeros(3)

        x0 = self.p.r1 + self.p.r2
        x1 = x0 * cos(psi_x)
        x2 = cos(beta_z)
        x3 = sin(phi_y)
        x4 = cos(beta_y)
        x5 = cos(phi_x)
        x6 = x3 * x4 * x5
        x7 = sin(phi_x)
        x8 = sin(beta_z)
        x9 = cos(beta_x)
        x10 = x8 * x9
        x11 = sin(beta_y)
        x12 = sin(beta_x)
        x13 = x12 * x2
        x14 = x5 * cos(phi_y)
        x15 = x12 * x8
        x16 = x2 * x9
        r_S1S2[0] = x1 * sin(psi_y)
        r_S1S2[1] = -x0 * sin(psi_x)
        r_S1S2[2] = x1 * cos(psi_y)
        r_S2S3[0] = -self.p.l * (x14 * (x11 * x16 + x15) + x2 * x6 - x7 * (-x10 + x11 * x13))
        r_S2S3[1] = -self.p.l * (x14 * (x10 * x11 - x13) + x6 * x8 - x7 * (x11 * x15 + x16))
        r_S2S3[2] = -self.p.l * (-x11 * x3 * x5 - x12 * x4 * x7 + x14 * x4 * x9)

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
