"""Dynamic model of 3D version of Double Ball Balancer

This module contains the class DynamicModel for simulating the non-linear dynamics of the 3D Double Ball Balancer and a corresponding class ModelParam containing the relevant parameters.

author: Christof Dubs
"""
import numpy as np
from numpy import sin, cos, tan, exp, pi
from scipy.integrate import odeint
from pyrotation import Quaternion, quat_from_angle_vector


class ModelParam(object):
    """Physical parameters of 3D Double Ball Balancer

    The Double Ball Balancer consists of 3 bodies:
        1: lower ball
        2: upper ball
        3: lever arm

    Physical parameters that multiple bodies have are indexed accordingly.

    Attributes:
        a: ratio of contact area radius vs ball radius for torsional friction [-]
        g: Gravitational constant [m/s^2]
        l : Arm length of lever [m] (distance from rotation axis to center of mass)
        m1: Mass of lower ball [kg]
        m2: Mass of upper ball [kg]
        m3: Mass of lever arm [kg]
        mu1: Torsional friction coefficient between lower ball and ground [-]
        mu12: Torsional friction coefficient between lower ball and upper ball [-]
        r1: Radius of lower ball [m]
        r2: Radius of upper ball [m]
        tau: time constant of speed controlled motor [s]
        theta1: Mass moment of inertia of lower ball wrt. its center of mass around all axes [kg*m^2]
        theta2: Mass moment of inertia of upper ball wrt. its center of mass around all axes [kg*m^2]
        theta3x: Mass moment of inertia of lever arm wrt. its center of mass around x axis [kg*m^2]
        theta3y: Mass moment of inertia of lever arm wrt. its center of mass around y axis [kg*m^2]
        theta3z: Mass moment of inertia of lever arm wrt. its center of mass around z axis [kg*m^2]
    """

    def __init__(self,):
        """Initializes the parameters to default values"""
        self.a = 0.1
        self.g = 9.81
        self.l = 0.5
        self.m1 = 1.0
        self.m2 = 1.0
        self.m3 = 1.0
        self.mu1 = 0.8
        self.mu12 = 0.8
        self.r1 = 1.0
        self.r2 = 1.0
        self.tau = 0.1
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
        return self.a >= 0 and self.g > 0 and self.l > 0 and self.m1 > 0 and self.m2 > 0 and self.m3 > 0 and self.mu1 >= 0 and self.mu12 >= 0 and self.r1 > 0 and self.r2 > 0 and self.tau > 0 and self.theta1 > 0 and self.theta2 > 0 and self.theta3x > 0 and self.theta3y > 0 and self.theta3z > 0


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


class ModelState(object):
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


class DynamicModel(object):
    """Simulation interface for the 3D Double Ball Balancer

    Attributes:
        p (ModelParam): physical parameters
        state (ModelState): 22-dimensional state

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
            ModelState containing the time derivatives of all states
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

        # lambda for auto-generated sympy.Max() function
        def Max(x, y): return np.max([x, y])

        # auto-generated symbolic expressions
        x0 = self.p.r1**2
        x1 = tan(psi_y)
        x2 = x1**2
        x3 = x0 * x2
        x4 = cos(psi_y)
        x5 = x4**2
        x6 = 1 / x5
        x7 = tan(psi_x)
        x8 = x7**2
        x9 = x0 * x6 * x8
        x10 = cos(psi_x)
        x11 = x10 * x4
        x12 = Max(self.p.r1, self.p.r2)
        x13 = sin(psi_x)
        x14 = self.p.r1 * x1
        x15 = self.p.m2 * x14
        x16 = self.p.m3 * x14
        x17 = sin(psi_y)
        x18 = x10 * x17
        x19 = 1 / x4
        x20 = self.p.r1 * x19 * x7
        x21 = self.p.m2 * x20
        x22 = self.p.m3 * x20
        x23 = sin(beta_y)
        x24 = w_2x * x23
        x25 = sin(beta_x)
        x26 = cos(beta_y)
        x27 = x25 * x26
        x28 = w_2y * x27
        x29 = cos(beta_x)
        x30 = x26 * x29
        x31 = w_2z * x30
        x32 = cos(beta_z)
        x33 = w_2x * x26
        x34 = sin(beta_z)
        x35 = x25 * x34
        x36 = x29 * x32
        x37 = x23 * x36 + x35
        x38 = x29 * x34
        x39 = x25 * x32
        x40 = x23 * x39
        x41 = -x38 + x40
        x42 = 1 / self.p.r1
        x43 = self.p.r2 * x38
        x44 = w_2y * x43
        x45 = self.p.r2 * x26
        x46 = x32 * x45
        x47 = self.p.r2 * x35
        x48 = w_2z * x47
        x49 = self.p.r2 * x39
        x50 = w_2y * x49
        x51 = self.p.r2 * x36
        x52 = w_2z * x51
        x53 = -w_2x * x46 - x23 * x50 - x23 * x52 + x44 - x48
        x54 = self.p.r1 * x19
        x55 = self.p.r2 * x19
        x56 = self.p.r2 * x1
        x57 = x23 * x35
        x58 = x36 + x57
        x59 = x23 * x38
        x60 = -x39 + x59
        x61 = w_2z * x49
        x62 = x34 * x45
        x63 = w_2x * x62
        x64 = w_2y * x51
        x65 = w_2y * x47
        x66 = x23 * x65
        x67 = w_2z * x43
        x68 = x23 * x67
        x69 = x61 - x63 - x64 - x66 - x68
        x70 = x14 * x7
        x71 = x56 * x7
        x72 = self.p.r2 * x19 * x7
        x73 = -1 + 2 / (exp(x11 * (-w_1z - x24 + x28 + x31) - x13 * (w_2y * x58 + w_2z * x60 + x33 * x34 - x42 * (-psi_x_dot * x70 - psi_x_dot * x71 + psi_y_dot * self.p.r1 + psi_y_dot * self.p.r2 - w_1z *
                                                                                                                  x20 + x24 * x72 - x28 * x72 - x31 * x72 + x69)) + x18 * (w_2y * x41 + w_2z * x37 + x32 * x33 - x42 * (psi_x_dot * x54 + psi_x_dot * x55 + w_1z * x14 - x24 * x56 + x28 * x56 + x31 * x56 + x53))) + 1)
        x74 = 3 * pi * self.p.a * self.p.mu12 * x12 * x73 * \
            (-x13 * (-x15 - x16) + x18 * (-x21 - x22)) / 16
        x75 = x11 * x74
        x76 = x18 * x74
        x77 = self.p.theta1 * x1 + x76
        x78 = x19 * x7
        x79 = x13 * x74
        x80 = -self.p.theta1 * x78 - x79
        x81 = self.p.m1 * self.p.r1 * x1
        x82 = -x54 - x55
        x83 = self.p.m1 * self.p.r1 * x19 * x7
        x84 = -x70 - x71
        x85 = self.p.r1 + self.p.r2
        x86 = x10 * x85
        x87 = x82 - x86
        x88 = x13 * x17 * x85
        x89 = x84 - x88
        x90 = -x15 * x87 - x16 * x87 - x21 * x89 - x22 * x89 - x81 * x82 - x83 * x84
        x91 = -1 + 2 / (exp(w_1z) + 1)
        x92 = 3 * pi * self.p.a * self.p.mu1 * self.p.r1 * x91 / 16
        x93 = x13 * x4 * x85
        x94 = self.p.m3 * x93
        x95 = -self.p.m2 * x93 - x94
        x96 = self.p.m2 * x87
        x97 = self.p.m3 * x87
        x98 = self.p.m2 * x89
        x99 = self.p.m3 * x89
        x100 = 3 * pi * self.p.a * self.p.mu12 * x12 * x73 * \
            (x11 * x95 - x13 * (x96 + x97) + x18 * (x98 + x99)) / 16
        x101 = x100 * x11
        x102 = self.p.theta1 * x42
        x103 = x54 + x55
        x104 = x100 * x18
        x105 = x102 * x103 + x104
        x106 = x100 * x13
        x107 = x102 * x84 - x106
        x108 = self.p.m1 * x85
        x109 = x4 * x86
        x110 = x109 + x85
        x111 = -x108 * x20 - x110 * x21 - x110 * x22
        x112 = x10 * x17 * x85
        x113 = self.p.m3 * x112
        x114 = -self.p.m2 * x112 - x113
        x115 = self.p.m2 * x110
        x116 = self.p.m3 * x110
        x117 = x11 * x114 + x18 * (x115 + x116)
        x118 = 3 * pi * self.p.a * self.p.mu12 * x117 * x12 * x73 / 16
        x119 = x11 * x118
        x120 = 3 * pi * self.p.a * self.p.mu12 * x10 * x117 * x12 * x17 * x73 / 16
        x121 = x118 * x13
        x122 = x102 * x85 - x121
        x123 = x23 * x56
        x124 = x123 + x46
        x125 = x23 * x72
        x126 = x125 - x62
        x127 = sin(phi_x)
        x128 = sin(phi_y)
        x129 = x127 * x128
        x130 = cos(phi_y)
        x131 = x130 * x26
        x132 = x131 * x34
        x133 = x128 * x60
        x134 = x132 - x133
        x135 = self.p.l * x134
        x136 = x128 * x26
        x137 = x136 * x34
        x138 = x127 * x137
        x139 = cos(phi_x)
        x140 = x130 * x60
        x141 = self.p.l * (x127 * x140 + x138 + x139 * x58)
        x142 = x124 - x129 * x135 + x130 * x141
        x143 = x131 * x32
        x144 = x128 * x37
        x145 = x143 - x144
        x146 = self.p.l * x145
        x147 = x127 * x136
        x148 = x130 * x37
        x149 = x127 * x148 + x139 * x41 + x147 * x32
        x150 = self.p.l * x149
        x151 = x126 - x129 * x146 + x130 * x150
        x152 = -x124 * x15 - x124 * x81 - x126 * x21 - x126 * x83 - x142 * x16 - x151 * x22
        x153 = 3 * pi * self.p.a * self.p.m3 * self.p.mu1 * self.p.r1 * x91 / 16
        x154 = x130 * x23
        x155 = x128 * x30
        x156 = -x154 - x155
        x157 = self.p.l * x156
        x158 = x128 * x23
        x159 = x127 * x158
        x160 = x130 * x30
        x161 = x127 * x160
        x162 = x139 * x27 - x159 + x161
        x163 = self.p.l * x162
        x164 = -x129 * x157 + x130 * x163
        x165 = self.p.m3 * x10 * x4
        x166 = self.p.m2 * x124
        x167 = self.p.m3 * x142
        x168 = self.p.m2 * x126
        x169 = self.p.m3 * x151
        x170 = 3 * pi * self.p.a * self.p.mu12 * x12 * x73 * \
            (-x13 * (x166 + x167) + x164 * x165 + x18 * (x168 + x169)) / 16
        x171 = x11 * x170
        x172 = -x123 - x46
        x173 = x170 * x18
        x174 = x102 * x172 + x173
        x175 = x13 * x170
        x176 = x102 * x126 - x175
        x177 = x23 * x49
        x178 = x27 * x56
        x179 = x177 - x178 - x43
        x180 = x27 * x72
        x181 = -x180 - x23 * x47 - x51
        x182 = self.p.l * x139
        x183 = x134 * x182
        x184 = x179 - x183
        x185 = x145 * x182
        x186 = x181 - x185
        x187 = -x15 * x179 - x16 * x184 - x179 * x81 - x181 * x21 - x181 * x83 - x186 * x22
        x188 = self.p.l * x139 * x156
        x189 = x153 * x188
        x190 = -x165 * x188
        x191 = self.p.m3 * x184
        x192 = self.p.m3 * x186
        x193 = 3 * pi * self.p.a * self.p.mu12 * x12 * x73 * \
            (-x13 * (self.p.m2 * x179 + x191) + x18 * (self.p.m2 * x181 + x192) + x190) / 16
        x194 = x11 * x193
        x195 = -x177 + x178 + x43
        x196 = x18 * x193
        x197 = x102 * x195 + x196
        x198 = x13 * x193
        x199 = x102 * x181 - x198
        x200 = x23 * x51
        x201 = x30 * x56
        x202 = x200 - x201 + x47
        x203 = x30 * x72
        x204 = -x203 - x23 * x43 + x49
        x205 = x127 * x130
        x206 = -x128 * x141 - x135 * x205 + x202
        x207 = -x128 * x150 - x146 * x205 + x204
        x208 = -x15 * x202 - x16 * x206 - x202 * x81 - x204 * x21 - x204 * x83 - x207 * x22
        x209 = -x128 * x163 - x157 * x205
        x210 = self.p.m2 * x202
        x211 = self.p.m3 * x206
        x212 = self.p.m2 * x204
        x213 = self.p.m3 * x207
        x214 = 3 * pi * self.p.a * self.p.mu12 * x12 * x73 * \
            (-x13 * (x210 + x211) + x165 * x209 + x18 * (x212 + x213)) / 16
        x215 = x11 * x214
        x216 = -x200 + x201 - x47
        x217 = x18 * x214
        x218 = x102 * x216 + x217
        x219 = x13 * x214
        x220 = x102 * x204 - x219
        x221 = self.p.m3 * x13
        x222 = self.p.m3 * x10 * x17
        x223 = -x141 * x221 + x150 * x222 + x163 * x165
        x224 = 3 * pi * self.p.a * self.p.mu12 * x10 * x12 * x223 * x4 * x73 / 16
        x225 = 3 * pi * self.p.a * self.p.mu12 * x10 * x12 * x17 * x223 * x73 / 16
        x226 = 3 * pi * self.p.a * self.p.mu12 * x12 * x13 * x223 * x73 / 16
        x227 = x183 * x221 - x185 * x222 + x190
        x228 = 3 * pi * self.p.a * self.p.mu12 * x12 * x227 * x73 / 16
        x229 = x11 * x228
        x230 = 3 * pi * self.p.a * self.p.mu12 * x10 * x12 * x17 * x227 * x73 / 16
        x231 = 3 * pi * self.p.a * self.p.mu12 * x12 * x13 * x227 * x73 / 16
        x232 = x42 * x77
        x233 = x42 * x80
        x234 = x85**2
        x235 = self.p.m2 * x234
        x236 = x13**2 * x5
        x237 = self.p.m3 * x234
        x238 = x87**2
        x239 = x89**2
        x240 = x105 * x42
        x241 = x107 * x42
        x242 = x10 * x13 * x17 * x4
        x243 = x108 * x84 + x115 * x89 + x116 * x89 + x235 * x242 + x237 * x242
        x244 = 3 * pi * self.p.a * self.p.mu12 * x10 * x117 * x12 * x17 * x42 * x73 / 16
        x245 = x122 * x42
        x246 = self.p.m1 * x124
        x247 = self.p.m1 * x84
        x248 = x126 * x247 + x142 * x97 + x151 * x99 - x164 * x94 + x166 * x87 + x168 * x89 + x246 * x82
        x249 = x174 * x42
        x250 = x176 * x42
        x251 = x188 * x94
        x252 = self.p.m1 * x82
        x253 = x179 * x252 + x179 * x96 + x181 * x247 + x181 * x98 + x184 * x97 + x186 * x99 + x251
        x254 = x197 * x42
        x255 = x199 * x42
        x256 = x202 * x252 + x202 * x96 + x204 * x247 + x204 * x98 + x206 * x97 + x207 * x99 - x209 * x94
        x257 = x218 * x42
        x258 = x220 * x42
        x259 = 3 * pi * self.p.a * self.p.mu12 * x12 * x13 * x223 * x42 * x73 / 16
        x260 = 3 * pi * self.p.a * self.p.mu12 * x10 * x12 * x17 * x223 * x42 * x73 / 16
        x261 = 3 * pi * self.p.a * self.p.mu12 * x12 * x13 * x227 * x42 * x73 / 16
        x262 = 3 * pi * self.p.a * self.p.mu12 * x10 * x12 * x17 * x227 * x42 * x73 / 16
        x263 = x110**2
        x264 = x10**2 * x17**2
        x265 = x108 * x126 - x113 * x164 + x115 * x126 + x116 * x151
        x266 = x113 * x188
        x267 = x108 * x181 + x115 * x181 + x116 * x186 + x266
        x268 = x108 * x204 - x113 * x209 + x115 * x204 + x116 * x207
        x269 = x13 * x42 * x85
        x270 = x26 * x34
        x271 = x26 * x32
        x272 = x130**2
        x273 = x128**2
        x274 = x127**2
        x275 = self.p.theta3y * x274
        x276 = x139**2
        x277 = self.p.theta3z * x276
        x278 = x124**2
        x279 = x126**2
        x280 = self.p.theta3y * x127 * x139
        x281 = x128 * x280
        x282 = self.p.theta3z * x127 * x139
        x283 = -x128 * x282
        x284 = self.p.m1 * x126
        x285 = self.p.l * self.p.m3 * x139 * x156
        x286 = -x164 * x285
        x287 = x142 * x191 + x151 * x192 + x166 * x179 + x168 * \
            x181 + x179 * x246 + x181 * x284 + x281 + x283 + x286
        x288 = self.p.theta3x * x128
        x289 = x128 * x130
        x290 = self.p.m3 * x164 * x209 - x130 * x288 + x166 * x202 + x167 * x206 + x168 * \
            x204 + x169 * x207 + x202 * x246 + x204 * x284 + x275 * x289 + x277 * x289
        x291 = self.p.l * self.p.m3 * x162
        x292 = self.p.l**2 * self.p.m3
        x293 = self.p.theta3y * x276 + self.p.theta3z * x274 + x156**2 * x276 * x292
        x294 = x179**2
        x295 = x181**2
        x296 = x130 * x280
        x297 = -x130 * x282
        x298 = -x209 * x285
        x299 = self.p.m1 * x179 * x202 + self.p.m1 * x181 * x204 + x179 * \
            x210 + x181 * x212 + x191 * x206 + x192 * x207 + x296 + x297 + x298
        x300 = x202**2
        x301 = x204**2
        x302 = x2 + 1
        x303 = self.p.r1 * x302
        x304 = w_1z * x303
        x305 = psi_x_dot * x17 * x6
        x306 = self.p.r1 * x305
        x307 = self.p.r2 * x305
        x308 = self.p.r2 * w_2x * x23
        x309 = x302 * x308
        x310 = self.p.r2 * w_2y * x25 * x26
        x311 = x302 * x310
        x312 = self.p.r2 * w_2z * x26 * x29
        x313 = x302 * x312
        x314 = psi_y_dot * (-x304 - x306 - x307 + x309 - x311 - x313)
        x315 = w_2y * x25
        x316 = 1 / x29
        x317 = w_2z + x315 * x316
        x318 = 1 / (x25**2 * x26 * x316 + x30)
        x319 = self.p.m1 * x317 * x318
        x320 = x317 * x318
        x321 = w_2x + x23 * x320
        x322 = x23 * x64
        x323 = w_2z * x178
        x324 = w_2y * x201
        x325 = x23 * x61
        x326 = x322 + x323 - x324 - x325 + x65 + x67
        x327 = x321 * x326
        x328 = w_2y - x27 * x320
        x329 = x33 * x56
        x330 = x308 * x32
        x331 = x123 * x315
        x332 = x26 * x50
        x333 = w_2z * x29
        x334 = x123 * x333
        x335 = x312 * x32
        x336 = x329 - x330 + x331 + x332 + x334 + x335
        x337 = x316 * x328 * x336
        x338 = self.p.m1 * x314 + self.p.m1 * x327 + self.p.m1 * x337 + x319 * x69
        x339 = psi_x_dot**2 * x13 * x85
        x340 = self.p.m2 * x317 * x318
        x341 = self.p.m2 * x314 + self.p.m2 * x327 + self.p.m2 * x337 + self.p.m2 * x339 + x340 * x69
        x342 = psi_x_dot * x7
        x343 = x17 * x6 * x7
        x344 = -self.p.r1 * w_1z * x343 - self.p.r2 * x302 * x342 - \
            x303 * x342 + x308 * x343 - x310 * x343 - x312 * x343
        x345 = x8 + 1
        x346 = psi_x_dot * x345
        x347 = self.p.r2 * x19 * x345
        x348 = -w_1z * x345 * x54 - x14 * x346 + x24 * x347 - x28 * x347 - x31 * x347 - x346 * x56
        x349 = psi_x_dot * x348
        x350 = -w_2y * x203 + w_2z * x180 - x23 * x44 + x23 * x48 + x50 + x52
        x351 = x321 * x350
        x352 = w_2y * x23 * x25 * x72 + x125 * x333 - x26 * x65 - x26 * x67 + x308 * x34 + x33 * x72
        x353 = x316 * x328 * x352
        x354 = psi_y_dot * self.p.m1 * x344 + self.p.m1 * x349 + \
            self.p.m1 * x351 + self.p.m1 * x353 + x319 * x53
        x355 = psi_y_dot * (-psi_x_dot * x93 - psi_y_dot * x112 + x344)
        x356 = psi_x_dot * (-psi_x_dot * x112 - psi_y_dot * x93 + x348)
        x357 = self.p.m2 * x351 + self.p.m2 * x353 + self.p.m2 * x355 + self.p.m2 * x356 + x340 * x53
        x358 = psi_x_dot * (-psi_x_dot * x109 + psi_y_dot * x88)
        x359 = psi_y_dot * (psi_x_dot * x88 - psi_y_dot * x109)
        x360 = self.p.g * self.p.m2 + self.p.m2 * x358 + self.p.m2 * x359
        x361 = phi_x_dot * self.p.m3
        x362 = w_2x * x130
        x363 = w_2z * x128
        x364 = phi_x_dot + x362 - x363
        x365 = self.p.l * x364
        x366 = x139 * x30
        x367 = w_2x * x128
        x368 = w_2z * x130
        x369 = -phi_y_dot * x127 - w_2y * x127 + x139 * x367 + x139 * x368
        x370 = self.p.m3 * x321
        x371 = phi_y_dot * x139
        x372 = w_2y * x139
        x373 = x127 * x367
        x374 = x127 * x368
        x375 = x371 + x372 + x373 + x374
        x376 = self.p.l * x128 * x375
        x377 = x127 * x130 * x26
        x378 = self.p.m3 * x316 * x328
        x379 = self.p.l * x375
        x380 = x127 * x154
        x381 = phi_y_dot * self.p.m3
        x382 = x127 * x362 - x127 * x363
        x383 = -x367 - x368
        x384 = self.p.g * self.p.m3 + self.p.m3 * x358 + self.p.m3 * x359 + x361 * (-x157 * x369 + x365 * (-x127 * x27 + x130 * x366 - x139 * x158)) + x370 * (-x27 * x376 + x365 * (-x25 * x377 + x366)) + x378 * (
            x365 * (-x139 * x23 * x25 - x147 - x29 * x380) - x379 * (-x131 + x158 * x29)) + x381 * (-x157 * x382 + x163 * x383 + x365 * (-x127 * x155 - x380) - x379 * (x158 - x160))
        x385 = x360 + x384
        x386 = x139 * x60
        x387 = -x36 - x57
        x388 = x139 * x26
        x389 = self.p.m3 * x317 * x318
        x390 = self.p.m3 * x314 + self.p.m3 * x339 + x361 * (-x135 * x369 + x365 * (-x127 * x58 + x130 * x386 + x137 * x139)) + x370 * (x326 + x365 * (x205 * x387 + x386) + x376 * x387) + x378 * (
            x336 + x365 * (-x159 * x34 + x35 * x388 + x377 * x38) - x379 * (-x136 * x38 - x154 * x34)) + x381 * (-x135 * x382 + x141 * x383 + x365 * (x127 * x132 - x127 * x133) - x379 * (-x137 - x140)) + x389 * (-x145 * x379 + x149 * x365 + x69)
        x391 = x136 * x32
        x392 = x139 * x37
        x393 = x38 - x40
        x394 = x39 - x59
        x395 = self.p.m3 * x355 + self.p.m3 * x356 + x361 * (-x146 * x369 + x365 * (-x127 * x41 + x130 * x392 + x139 * x391)) + x370 * (x350 + x365 * (x205 * x393 + x392) + x376 * x393) + x378 * (x352 + x365 * (-x159 * x32 + x161 * x32 + x388 * x39) - x379 * (
            -x154 * x32 - x155 * x32)) + x381 * (-x146 * x382 + x150 * x383 + x365 * (x127 * x143 - x127 * x144) - x379 * (-x148 - x391)) + x389 * (x365 * (-x138 + x139 * x387 + x205 * x394) - x379 * (-x128 * x394 - x132) + x53)
        x396 = 3 * pi * self.p.a * self.p.mu12 * x12 * x73 * \
            (x11 * x385 - x13 * (x341 + x390) + x18 * (x357 + x395)) / 16
        x397 = x11 * x396
        x398 = psi_y_dot * x42
        x399 = x317 * x318 * x42
        x400 = x321 * x42
        x401 = x316 * x328 * x42
        x402 = x18 * x396
        x403 = self.p.theta1 * (x398 * (x304 + x306 + x307 - x309 + x311 + x313) + x399 * (-x61 + x63 + x64 + x66 + x68) +
                                x400 * (-x322 - x323 + x324 + x325 - x65 - x67) + x401 * (-x329 + x330 - x331 - x332 - x334 - x335)) + x402
        x404 = x13 * x396
        x405 = self.p.theta1 * (x344 * x398 + x349 * x42 + x350 *
                                x400 + x352 * x401 + x399 * x53) - x404
        x406 = x403 * x42
        x407 = x405 * x42
        x408 = x369 * x375
        x409 = phi_y_dot * self.p.theta3x * x383 - self.p.theta3y * x408 + self.p.theta3z * x408
        x410 = x364 * x375
        x411 = -self.p.theta3x * x410 + self.p.theta3y * x410 + self.p.theta3z * \
            (phi_x_dot * (-x371 - x372 - x373 - x374) + phi_y_dot * (x139 * x362 - x139 * x363))
        x412 = x139 * x411
        x413 = x364 * x369
        x414 = self.p.theta3x * x413 + self.p.theta3y * \
            (phi_x_dot * x369 + phi_y_dot * x382) - self.p.theta3z * x413
        x415 = 1 / self.p.tau
        A[0, 0] = self.p.m1 * x3 + self.p.m1 * x9 + self.p.m2 * x3 + self.p.m2 * x9 + \
            self.p.m3 * x3 + self.p.m3 * x9 + self.p.theta1 + x1 * x77 + x75 - x78 * x80
        A[0, 1] = x1 * x105 + x101 - x107 * x78 + x90 - x92 * x95
        A[0, 2] = x1 * x120 + x111 - x114 * x92 + x119 - x122 * x78
        A[0, 3] = x1 * x174 + x152 - x153 * x164 + x171 - x176 * x78
        A[0, 4] = x1 * x197 + x187 + x189 + x194 - x199 * x78
        A[0, 5] = x1 * x218 - x153 * x209 + x208 + x215 - x220 * x78
        A[0, 6] = x1 * x225 - x141 * x16 - x150 * x22 - x153 * x163 + x224 + x226 * x78
        A[0, 7] = x1 * x230 + x16 * x183 + x185 * x22 + x189 + x229 + x231 * x78
        A[1, 0] = x103 * x232 + x233 * x84 + x90
        A[1, 1] = self.p.m1 * x82**2 + self.p.m1 * x84**2 + self.p.m2 * x238 + self.p.m2 * x239 + \
            self.p.m3 * x238 + self.p.m3 * x239 + x103 * x240 + x235 * x236 + x236 * x237 + x241 * x84
        A[1, 2] = x103 * x244 + x243 + x245 * x84
        A[1, 3] = x103 * x249 + x248 + x250 * x84
        A[1, 4] = x103 * x254 + x253 + x255 * x84
        A[1, 5] = x103 * x257 + x256 + x258 * x84
        A[1, 6] = x103 * x260 + x141 * x97 + x150 * x99 - x163 * x94 - x259 * x84
        A[1, 7] = x103 * x262 - x183 * x97 - x185 * x99 + x251 - x261 * x84
        A[2, 0] = x111 + x233 * x85
        A[2, 1] = x241 * x85 + x243
        A[2, 2] = self.p.m1 * x234 + self.p.m2 * x263 + \
            self.p.m3 * x263 + x235 * x264 + x237 * x264 + x245 * x85
        A[2, 3] = x250 * x85 + x265
        A[2, 4] = x255 * x85 + x267
        A[2, 5] = x258 * x85 + x268
        A[2, 6] = -3 * pi * self.p.a * self.p.mu12 * x12 * \
            x223 * x269 * x73 / 16 - x113 * x163 + x116 * x150
        A[2, 7] = -x116 * x185 - x228 * x269 + x266
        A[3, 0] = x126 * x233 + x152 + x172 * x232 + x23 * x75 + x270 * x79 - x271 * x76
        A[3, 1] = x101 * x23 - x104 * x271 + x106 * x270 + x126 * x241 + x172 * x240 + x248
        A[3, 2] = x119 * x23 - x120 * x271 + x121 * x270 + x126 * x245 + x172 * x244 + x265
        A[3, 3] = self.p.m1 * x278 + self.p.m1 * x279 + self.p.m2 * x278 + self.p.m2 * x279 + self.p.m3 * x142**2 + self.p.m3 * x151**2 + self.p.m3 * \
            x164**2 + self.p.theta2 + self.p.theta3x * x272 + x126 * x250 + x171 * x23 + x172 * x249 - x173 * x271 + x175 * x270 + x273 * x275 + x273 * x277
        A[3, 4] = x126 * x255 + x172 * x254 + x194 * x23 - x196 * x271 + x198 * x270 + x287
        A[3, 5] = x126 * x258 + x172 * x257 + x215 * x23 - x217 * x271 + x219 * x270 + x290
        A[3, 6] = self.p.theta3x * x130 - x126 * x259 + x141 * x167 + x150 * \
            x169 + x164 * x291 + x172 * x260 + x224 * x23 - x225 * x271 + x226 * x270
        A[3, 7] = -x126 * x261 - x167 * x183 - x169 * x185 + x172 * x262 + \
            x229 * x23 - x230 * x271 + x231 * x270 + x281 + x283 + x286
        A[4, 0] = x181 * x233 + x187 + x195 * x232 - x27 * x75 - x41 * x76 + x58 * x79
        A[4, 1] = -x101 * x27 - x104 * x41 + x106 * x58 + x181 * x241 + x195 * x240 + x253
        A[4, 2] = -x119 * x27 - x120 * x41 + x121 * x58 + x181 * x245 + x195 * x244 + x267
        A[4, 3] = -x171 * x27 - x173 * x41 + x175 * x58 + x181 * x250 + x195 * x249 + x287
        A[4, 4] = self.p.m1 * x294 + self.p.m1 * x295 + self.p.m2 * x294 + self.p.m2 * x295 + self.p.m3 * x184**2 + \
            self.p.m3 * x186**2 + self.p.theta2 + x181 * x255 - x194 * x27 + x195 * x254 - x196 * x41 + x198 * x58 + x293
        A[4, 5] = x181 * x258 + x195 * x257 - x215 * x27 - x217 * x41 + x219 * x58 + x299
        A[4, 6] = -x139 * x156 * x162 * x292 + x141 * x191 + x150 * x192 - \
            x181 * x259 + x195 * x260 - x224 * x27 - x225 * x41 + x226 * x58
        A[4, 7] = -x181 * x261 - x183 * x191 - x185 * x192 + \
            x195 * x262 - x229 * x27 - x230 * x41 + x231 * x58 + x293
        A[5, 0] = x204 * x233 + x208 + x216 * x232 - x30 * x75 - x37 * x76 + x60 * x79
        A[5, 1] = -x101 * x30 - x104 * x37 + x106 * x60 + x204 * x241 + x216 * x240 + x256
        A[5, 2] = -x119 * x30 - x120 * x37 + x121 * x60 + x204 * x245 + x216 * x244 + x268
        A[5, 3] = -x171 * x30 - x173 * x37 + x175 * x60 + x204 * x250 + x216 * x249 + x290
        A[5, 4] = -x194 * x30 - x196 * x37 + x198 * x60 + x204 * x255 + x216 * x254 + x299
        A[5, 5] = self.p.m1 * x300 + self.p.m1 * x301 + self.p.m2 * x300 + self.p.m2 * x301 + self.p.m3 * x206**2 + self.p.m3 * x207**2 + self.p.m3 * \
            x209**2 + self.p.theta2 + self.p.theta3x * x273 + x204 * x258 - x215 * x30 + x216 * x257 - x217 * x37 + x219 * x60 + x272 * x275 + x272 * x277
        A[5, 6] = x141 * x211 + x150 * x213 - x204 * x259 + x209 * x291 + \
            x216 * x260 - x224 * x30 - x225 * x37 + x226 * x60 - x288
        A[5, 7] = -x183 * x211 - x185 * x213 - x204 * x261 + x216 * \
            x262 - x229 * x30 - x230 * x37 + x231 * x60 + x296 + x297 + x298
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
        b[0] = -x1 * x403 + x14 * x338 + x14 * x341 + x14 * x390 + x20 * x354 + x20 * \
            x357 + x20 * x395 - x397 + x405 * x78 + x92 * (self.p.g * self.p.m1 + x385)
        b[1] = -x103 * x406 - x338 * x82 - x341 * x87 - x354 * x84 - x357 * \
            x89 + x360 * x93 + x384 * x93 - x390 * x87 - x395 * x89 - x407 * x84
        b[2] = -x110 * x357 - x110 * x395 + x112 * x360 + x112 * x384 - x354 * x85 - x407 * x85
        b[3] = -x124 * x338 - x124 * x341 - x126 * x354 - x126 * x357 - x126 * x407 - x128 * x412 - x129 * x414 - \
            x130 * x409 - x142 * x390 - x151 * x395 - x164 * x384 - x172 * x406 - x23 * x397 - x270 * x404 + x271 * x402
        b[4] = x127 * x411 - x139 * x414 - x179 * x338 - x179 * x341 - x181 * x354 - x181 * x357 - x181 * \
            x407 - x184 * x390 - x186 * x395 + x188 * x384 - x195 * x406 + x27 * x397 + x402 * x41 - x404 * x58
        b[5] = x128 * x409 - x130 * x412 - x202 * x338 - x202 * x341 - x204 * x354 - x204 * x357 - x204 * x407 - \
            x205 * x414 - x206 * x390 - x207 * x395 - x209 * x384 - x216 * x406 + x30 * x397 + x37 * x402 - x404 * x60
        b[6] = x415 * (omega_x_cmd - phi_x_dot)
        b[7] = x415 * (omega_y_cmd - phi_y_dot)

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
        z = radius * np.outer(np.sin(lon), np.sin(lat))
        y = radius * np.outer(np.ones(circle_res), np.cos(lat))

        R_IB = q_IB.rotation_matrix()

        x_rot = R_IB[0, 0] * x + R_IB[0, 1] * y + R_IB[0, 2] * z
        y_rot = R_IB[1, 0] * x + R_IB[1, 1] * y + R_IB[1, 2] * z
        z_rot = R_IB[2, 0] * x + R_IB[2, 1] * y + R_IB[2, 2] * z

        return [center[0] + x_rot, center[1] + y_rot, center[2] + z_rot]
