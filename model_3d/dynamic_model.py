"""Dynamic model of 3D version of Double Ball Balancer

This module contains the class DynamicModel for simulating the non-linear dynamics of the 3D Double Ball Balancer and a corresponding class ModelParam containing the relevant parameters.

author: Christof Dubs
"""

import itertools

import numpy as np
from numpy import cos, exp, pi, sin, tan
from pyrotation import Quaternion, quat_from_angle_vector
from scipy.integrate import odeint


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

    def __init__(self):
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

    def is_valid(self):
        """Checks validity of parameter configuration

        Returns:
            bool: True if valid, False if invalid.
        """
        return (
            self.a >= 0
            and self.g > 0
            and self.l > 0
            and self.m1 > 0
            and self.m2 > 0
            and self.m3 > 0
            and self.mu1 >= 0
            and self.mu12 >= 0
            and self.r1 > 0
            and self.r2 > 0
            and self.tau > 0
            and self.theta1 > 0
            and self.theta2 > 0
            and self.theta3x > 0
            and self.theta3y > 0
            and self.theta3z > 0
        )


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
            skip_checks (bool, optional): if set to True and x0 is provided, x0 is set without checking it.
        """
        if skip_checks and x0 is not None:
            self.x = x0
            return

        if x0 is None or not self.set_state(x0):
            self.x = np.zeros(STATE_SIZE, dtype=float)
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
            print("called set_state with argument of type {} instead of numpy.ndarray. Ignoring.".format(type(x0)))
            return False

        # make 1D version of x0
        x0_flat = x0.flatten()
        if len(x0_flat) != STATE_SIZE:
            print("called set_state with array of length {} instead of {}. Ignoring.".format(len(x0_flat), STATE_SIZE))
            return False

        q1_norm = np.linalg.norm(x0_flat[Q_1_W_IDX : Q_1_Z_IDX + 1])
        q2_norm = np.linalg.norm(x0_flat[Q_2_W_IDX : Q_2_Z_IDX + 1])

        # quaternion check
        if q1_norm == 0 or q2_norm == 0:
            return False

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
        return self.x[Q_1_W_IDX : Q_1_Z_IDX + 1]

    @q1.setter
    def q1(self, value):
        if isinstance(value, Quaternion):
            self.x[Q_1_W_IDX : Q_1_Z_IDX + 1] = value.q
            return
        if isinstance(value, np.ndarray):
            self.x[Q_1_W_IDX : Q_1_Z_IDX + 1] = value
            return
        print("failed to set x")

    @property
    def q2(self):
        return self.x[Q_2_W_IDX : Q_2_Z_IDX + 1]

    @q2.setter
    def q2(self, value):
        if isinstance(value, Quaternion):
            self.x[Q_2_W_IDX : Q_2_Z_IDX + 1] = value.q
            return
        if isinstance(value, np.ndarray):
            self.x[Q_2_W_IDX : Q_2_Z_IDX + 1] = value
            return
        print("failed to set x")

    @property
    def q3(self):
        return (
            Quaternion(self.q2)
            * quat_from_angle_vector(np.array([0, self.phi_y, 0]))
            * quat_from_angle_vector(np.array([self.phi_x, 0, 0]))
        )

    @property
    def R_IB2(self):
        return Quaternion(self.q2).rotation_matrix()

    @property
    def phi(self):
        return self.x[PHI_X_IDX : PHI_Y_IDX + 1]

    @phi.setter
    def phi(self, value):
        self.x[PHI_X_IDX : PHI_Y_IDX + 1] = value

    @property
    def phi_dot(self):
        return self.x[PHI_X_DOT_IDX : PHI_Y_DOT_IDX + 1]

    @phi_dot.setter
    def phi_dot(self, value):
        self.x[PHI_X_DOT_IDX : PHI_Y_DOT_IDX + 1] = value

    @property
    def psi(self):
        return self.x[PSI_X_IDX : PSI_Y_IDX + 1]

    @psi.setter
    def psi(self, value):
        self.x[PSI_X_IDX : PSI_Y_IDX + 1] = value

    @property
    def psi_dot(self):
        return self.x[PSI_X_DOT_IDX : PSI_Y_DOT_IDX + 1]

    @psi_dot.setter
    def psi_dot(self, value):
        self.x[PSI_X_DOT_IDX : PSI_Y_DOT_IDX + 1] = value

    @property
    def pos(self):
        return self.x[X_IDX : Y_IDX + 1]

    @pos.setter
    def pos(self, value):
        self.x[X_IDX : Y_IDX + 1] = value

    @property
    def omega(self):
        return self.x[OMEGA_1_Z_IDX : PHI_Y_DOT_IDX + 1]

    @omega.setter
    def omega(self, value):
        self.x[OMEGA_1_Z_IDX : PHI_Y_DOT_IDX + 1] = value

    @property
    def omega_1_z(self):
        return self.x[OMEGA_1_Z_IDX]

    @omega_1_z.setter
    def omega_1_z(self, value):
        self.x[OMEGA_1_Z_IDX] = value

    @property
    def omega_2(self):
        return self.x[OMEGA_2_X_IDX : OMEGA_2_Z_IDX + 1]

    @omega_2.setter
    def omega_2(self, value):
        self.x[OMEGA_2_X_IDX : OMEGA_2_Z_IDX + 1] = value


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
            print("Warning: not all parameters set!")

        if x0 is not None:
            if not isinstance(x0, ModelState):
                print("invalid type passed as initial state")
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

    def is_irrecoverable(self, state=None, contact_forces=None, omega_cmd=None, ignore_force_check=False):
        """Checks if system is recoverable

        args:
            state (ModelState, optional): state. If not specified, the internal state is checked
            contact_forces(list(numpy.ndarray), optional): contact forces [N]. If not specified, will be internally calculated
            omega_cmd (numpy.ndarray, optional): motor speed commands [rad/s] used for contact force calculation if contact_forces are not specified
            ignore_force_check (optional): If set to True, will skip the contact forces check

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

        # lift off: contact force between lower and upper ball <= 0
        if not ignore_force_check:
            if contact_forces is None:
                contact_forces = self.compute_contact_forces(state, omega_cmd)

            if np.dot(contact_forces[1], self._compute_e_S1S2(state)) <= 0:
                return True

        return False

    def get_visualization(self, state=None, contact_forces=None, omega_cmd=None, visualize_contact_forces=False):
        """Get visualization of the system for plotting

        Usage example:
            vis = model.get_visualization()
            ax.plot_wireframe(*vis['lower_ball'])
            ax.quiver(*vis['F1'])

        args:
            state (ModelState, optional): state. If not specified, the internal state is checked
            contact_forces(list(numpy.ndarray), optional): contact forces [N]. If not specified, will be internally calculated
            omega_cmd (numpy.ndarray, optional): motor speed commands [rad/s] used for contact force calculation if contact_forces are not specified
            visualize_contact_forces (optional): contact forces will only be visualized if set to True

        Returns:
            dict: dictionary with keys "lower_ball", "upper_ball" and "lever_arm". The value for each key is a list with three elements: a list of x coordinates, a list of y coordinates and a list of z coordinates.
        """
        if state is None:
            state = self.state

        vis = {}

        r_OSi = self._compute_r_OSi(state)
        vis["lower_ball"] = self._compute_ball_visualization(r_OSi[0], self.p.r1, Quaternion(state.q1))
        vis["upper_ball"] = self._compute_ball_visualization(r_OSi[1], self.p.r2, Quaternion(state.q2))
        vis["lever_arm"] = [np.array([[r_OSi[1][i], r_OSi[2][i]]]) for i in range(3)]

        if visualize_contact_forces:
            if contact_forces is None:
                contact_forces = self.compute_contact_forces(state, omega_cmd)

            force_scale = 0.05
            contact_pt_1 = np.array([r_OSi[0][0], r_OSi[0][1], 0])
            vis["F1"] = list(itertools.chain.from_iterable([contact_pt_1, force_scale * contact_forces[0]]))

            contact_pt_2 = r_OSi[0] + self.p.r1 * self._compute_e_S1S2(state)
            vis["F12"] = list(itertools.chain.from_iterable([contact_pt_2, force_scale * contact_forces[1]]))

            vis["F23"] = list(itertools.chain.from_iterable([r_OSi[1], force_scale * contact_forces[2]]))

        return vis

    def compute_contact_forces(self, state=None, omega_cmd=None):
        """computes contact forces between bodies

        This function computes the contact forces between the rigid bodies.
        args:
            x (numpy.ndarray, optional): state. If not specified, the internal state is used
            omega_cmd(numpy.ndarray, optional): motor speed command [rad/s]. Defaults to zero if not specified
        Returns: list of the 3 contact forces [F1, F12, F23] with:
        - F1: force from ground onto lower ball
        - F12: force from lower ball onto upper ball
        - F23: force from upper ball onto lever arm
        """
        if state is None:
            state = self.state
        if omega_cmd is None:
            print("Warning: no omega_cmd specified for contact force calculation; default to [0 0]")
            omega_cmd = np.zeros(2)

        [r_xx, r_xy, r_xz, r_yx, r_yy, r_yz, r_zx, r_zy, r_zz] = state.R_IB2.reshape(9)
        [phi_x, phi_y] = state.phi
        [phi_x_dot, phi_y_dot] = state.phi_dot
        [psi_x, psi_y] = state.psi
        [psi_x_dot, psi_y_dot] = state.psi_dot
        w_1z = state.omega_1_z
        [w_2x, w_2y, w_2z] = state.omega_2

        [w_1_dot_z, psi_x_ddot, psi_y_ddot, w_2_dot_x, w_2_dot_y, w_2_dot_z, phi_x_ddot, phi_y_ddot] = (
            self._compute_omega_dot(state, omega_cmd)
        )

        F1 = np.zeros(3)
        F12 = np.zeros(3)
        F23 = np.zeros(3)

        x0 = self.p.r1 + self.p.r2
        x1 = tan(psi_x)
        x2 = cos(psi_y)
        x3 = 1 / x2
        x4 = self.p.r1 * w_1_dot_z * x1 * x3
        x5 = self.p.m1 * self.p.r2 * w_2x
        x6 = r_yy * w_2z - r_yz * w_2y
        x7 = self.p.m1 * self.p.r2 * w_2y
        x8 = -r_yx * w_2z + r_yz * w_2x
        x9 = self.p.m1 * self.p.r2 * w_2z
        x10 = r_yx * w_2y - r_yy * w_2x
        x11 = psi_x_ddot * self.p.m1
        x12 = tan(psi_y)
        x13 = self.p.r1 * x12
        x14 = self.p.r2 * x12
        x15 = -x1 * x13 - x1 * x14
        x16 = self.p.r2 * x1 * x3
        x17 = -r_yx * self.p.r2 - r_zx * x16
        x18 = w_2_dot_x * x17
        x19 = -r_yy * self.p.r2 - r_zy * x16
        x20 = w_2_dot_y * x19
        x21 = -r_yz * self.p.r2 - r_zz * x16
        x22 = w_2_dot_z * x21
        x23 = self.p.m1 * self.p.r2 * x1 * x3
        x24 = r_zy * w_2z - r_zz * w_2y
        x25 = w_2x * x24
        x26 = -r_zx * w_2z + r_zz * w_2x
        x27 = w_2y * x26
        x28 = r_zx * w_2y - r_zy * w_2x
        x29 = w_2z * x28
        x30 = psi_x_dot * x1
        x31 = x12**2 + 1
        x32 = self.p.r1 * x31
        x33 = self.p.r2 * x31
        x34 = sin(psi_y)
        x35 = x2 ** (-2)
        x36 = r_zx * w_2x
        x37 = self.p.r2 * x1 * x34 * x35
        x38 = r_zy * w_2y
        x39 = r_zz * w_2z
        x40 = -self.p.r1 * w_1z * x1 * x34 * x35 - x30 * x32 - x30 * x33 - x36 * x37 - x37 * x38 - x37 * x39
        x41 = x1**2 + 1
        x42 = psi_x_dot * x41
        x43 = self.p.r1 * x3
        x44 = self.p.r2 * x3 * x41
        x45 = -w_1z * x41 * x43 - x13 * x42 - x14 * x42 - x36 * x44 - x38 * x44 - x39 * x44
        x46 = self.p.m2 * self.p.r2 * w_2x
        x47 = self.p.m2 * self.p.r2 * w_2y
        x48 = self.p.m2 * self.p.r2 * w_2z
        x49 = cos(psi_x)
        x50 = x0 * x49
        x51 = x2 * x50
        x52 = psi_y_ddot * (x0 + x51)
        x53 = self.p.m2 * self.p.r2 * x1 * x3
        x54 = sin(psi_x)
        x55 = x0 * x34 * x54
        x56 = psi_x_ddot * (x15 - x55)
        x57 = x0 * x2 * x54
        x58 = x0 * x34 * x49
        x59 = psi_y_dot * (-psi_x_dot * x57 - psi_y_dot * x58 + x40)
        x60 = psi_x_dot * (-psi_x_dot * x58 - psi_y_dot * x57 + x45)
        x61 = self.p.m3 * self.p.r2 * w_2x
        x62 = self.p.m3 * self.p.r2 * w_2y
        x63 = self.p.m3 * self.p.r2 * w_2z
        x64 = phi_y_ddot * self.p.m3
        x65 = cos(phi_x)
        x66 = self.p.l * x65
        x67 = cos(phi_y)
        x68 = r_xx * x67
        x69 = sin(phi_y)
        x70 = r_xz * x69
        x71 = x68 - x70
        x72 = x66 * x71
        x73 = self.p.m3 * self.p.r2 * x1 * x3
        x74 = phi_x_ddot * self.p.m3
        x75 = sin(phi_x)
        x76 = r_xx * x69
        x77 = r_xz * x67
        x78 = self.p.l * (r_xy * x65 + x75 * x76 + x75 * x77)
        x79 = -r_xx * w_2z + r_xz * w_2x
        x80 = w_2x * x67
        x81 = w_2z * x69
        x82 = phi_x_dot + x80 - x81
        x83 = self.p.l * self.p.m3 * x65 * x82
        x84 = self.p.m3 * w_2_dot_y
        x85 = r_xx * w_2y - r_xy * w_2x
        x86 = self.p.l * x75 * x82
        x87 = w_2x * x69
        x88 = w_2z * x67
        x89 = self.p.l * (phi_y_dot * x65 + w_2y * x65 + x75 * x87 + x75 * x88)
        x90 = self.p.m3 * (x67 * x86 + x69 * x89)
        x91 = r_xy * w_2z - r_xz * w_2y
        x92 = self.p.m3 * (-x67 * x89 + x69 * x86)
        x93 = self.p.m3 * w_2_dot_x
        x94 = self.p.l * x71 * x75
        x95 = self.p.m3 * w_2_dot_z
        x96 = phi_x_dot * self.p.m3
        x97 = self.p.l * x82
        x98 = self.p.l * (-phi_y_dot * x75 - w_2y * x75 + x65 * x87 + x65 * x88)
        x99 = phi_y_dot * self.p.m3
        x100 = self.p.l * (x75 * x80 - x75 * x81)
        x101 = -x87 - x88
        x102 = (
            -self.p.m3 * x4
            + self.p.m3 * x52
            + self.p.m3 * x56
            + self.p.m3 * x59
            + self.p.m3 * x60
            - x10 * x63
            - x25 * x73
            - x27 * x73
            - x29 * x73
            - x6 * x61
            - x62 * x8
            - x64 * x72
            + x74 * x78
            + x79 * x83
            + x84 * (x19 - x72)
            + x85 * x90
            + x91 * x92
            + x93 * (x17 + x67 * x78 - x69 * x94)
            + x95 * (x21 - x67 * x94 - x69 * x78)
            + x96 * (-x71 * x98 + x97 * (-r_xy * x75 + x65 * x76 + x65 * x77))
            + x99 * (-x100 * x71 + x101 * x78 - x89 * (-x76 - x77) + x97 * (x68 * x75 - x70 * x75))
        )
        x103 = (
            self.p.m2 * x18
            + self.p.m2 * x20
            + self.p.m2 * x22
            - self.p.m2 * x4
            + self.p.m2 * x52
            + self.p.m2 * x56
            + self.p.m2 * x59
            + self.p.m2 * x60
            - x10 * x48
            + x102
            - x25 * x53
            - x27 * x53
            - x29 * x53
            - x46 * x6
            - x47 * x8
        )
        x104 = self.p.r1 * w_1_dot_z * x12
        x105 = r_xx * self.p.r2 - r_zx * x14
        x106 = w_2_dot_x * x105
        x107 = r_xy * self.p.r2 - r_zy * x14
        x108 = w_2_dot_y * x107
        x109 = r_xz * self.p.r2 - r_zz * x14
        x110 = w_2_dot_z * x109
        x111 = self.p.m1 * self.p.r2 * x12
        x112 = -self.p.r2 * x3 - x43
        x113 = psi_x_dot * x34 * x35
        x114 = psi_y_dot * (-self.p.r1 * x113 - self.p.r2 * x113 - w_1z * x32 - x33 * x36 - x33 * x38 - x33 * x39)
        x115 = psi_x_dot**2 * x0 * x54
        x116 = self.p.m2 * self.p.r2 * x12
        x117 = psi_x_ddot * (x112 - x50)
        x118 = self.p.m3 * self.p.r2 * x12
        x119 = r_yx * x67
        x120 = r_yz * x69
        x121 = x119 - x120
        x122 = x121 * x66
        x123 = r_yx * x69
        x124 = r_yz * x67
        x125 = self.p.l * (r_yy * x65 + x123 * x75 + x124 * x75)
        x126 = self.p.l * x121 * x75
        x127 = (
            -self.p.m3 * x104
            + self.p.m3 * x114
            + self.p.m3 * x115
            + self.p.m3 * x117
            + x10 * x90
            - x118 * x25
            - x118 * x27
            - x118 * x29
            - x122 * x64
            + x125 * x74
            + x6 * x92
            + x61 * x91
            + x62 * x79
            + x63 * x85
            + x8 * x83
            + x84 * (x107 - x122)
            + x93 * (x105 + x125 * x67 - x126 * x69)
            + x95 * (x109 - x125 * x69 - x126 * x67)
            + x96 * (-x121 * x98 + x97 * (-r_yy * x75 + x123 * x65 + x124 * x65))
            + x99 * (-x100 * x121 + x101 * x125 - x89 * (-x123 - x124) + x97 * (x119 * x75 - x120 * x75))
        )
        x128 = (
            -self.p.m2 * x104
            + self.p.m2 * x106
            + self.p.m2 * x108
            + self.p.m2 * x110
            + self.p.m2 * x114
            + self.p.m2 * x115
            + self.p.m2 * x117
            - x116 * x25
            - x116 * x27
            - x116 * x29
            + x127
            + x46 * x91
            + x47 * x79
            + x48 * x85
        )
        x129 = psi_x_ddot * x0 * x2 * x54
        x130 = psi_y_ddot * x0 * x34 * x49
        x131 = psi_x_dot * (-psi_x_dot * x51 + psi_y_dot * x55)
        x132 = psi_y_dot * (psi_x_dot * x55 - psi_y_dot * x51)
        x133 = r_zx * x67
        x134 = r_zz * x69
        x135 = x133 - x134
        x136 = self.p.l * x135 * x65
        x137 = r_zx * x69
        x138 = r_zz * x67
        x139 = self.p.l * (r_zy * x65 + x137 * x75 + x138 * x75)
        x140 = self.p.l * x135 * x75
        x141 = (
            self.p.g * self.p.m3
            - self.p.m3 * x129
            - self.p.m3 * x130
            + self.p.m3 * x131
            + self.p.m3 * x132
            - x136 * x64
            - x136 * x84
            + x139 * x74
            + x24 * x92
            + x26 * x83
            + x28 * x90
            + x93 * (x139 * x67 - x140 * x69)
            + x95 * (-x139 * x69 - x140 * x67)
            + x96 * (-x135 * x98 + x97 * (-r_zy * x75 + x137 * x65 + x138 * x65))
            + x99 * (-x100 * x135 + x101 * x139 - x89 * (-x137 - x138) + x97 * (x133 * x75 - x134 * x75))
        )
        x142 = self.p.g * self.p.m2 - self.p.m2 * x129 - self.p.m2 * x130 + self.p.m2 * x131 + self.p.m2 * x132 + x141
        F1[0] = (
            psi_x_dot * self.p.m1 * x45
            + psi_y_ddot * self.p.m1 * x0
            + psi_y_dot * self.p.m1 * x40
            + self.p.m1 * x18
            + self.p.m1 * x20
            + self.p.m1 * x22
            - self.p.m1 * x4
            - x10 * x9
            + x103
            + x11 * x15
            - x23 * x25
            - x23 * x27
            - x23 * x29
            - x5 * x6
            - x7 * x8
        )
        F1[1] = (
            -self.p.m1 * x104
            + self.p.m1 * x106
            + self.p.m1 * x108
            + self.p.m1 * x110
            + self.p.m1 * x114
            + x11 * x112
            - x111 * x25
            - x111 * x27
            - x111 * x29
            + x128
            + x5 * x91
            + x7 * x79
            + x85 * x9
        )
        F1[2] = self.p.g * self.p.m1 + x142
        F12[0] = x103
        F12[1] = x128
        F12[2] = x142
        F23[0] = x102
        F23[1] = x127
        F23[2] = x141

        return [F1, F12, F23]

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
        if self.is_irrecoverable(state=eval_state, ignore_force_check=True):
            return np.zeros(np.shape(eval_state.x))

        xdot = ModelState()

        xdot.omega = self._compute_omega_dot(eval_state, omega_cmd)

        omega_1 = self._get_lower_ball_omega(eval_state)

        xdot.q1 = Quaternion(eval_state.q1).q_dot(omega_1, frame="inertial")

        xdot.q2 = Quaternion(eval_state.q2).q_dot(eval_state.omega_2, frame="body")

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
        [r_xx, r_xy, r_xz, r_yx, r_yy, r_yz, r_zx, r_zy, r_zz] = state.R_IB2.reshape(9)
        [psi_x, psi_y] = state.psi
        [psi_x_dot, psi_y_dot] = state.psi_dot
        w_1z = state.omega_1_z
        [w_2x, w_2y, w_2z] = state.omega_2

        omega_1 = np.zeros(3)

        x0 = 1 / self.p.r1
        x1 = tan(psi_y)
        x2 = self.p.r1 * x1
        x3 = self.p.r2 * w_2x
        x4 = self.p.r2 * w_2y
        x5 = self.p.r2 * w_2z
        x6 = r_zx * self.p.r2 * w_2x
        x7 = r_zy * self.p.r2 * w_2y
        x8 = r_zz * self.p.r2 * w_2z
        x9 = 1 / cos(psi_y)
        x10 = psi_x_dot * x9
        x11 = tan(psi_x)
        x12 = psi_x_dot * x11
        x13 = x11 * x9
        omega_1[0] = x0 * (
            -r_xx * x3
            - r_xy * x4
            - r_xz * x5
            + self.p.r1 * x10
            + self.p.r2 * x10
            + w_1z * x2
            + x1 * x6
            + x1 * x7
            + x1 * x8
        )
        omega_1[1] = x0 * (
            psi_y_dot * self.p.r1
            + psi_y_dot * self.p.r2
            - r_yx * x3
            - r_yy * x4
            - r_yz * x5
            - self.p.r1 * w_1z * x13
            - self.p.r2 * x1 * x12
            - x12 * x2
            - x13 * x6
            - x13 * x7
            - x13 * x8
        )
        omega_1[2] = w_1z

        return omega_1

    def _compute_omega_dot(self, state, omega_cmd):
        """computes angular acceleration matrix of rotational part of system dynamics (equal to jacobian matrix since dynamics are linear in angular accelerations)

        The non-linear rotational dynamics are of the form

        A * [omega_1_z_dot, psi_x_ddot, psi_y_ddot, omega_2_x_dot, omega_2_y_dot, omega_2_z_dot, phi_x_ddot, phi_y_ddot] = b

        where A = A(phi_x, phi_y, phi_x_dot, phi_y_dot, psi_x, psi_y) and b(state, inputs).

        args:
            state (ModelState): current state
            omega_cmd (np.ndarray): motor speed commands [rad/s]

        Returns: array containing the time derivative of the angular velocity state [rad/s^2]
        """
        [r_xx, r_xy, r_xz, r_yx, r_yy, r_yz, r_zx, r_zy, r_zz] = state.R_IB2.reshape(9)
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
        def Max(x, y):
            return np.max([x, y])

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
        x23 = -x13 * (-x15 - x16) + x18 * (-x21 - x22)
        x24 = r_zx * w_2x
        x25 = r_zy * w_2y
        x26 = r_zz * w_2z
        x27 = 1 / self.p.r1
        x28 = r_xx * self.p.r2
        x29 = r_xy * self.p.r2
        x30 = r_xz * self.p.r2
        x31 = self.p.r2 * x1
        x32 = self.p.r1 * x19
        x33 = self.p.r2 * x19
        x34 = r_yx * self.p.r2
        x35 = r_yy * self.p.r2
        x36 = r_yz * self.p.r2
        x37 = x14 * x7
        x38 = x31 * x7
        x39 = self.p.r2 * x19 * x7
        x40 = -1 + 2 / (
            exp(
                x11 * (-w_1z + x24 + x25 + x26)
                - x13
                * (
                    r_yx * w_2x
                    + r_yy * w_2y
                    + r_yz * w_2z
                    - x27
                    * (
                        -psi_x_dot * x37
                        - psi_x_dot * x38
                        + psi_y_dot * self.p.r1
                        + psi_y_dot * self.p.r2
                        - w_1z * x20
                        - w_2x * x34
                        - w_2y * x35
                        - w_2z * x36
                        - x24 * x39
                        - x25 * x39
                        - x26 * x39
                    )
                )
                + x18
                * (
                    r_xx * w_2x
                    + r_xy * w_2y
                    + r_xz * w_2z
                    - x27
                    * (
                        psi_x_dot * x32
                        + psi_x_dot * x33
                        + w_1z * x14
                        - w_2x * x28
                        - w_2y * x29
                        - w_2z * x30
                        + x24 * x31
                        + x25 * x31
                        + x26 * x31
                    )
                )
            )
            + 1
        )
        x41 = 3 * pi * self.p.a * self.p.mu12 * x12 * x23 * x40 / 16
        x42 = x11 * x41
        x43 = x18 * x41
        x44 = self.p.theta1 * x1 + x43
        x45 = x19 * x7
        x46 = 3 * pi * self.p.a * self.p.mu12 * x12 * x13 * x23 * x40 / 16
        x47 = -self.p.theta1 * x45 - x46
        x48 = self.p.m1 * self.p.r1 * x1
        x49 = -x32 - x33
        x50 = self.p.m1 * self.p.r1 * x19 * x7
        x51 = -x37 - x38
        x52 = self.p.r1 + self.p.r2
        x53 = x10 * x52
        x54 = x49 - x53
        x55 = x13 * x17 * x52
        x56 = x51 - x55
        x57 = -x15 * x54 - x16 * x54 - x21 * x56 - x22 * x56 - x48 * x49 - x50 * x51
        x58 = -1 + 2 / (exp(w_1z) + 1)
        x59 = 3 * pi * self.p.a * self.p.mu1 * self.p.r1 * x58 / 16
        x60 = x13 * x4 * x52
        x61 = self.p.m3 * x60
        x62 = -self.p.m2 * x60 - x61
        x63 = self.p.m3 * x54
        x64 = self.p.m3 * x56
        x65 = x11 * x62 - x13 * (self.p.m2 * x54 + x63) + x18 * (self.p.m2 * x56 + x64)
        x66 = 3 * pi * self.p.a * self.p.mu12 * x10 * x12 * x4 * x40 / 16
        x67 = x65 * x66
        x68 = self.p.theta1 * x27
        x69 = x32 + x33
        x70 = 3 * pi * self.p.a * self.p.mu12 * x12 * x40 / 16
        x71 = x10 * x17 * x65 * x70
        x72 = x68 * x69 + x71
        x73 = 3 * pi * self.p.a * self.p.mu12 * x12 * x13 * x40 / 16
        x74 = x65 * x73
        x75 = x51 * x68 - x74
        x76 = self.p.m1 * x52
        x77 = x4 * x53
        x78 = x52 + x77
        x79 = -x20 * x76 - x21 * x78 - x22 * x78
        x80 = x10 * x17 * x52
        x81 = self.p.m3 * x80
        x82 = -self.p.m2 * x80 - x81
        x83 = self.p.m2 * x78
        x84 = self.p.m3 * x78
        x85 = x11 * x82 + x18 * (x83 + x84)
        x86 = x66 * x85
        x87 = 3 * pi * self.p.a * self.p.mu12 * x10 * x12 * x17 * x40 * x85 / 16
        x88 = x73 * x85
        x89 = x52 * x68 - x88
        x90 = r_zx * x31
        x91 = x28 - x90
        x92 = -r_zx * x39 - x34
        x93 = sin(phi_y)
        x94 = sin(phi_x)
        x95 = cos(phi_y)
        x96 = r_yx * x95
        x97 = r_yz * x93
        x98 = x96 - x97
        x99 = self.p.l * x94 * x98
        x100 = cos(phi_x)
        x101 = r_yx * x93
        x102 = r_yz * x95
        x103 = self.p.l * (r_yy * x100 + x101 * x94 + x102 * x94)
        x104 = x103 * x95 + x91 - x93 * x99
        x105 = r_xx * x95
        x106 = r_xz * x93
        x107 = x105 - x106
        x108 = self.p.l * x107 * x94
        x109 = r_xx * x93
        x110 = r_xz * x95
        x111 = self.p.l * (r_xy * x100 + x109 * x94 + x110 * x94)
        x112 = -x108 * x93 + x111 * x95 + x92
        x113 = -x104 * x16 - x112 * x22 - x15 * x91 - x21 * x92 - x48 * x91 - x50 * x92
        x114 = 3 * pi * self.p.a * self.p.m3 * self.p.mu1 * self.p.r1 * x58 / 16
        x115 = r_zx * x95
        x116 = r_zz * x93
        x117 = x115 - x116
        x118 = self.p.l * x117 * x94
        x119 = r_zx * x93
        x120 = r_zz * x95
        x121 = r_zy * x100 + x119 * x94 + x120 * x94
        x122 = self.p.l * x121
        x123 = -x118 * x93 + x122 * x95
        x124 = self.p.m3 * x10 * x4
        x125 = self.p.m2 * x91
        x126 = self.p.m3 * x104
        x127 = self.p.m2 * x92
        x128 = self.p.m3 * x112
        x129 = x123 * x124 - x13 * (x125 + x126) + x18 * (x127 + x128)
        x130 = x129 * x66
        x131 = -x28 + x90
        x132 = x10 * x129 * x17 * x70
        x133 = x131 * x68 + x132
        x134 = x129 * x73
        x135 = -x134 + x68 * x92
        x136 = r_zy * x31
        x137 = -x136 + x29
        x138 = -r_zy * x39 - x35
        x139 = self.p.l * x100
        x140 = x139 * x98
        x141 = x137 - x140
        x142 = x107 * x139
        x143 = x138 - x142
        x144 = -x137 * x15 - x137 * x48 - x138 * x21 - x138 * x50 - x141 * x16 - x143 * x22
        x145 = self.p.l * x100 * x117
        x146 = x114 * x145
        x147 = -x124 * x145
        x148 = self.p.m2 * x137
        x149 = self.p.m3 * x141
        x150 = self.p.m2 * x138
        x151 = self.p.m3 * x143
        x152 = -x13 * (x148 + x149) + x147 + x18 * (x150 + x151)
        x153 = x152 * x66
        x154 = x136 - x29
        x155 = x10 * x152 * x17 * x70
        x156 = x154 * x68 + x155
        x157 = x152 * x73
        x158 = x138 * x68 - x157
        x159 = r_zz * x31
        x160 = -x159 + x30
        x161 = -r_zz * x39 - x36
        x162 = -x103 * x93 + x160 - x95 * x99
        x163 = -x108 * x95 - x111 * x93 + x161
        x164 = -x15 * x160 - x16 * x162 - x160 * x48 - x161 * x21 - x161 * x50 - x163 * x22
        x165 = -x118 * x95 - x122 * x93
        x166 = self.p.m2 * x160
        x167 = self.p.m3 * x162
        x168 = self.p.m2 * x161
        x169 = self.p.m3 * x163
        x170 = x124 * x165 - x13 * (x166 + x167) + x18 * (x168 + x169)
        x171 = x170 * x66
        x172 = x159 - x30
        x173 = x10 * x17 * x170 * x70
        x174 = x172 * x68 + x173
        x175 = x170 * x73
        x176 = x161 * x68 - x175
        x177 = self.p.m3 * x13
        x178 = self.p.m3 * x10 * x17
        x179 = -x103 * x177 + x111 * x178 + x122 * x124
        x180 = x179 * x66
        x181 = 3 * pi * self.p.a * self.p.mu12 * x10 * x12 * x17 * x179 * x40 / 16
        x182 = 3 * pi * self.p.a * self.p.mu12 * x12 * x13 * x179 * x40 / 16
        x183 = x140 * x177 - x142 * x178 + x147
        x184 = x183 * x66
        x185 = 3 * pi * self.p.a * self.p.mu12 * x10 * x12 * x17 * x183 * x40 / 16
        x186 = 3 * pi * self.p.a * self.p.mu12 * x12 * x13 * x183 * x40 / 16
        x187 = x27 * x44
        x188 = x27 * x47
        x189 = x52**2
        x190 = self.p.m2 * x189
        x191 = x13**2 * x5
        x192 = self.p.m3 * x189
        x193 = x54**2
        x194 = x56**2
        x195 = x27 * x72
        x196 = x27 * x75
        x197 = x10 * x13 * x17 * x4
        x198 = x190 * x197 + x192 * x197 + x51 * x76 + x56 * x83 + x56 * x84
        x199 = 3 * pi * self.p.a * self.p.mu12 * x10 * x12 * x17 * x27 * x40 * x85 / 16
        x200 = x27 * x89
        x201 = self.p.m1 * x91
        x202 = self.p.m1 * x92
        x203 = x104 * x63 + x112 * x64 - x123 * x61 + x125 * x54 + x127 * x56 + x201 * x49 + x202 * x51
        x204 = x133 * x27
        x205 = x135 * x27
        x206 = x145 * x61
        x207 = self.p.m1 * x137
        x208 = self.p.m1 * x138
        x209 = x141 * x63 + x143 * x64 + x148 * x54 + x150 * x56 + x206 + x207 * x49 + x208 * x51
        x210 = x156 * x27
        x211 = x158 * x27
        x212 = (
            self.p.m1 * x160 * x49
            + self.p.m1 * x161 * x51
            + x162 * x63
            + x163 * x64
            - x165 * x61
            + x166 * x54
            + x168 * x56
        )
        x213 = x174 * x27
        x214 = x176 * x27
        x215 = 3 * pi * self.p.a * self.p.mu12 * x12 * x13 * x179 * x27 * x40 / 16
        x216 = 3 * pi * self.p.a * self.p.mu12 * x10 * x12 * x17 * x179 * x27 * x40 / 16
        x217 = 3 * pi * self.p.a * self.p.mu12 * x12 * x13 * x183 * x27 * x40 / 16
        x218 = 3 * pi * self.p.a * self.p.mu12 * x10 * x12 * x17 * x183 * x27 * x40 / 16
        x219 = x78**2
        x220 = x10**2 * x17**2
        x221 = x112 * x84 - x123 * x81 + x76 * x92 + x83 * x92
        x222 = x27 * x52
        x223 = x145 * x81
        x224 = x138 * x76 + x138 * x83 + x143 * x84 + x223
        x225 = x161 * x76 + x161 * x83 + x163 * x84 - x165 * x81
        x226 = 3 * pi * self.p.a * self.p.mu12 * x12 * x13 * x27 * x40 * x52 / 16
        x227 = x95**2
        x228 = x93**2
        x229 = x94**2
        x230 = self.p.theta3y * x229
        x231 = x100**2
        x232 = self.p.theta3z * x231
        x233 = x91**2
        x234 = x92**2
        x235 = self.p.theta3y * x100 * x94
        x236 = x235 * x93
        x237 = self.p.theta3z * x100 * x94
        x238 = -x237 * x93
        x239 = self.p.l * self.p.m3 * x100 * x117
        x240 = -x123 * x239
        x241 = x104 * x149 + x112 * x151 + x125 * x137 + x127 * x138 + x137 * x201 + x138 * x202 + x236 + x238 + x240
        x242 = self.p.theta3x * x93
        x243 = x93 * x95
        x244 = (
            self.p.m3 * x123 * x165
            + x125 * x160
            + x126 * x162
            + x127 * x161
            + x128 * x163
            + x160 * x201
            + x161 * x202
            + x230 * x243
            + x232 * x243
            - x242 * x95
        )
        x245 = self.p.l * self.p.m3 * x121
        x246 = self.p.l**2 * self.p.m3
        x247 = self.p.theta3y * x231 + self.p.theta3z * x229 + x117**2 * x231 * x246
        x248 = x137**2
        x249 = x138**2
        x250 = x235 * x95
        x251 = -x237 * x95
        x252 = -x165 * x239
        x253 = x148 * x160 + x149 * x162 + x150 * x161 + x151 * x163 + x160 * x207 + x161 * x208 + x250 + x251 + x252
        x254 = x160**2
        x255 = x161**2
        x256 = self.p.m1 * self.p.r2 * w_2x
        x257 = r_xy * w_2z - r_xz * w_2y
        x258 = self.p.m1 * self.p.r2 * w_2y
        x259 = -r_xx * w_2z + r_xz * w_2x
        x260 = self.p.m1 * self.p.r2 * w_2z
        x261 = r_xx * w_2y - r_xy * w_2x
        x262 = self.p.m1 * self.p.r2 * x1
        x263 = r_zy * w_2z - r_zz * w_2y
        x264 = w_2x * x263
        x265 = -r_zx * w_2z + r_zz * w_2x
        x266 = w_2y * x265
        x267 = r_zx * w_2y - r_zy * w_2x
        x268 = w_2z * x267
        x269 = x2 + 1
        x270 = self.p.r1 * x269
        x271 = w_1z * x270
        x272 = psi_x_dot * x17 * x6
        x273 = self.p.r1 * x272
        x274 = self.p.r2 * x272
        x275 = self.p.r2 * x269
        x276 = x24 * x275
        x277 = x25 * x275
        x278 = x26 * x275
        x279 = psi_y_dot * (-x271 - x273 - x274 - x276 - x277 - x278)
        x280 = self.p.m1 * x279 + x256 * x257 + x258 * x259 + x260 * x261 - x262 * x264 - x262 * x266 - x262 * x268
        x281 = psi_x_dot**2 * x13 * x52
        x282 = self.p.m2 * self.p.r2 * w_2x
        x283 = self.p.m2 * self.p.r2 * w_2y
        x284 = self.p.m2 * self.p.r2 * w_2z
        x285 = self.p.m2 * self.p.r2 * x1
        x286 = (
            self.p.m2 * x279
            + self.p.m2 * x281
            + x257 * x282
            + x259 * x283
            + x261 * x284
            - x264 * x285
            - x266 * x285
            - x268 * x285
        )
        x287 = r_yy * w_2z - r_yz * w_2y
        x288 = -r_yx * w_2z + r_yz * w_2x
        x289 = r_yx * w_2y - r_yy * w_2x
        x290 = self.p.m1 * self.p.r2 * x19 * x7
        x291 = psi_x_dot * x7
        x292 = self.p.r2 * x17 * x6 * x7
        x293 = -self.p.r1 * w_1z * x17 * x6 * x7 - x24 * x292 - x25 * x292 - x26 * x292 - x270 * x291 - x275 * x291
        x294 = x8 + 1
        x295 = psi_x_dot * x294
        x296 = self.p.r2 * x19 * x294
        x297 = -w_1z * x294 * x32 - x14 * x295 - x24 * x296 - x25 * x296 - x26 * x296 - x295 * x31
        x298 = psi_x_dot * x297
        x299 = (
            psi_y_dot * self.p.m1 * x293
            + self.p.m1 * x298
            - x256 * x287
            - x258 * x288
            - x260 * x289
            - x264 * x290
            - x266 * x290
            - x268 * x290
        )
        x300 = self.p.m2 * self.p.r2 * x19 * x7
        x301 = psi_y_dot * (-psi_x_dot * x60 - psi_y_dot * x80 + x293)
        x302 = psi_x_dot * (-psi_x_dot * x80 - psi_y_dot * x60 + x297)
        x303 = (
            self.p.m2 * x301
            + self.p.m2 * x302
            - x264 * x300
            - x266 * x300
            - x268 * x300
            - x282 * x287
            - x283 * x288
            - x284 * x289
        )
        x304 = psi_x_dot * (-psi_x_dot * x77 + psi_y_dot * x55)
        x305 = psi_y_dot * (psi_x_dot * x55 - psi_y_dot * x77)
        x306 = self.p.g * self.p.m2 + self.p.m2 * x304 + self.p.m2 * x305
        x307 = w_2x * x95
        x308 = w_2z * x93
        x309 = phi_x_dot + x307 - x308
        x310 = self.p.l * self.p.m3 * x100 * x309
        x311 = self.p.l * x309 * x94
        x312 = phi_y_dot * x100
        x313 = w_2y * x100
        x314 = w_2x * x93
        x315 = x314 * x94
        x316 = w_2z * x95
        x317 = x316 * x94
        x318 = x312 + x313 + x315 + x317
        x319 = self.p.l * x318
        x320 = self.p.m3 * (x311 * x95 + x319 * x93)
        x321 = self.p.m3 * (x311 * x93 - x319 * x95)
        x322 = phi_x_dot * self.p.m3
        x323 = self.p.l * x309
        x324 = -phi_y_dot * x94 - w_2y * x94 + x100 * x314 + x100 * x316
        x325 = self.p.l * x324
        x326 = phi_y_dot * self.p.m3
        x327 = x307 * x94 - x308 * x94
        x328 = self.p.l * x327
        x329 = -x314 - x316
        x330 = (
            self.p.g * self.p.m3
            + self.p.m3 * x304
            + self.p.m3 * x305
            + x263 * x321
            + x265 * x310
            + x267 * x320
            + x322 * (-x117 * x325 + x323 * (-r_zy * x94 + x100 * x119 + x100 * x120))
            + x326 * (-x117 * x328 + x122 * x329 - x319 * (-x119 - x120) + x323 * (x115 * x94 - x116 * x94))
        )
        x331 = x306 + x330
        x332 = self.p.m3 * self.p.r2 * w_2x
        x333 = self.p.m3 * self.p.r2 * w_2y
        x334 = self.p.m3 * self.p.r2 * w_2z
        x335 = self.p.m3 * self.p.r2 * x1
        x336 = (
            self.p.m3 * x279
            + self.p.m3 * x281
            + x257 * x332
            + x259 * x333
            + x261 * x334
            - x264 * x335
            - x266 * x335
            - x268 * x335
            + x287 * x321
            + x288 * x310
            + x289 * x320
            + x322 * (x323 * (-r_yy * x94 + x100 * x101 + x100 * x102) - x325 * x98)
            + x326 * (x103 * x329 - x319 * (-x101 - x102) + x323 * (x94 * x96 - x94 * x97) - x328 * x98)
        )
        x337 = self.p.m3 * self.p.r2 * x19 * x7
        x338 = (
            self.p.m3 * x301
            + self.p.m3 * x302
            + x257 * x321
            + x259 * x310
            + x261 * x320
            - x264 * x337
            - x266 * x337
            - x268 * x337
            - x287 * x332
            - x288 * x333
            - x289 * x334
            + x322 * (-x107 * x325 + x323 * (-r_xy * x94 + x100 * x109 + x100 * x110))
            + x326 * (-x107 * x328 + x111 * x329 - x319 * (-x109 - x110) + x323 * (x105 * x94 - x106 * x94))
        )
        x339 = x11 * x331 - x13 * (x286 + x336) + x18 * (x303 + x338)
        x340 = x339 * x66
        x341 = self.p.r2 * w_2x * x27
        x342 = self.p.r2 * w_2y * x27
        x343 = self.p.r2 * w_2z * x27
        x344 = self.p.r2 * x1 * x27
        x345 = psi_y_dot * x27
        x346 = x10 * x17 * x339 * x70
        x347 = (
            self.p.theta1
            * (
                -x257 * x341
                - x259 * x342
                - x261 * x343
                + x264 * x344
                + x266 * x344
                + x268 * x344
                + x345 * (x271 + x273 + x274 + x276 + x277 + x278)
            )
            + x346
        )
        x348 = self.p.r2 * x19 * x27 * x7
        x349 = x339 * x73
        x350 = (
            self.p.theta1
            * (
                -x264 * x348
                - x266 * x348
                - x268 * x348
                + x27 * x298
                - x287 * x341
                - x288 * x342
                - x289 * x343
                + x293 * x345
            )
            - x349
        )
        x351 = x27 * x347
        x352 = x27 * x350
        x353 = x318 * x324
        x354 = phi_y_dot * self.p.theta3x * x329 - self.p.theta3y * x353 + self.p.theta3z * x353
        x355 = x309 * x318
        x356 = (
            -self.p.theta3x * x355
            + self.p.theta3y * x355
            + self.p.theta3z * (phi_x_dot * (-x312 - x313 - x315 - x317) + phi_y_dot * (x100 * x307 - x100 * x308))
        )
        x357 = x100 * x356
        x358 = x309 * x324
        x359 = self.p.theta3x * x358 + self.p.theta3y * (phi_x_dot * x324 + phi_y_dot * x327) - self.p.theta3z * x358
        x360 = x359 * x94
        x361 = 1 / self.p.tau
        A[0, 0] = (
            self.p.m1 * x3
            + self.p.m1 * x9
            + self.p.m2 * x3
            + self.p.m2 * x9
            + self.p.m3 * x3
            + self.p.m3 * x9
            + self.p.theta1
            + x1 * x44
            + x42
            - x45 * x47
        )
        A[0, 1] = x1 * x72 - x45 * x75 + x57 - x59 * x62 + x67
        A[0, 2] = x1 * x87 - x45 * x89 - x59 * x82 + x79 + x86
        A[0, 3] = x1 * x133 + x113 - x114 * x123 + x130 - x135 * x45
        A[0, 4] = x1 * x156 + x144 + x146 + x153 - x158 * x45
        A[0, 5] = x1 * x174 - x114 * x165 + x164 + x171 - x176 * x45
        A[0, 6] = x1 * x181 - x103 * x16 - x111 * x22 - x114 * x122 + x180 + x182 * x45
        A[0, 7] = x1 * x185 + x140 * x16 + x142 * x22 + x146 + x184 + x186 * x45
        A[1, 0] = x187 * x69 + x188 * x51 + x57
        A[1, 1] = (
            self.p.m1 * x49**2
            + self.p.m1 * x51**2
            + self.p.m2 * x193
            + self.p.m2 * x194
            + self.p.m3 * x193
            + self.p.m3 * x194
            + x190 * x191
            + x191 * x192
            + x195 * x69
            + x196 * x51
        )
        A[1, 2] = x198 + x199 * x69 + x200 * x51
        A[1, 3] = x203 + x204 * x69 + x205 * x51
        A[1, 4] = x209 + x210 * x69 + x211 * x51
        A[1, 5] = x212 + x213 * x69 + x214 * x51
        A[1, 6] = x103 * x63 + x111 * x64 - x122 * x61 - x215 * x51 + x216 * x69
        A[1, 7] = -x140 * x63 - x142 * x64 + x206 - x217 * x51 + x218 * x69
        A[2, 0] = x188 * x52 + x79
        A[2, 1] = x196 * x52 + x198
        A[2, 2] = self.p.m1 * x189 + self.p.m2 * x219 + self.p.m3 * x219 + x190 * x220 + x192 * x220 + x200 * x52
        A[2, 3] = x135 * x222 + x221
        A[2, 4] = x211 * x52 + x224
        A[2, 5] = x176 * x222 + x225
        A[2, 6] = x111 * x84 - x122 * x81 - x179 * x226
        A[2, 7] = -x142 * x84 - x183 * x226 + x223
        A[3, 0] = -r_xx * x43 + r_yx * x46 - r_zx * x42 + x113 + x131 * x187 + x188 * x92
        A[3, 1] = -r_xx * x71 + r_yx * x74 - r_zx * x67 + x131 * x195 + x196 * x92 + x203
        A[3, 2] = -r_xx * x87 + r_yx * x88 - r_zx * x86 + x131 * x199 + x200 * x92 + x221
        A[3, 3] = (
            -r_xx * x132
            + r_yx * x134
            - r_zx * x130
            + self.p.m1 * x233
            + self.p.m1 * x234
            + self.p.m2 * x233
            + self.p.m2 * x234
            + self.p.m3 * x104**2
            + self.p.m3 * x112**2
            + self.p.m3 * x123**2
            + self.p.theta2
            + self.p.theta3x * x227
            + x131 * x204
            + x205 * x92
            + x228 * x230
            + x228 * x232
        )
        A[3, 4] = -r_xx * x155 + r_yx * x157 - r_zx * x153 + x131 * x210 + x211 * x92 + x241
        A[3, 5] = -r_xx * x173 + r_yx * x175 - r_zx * x171 + x131 * x213 + x214 * x92 + x244
        A[3, 6] = (
            -r_xx * x181
            + r_yx * x182
            - r_zx * x180
            + self.p.theta3x * x95
            + x103 * x126
            + x111 * x128
            + x123 * x245
            + x131 * x216
            - x215 * x92
        )
        A[3, 7] = (
            -r_xx * x185
            + r_yx * x186
            - r_zx * x184
            - x126 * x140
            - x128 * x142
            + x131 * x218
            - x217 * x92
            + x236
            + x238
            + x240
        )
        A[4, 0] = -r_xy * x43 + r_yy * x46 - r_zy * x42 + x138 * x188 + x144 + x154 * x187
        A[4, 1] = -r_xy * x71 + r_yy * x74 - r_zy * x67 + x138 * x196 + x154 * x195 + x209
        A[4, 2] = -r_xy * x87 + r_yy * x88 - r_zy * x86 + x138 * x200 + x154 * x199 + x224
        A[4, 3] = -r_xy * x132 + r_yy * x134 - r_zy * x130 + x138 * x205 + x154 * x204 + x241
        A[4, 4] = (
            -r_xy * x155
            + r_yy * x157
            - r_zy * x153
            + self.p.m1 * x248
            + self.p.m1 * x249
            + self.p.m2 * x248
            + self.p.m2 * x249
            + self.p.m3 * x141**2
            + self.p.m3 * x143**2
            + self.p.theta2
            + x138 * x211
            + x154 * x210
            + x247
        )
        A[4, 5] = -r_xy * x173 + r_yy * x175 - r_zy * x171 + x138 * x214 + x154 * x213 + x253
        A[4, 6] = (
            -r_xy * x181
            + r_yy * x182
            - r_zy * x180
            - x100 * x117 * x121 * x246
            + x103 * x149
            + x111 * x151
            - x138 * x215
            + x154 * x216
        )
        A[4, 7] = (
            -r_xy * x185 + r_yy * x186 - r_zy * x184 - x138 * x217 - x140 * x149 - x142 * x151 + x154 * x218 + x247
        )
        A[5, 0] = -r_xz * x43 + r_yz * x46 - r_zz * x42 + x161 * x188 + x164 + x172 * x187
        A[5, 1] = -r_xz * x71 + r_yz * x74 - r_zz * x67 + x161 * x196 + x172 * x195 + x212
        A[5, 2] = -r_xz * x87 + r_yz * x88 - r_zz * x86 + x161 * x200 + x172 * x199 + x225
        A[5, 3] = -r_xz * x132 + r_yz * x134 - r_zz * x130 + x161 * x205 + x172 * x204 + x244
        A[5, 4] = -r_xz * x155 + r_yz * x157 - r_zz * x153 + x161 * x211 + x172 * x210 + x253
        A[5, 5] = (
            -r_xz * x173
            + r_yz * x175
            - r_zz * x171
            + self.p.m1 * x254
            + self.p.m1 * x255
            + self.p.m2 * x254
            + self.p.m2 * x255
            + self.p.m3 * x162**2
            + self.p.m3 * x163**2
            + self.p.m3 * x165**2
            + self.p.theta2
            + self.p.theta3x * x228
            + x161 * x214
            + x172 * x213
            + x227 * x230
            + x227 * x232
        )
        A[5, 6] = (
            -r_xz * x181
            + r_yz * x182
            - r_zz * x180
            + x103 * x167
            + x111 * x169
            - x161 * x215
            + x165 * x245
            + x172 * x216
            - x242
        )
        A[5, 7] = (
            -r_xz * x185
            + r_yz * x186
            - r_zz * x184
            - x140 * x167
            - x142 * x169
            - x161 * x217
            + x172 * x218
            + x250
            + x251
            + x252
        )
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
        b[0] = (
            -x1 * x347
            + x14 * x280
            + x14 * x286
            + x14 * x336
            + x20 * x299
            + x20 * x303
            + x20 * x338
            - x340
            + x350 * x45
            + x59 * (self.p.g * self.p.m1 + x331)
        )
        b[1] = (
            -x280 * x49
            - x286 * x54
            - x299 * x51
            - x303 * x56
            + x306 * x60
            + x330 * x60
            - x336 * x54
            - x338 * x56
            - x351 * x69
            - x352 * x51
        )
        b[2] = -x222 * x350 - x299 * x52 - x303 * x78 + x306 * x80 + x330 * x80 - x338 * x78
        b[3] = (
            r_xx * x346
            - r_yx * x349
            + r_zx * x340
            - x104 * x336
            - x112 * x338
            - x123 * x330
            - x131 * x351
            - x280 * x91
            - x286 * x91
            - x299 * x92
            - x303 * x92
            - x352 * x92
            - x354 * x95
            - x357 * x93
            - x360 * x93
        )
        b[4] = (
            r_xy * x346
            - r_yy * x349
            + r_zy * x340
            - x100 * x359
            - x137 * x280
            - x137 * x286
            - x138 * x299
            - x138 * x303
            - x138 * x352
            - x141 * x336
            - x143 * x338
            + x145 * x330
            - x154 * x351
            + x356 * x94
        )
        b[5] = (
            r_xz * x346
            - r_yz * x349
            + r_zz * x340
            - x160 * x280
            - x160 * x286
            - x161 * x299
            - x161 * x303
            - x161 * x352
            - x162 * x336
            - x163 * x338
            - x165 * x330
            - x172 * x351
            + x354 * x93
            - x357 * x95
            - x360 * x95
        )
        b[6] = x361 * (omega_x_cmd - phi_x_dot)
        b[7] = x361 * (omega_y_cmd - phi_y_dot)

        omega_dot = np.linalg.solve(A, b)

        return omega_dot

    def _compute_r_OSi(self, state):
        """computes center of mass locations of all bodies

        args:
            state (ModelState): current state

        Returns: list of x/y/z coordinates of center of mass of lower ball, upper ball, and lever arm.
        """
        [r_xx, r_xy, r_xz, r_yx, r_yy, r_yz, r_zx, r_zy, r_zz] = state.R_IB2.reshape(9)
        [pos_x, pos_y] = state.pos
        [psi_x, psi_y] = state.psi
        [phi_x, phi_y] = state.phi

        r_OS1 = np.array([pos_x, pos_y, self.p.r1])

        r_S1S2 = np.zeros(3)
        r_S2S3 = np.zeros(3)

        x0 = self.p.r1 + self.p.r2
        x1 = x0 * cos(psi_x)
        x2 = sin(phi_x)
        x3 = cos(phi_x)
        x4 = x3 * sin(phi_y)
        x5 = x3 * cos(phi_y)
        r_S1S2[0] = x1 * sin(psi_y)
        r_S1S2[1] = -x0 * sin(psi_x)
        r_S1S2[2] = x1 * cos(psi_y)
        r_S2S3[0] = -self.p.l * (r_xx * x4 - r_xy * x2 + r_xz * x5)
        r_S2S3[1] = -self.p.l * (r_yx * x4 - r_yy * x2 + r_yz * x5)
        r_S2S3[2] = -self.p.l * (r_zx * x4 - r_zy * x2 + r_zz * x5)

        r_OS2 = r_OS1 + r_S1S2
        r_OS3 = r_OS2 + r_S2S3

        return [r_OS1, r_OS2, r_OS3]

    def _compute_e_S1S2(self, state):
        """computes the unit vector (x/y/z) pointing from lower ball center to upper ball center

        args:
            state (ModelState): current state
        returns:
            array containing unit direction (x/y/z) pointing from lower ball center to upper ball center
        """
        [psi_x, psi_y] = state.psi
        return np.array([sin(psi_y) * cos(psi_x), -sin(psi_x), cos(psi_x) * cos(psi_y)])

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
        lat = np.linspace(0, np.pi, circle_res // 2)
        x = radius * np.outer(np.cos(lon), np.sin(lat))
        z = radius * np.outer(np.sin(lon), np.sin(lat))
        y = radius * np.outer(np.ones(circle_res), np.cos(lat))

        R_IB = q_IB.rotation_matrix()

        x_rot = R_IB[0, 0] * x + R_IB[0, 1] * y + R_IB[0, 2] * z
        y_rot = R_IB[1, 0] * x + R_IB[1, 1] * y + R_IB[1, 2] * z
        z_rot = R_IB[2, 0] * x + R_IB[2, 1] * y + R_IB[2, 2] * z

        return [center[0] + x_rot, center[1] + y_rot, center[2] + z_rot]
