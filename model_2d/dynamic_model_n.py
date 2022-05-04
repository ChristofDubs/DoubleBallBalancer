"""Dynamic Model Base Class for 2D version of N-Ball Balancer

author: Christof Dubs
"""
from abc import ABC, abstractmethod
import itertools

import numpy as np
from scipy.integrate import odeint


class NBallDynamicModel(ABC):
    def __init__(self, state_size: int, params: dict, x0: np.ndarray):
        self.state_size = state_size
        self.param = params
        if not self.set_state(x0):
            self.x = np.zeros(self.state_size)

    @abstractmethod
    def computeOmegaDot(self, x, param, omega_cmd):
        pass

    @abstractmethod
    def computeContactForces(self, x, param, omega_cmd):
        pass

    @abstractmethod
    def computePositions(self, x, param):
        pass

    @abstractmethod
    def computeBallAngles(self, x, param):
        pass

    def compute_contact_forces(self, x=None, omega_cmd=0):
        if x is None:
            x = self.x
        return self.computeContactForces(x, self.param, omega_cmd)

    def set_state(self, x0: np.ndarray):
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
        if len(x0_flat) != self.state_size:
            print(
                'called set_state with array of length {} instead of {}. Ignoring.'.format(
                    len(x0_flat), self.state_size))
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
        h = self.state_size // 2
        if not self.is_recoverable(ignore_force_check=True):
            return np.concatenate([x[h:], -100 * x[h:]])

        omega_dot = self.computeOmegaDot(x, self.param, u)
        return np.concatenate([x[h:], omega_dot.flatten()])

    def is_recoverable(
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
            bool: False if state is irrecoverable, True if recoverable
        """
        if x is None:
            x = self.x

        # any ball expect bottom ball touching the ground
        r_OS_i = self.computePositions(x, self.param)
        N = len(r_OS_i) - 1
        for i in range(1, N):
            if r_OS_i[i][1] < self.param[f'r_{i}']:
                return False

        # any ball touching another ball apart from the immediate neighbors
        for i in range(N):
            for j in range(i + 2, N):
                if np.linalg.norm(r_OS_i[j] - r_OS_i[i]) < self.param[f'r_{i}'] + self.param[f'r_{j}']:
                    return False

        # lift off: contact force between two balls  <= 0
        if not ignore_force_check:
            if contact_forces is None:
                contact_forces = self.computeContactForces(x, self.param, omega_cmd)

            assert(len(contact_forces) == N + 1)
            for i in range(N):
                up_dir = np.array([0, 1, 0]) if i == 0 else r_OS_i[i] - r_OS_i[i - 1]
                if np.dot(contact_forces[i].flatten(), up_dir.flatten()) <= 0:
                    return False

        return True

    def get_visualization(self, x=None, contact_forces=None, omega_cmd=None):
        """Get visualization of the system for plotting

        Usage example:
            v = model.get_visualization()
            plt.plot(*v['0'])
            plt.arrow(*vis['F1'])

        args:
            x (numpy.ndarray, optional): state. If not specified, the internal state is used
            contact_forces(list(numpy.ndarray), optional): contact forces [N]. If not specified, will be internally calculated
            omega_cmd (optional): motor speed command [rad/s] used for contact force calculation if contact_forces are not specified

        Returns:
            dict: dictionary with keys "0", "1", ... , "N" and "F0", "F1", ... , "FN". The value for each key is a list with two elements: a list of x coordinates, and a list of y coordinates.
        """
        if x is None:
            x = self.x
        if contact_forces is None:
            contact_forces = self.compute_contact_forces(x, omega_cmd)

        vis = {}

        r_OS_i = self.computePositions(x, self.param)
        alpha_i = self.computeBallAngles(x, self.param)[0][0]

        N = len(alpha_i)

        for i in range(N):
            vis[str(i)] = self._compute_ball_visualization(r_OS_i[i].flatten(), self.param[f'r_{i}'], alpha_i[i])

        vis[str(N)] = [np.array([r_OS_i[-2][i], r_OS_i[-1][i]]) for i in range(2)]

        force_scale = 0.05
        vis['F0'] = list(itertools.chain.from_iterable(
            [np.array([r_OS_i[0][0][0], 0]), force_scale * contact_forces[0][:2].flatten()]))

        for i in range(1, N):
            j = i - 1
            r_i = self.param[f'r_{i}']
            r_j = self.param[f'r_{j}']

            contact_pt = r_OS_i[j] + (r_OS_i[i] - r_OS_i[j]) * r_j / (r_i + r_j)
            vis[f'F{i}'] = list(itertools.chain.from_iterable([contact_pt[:2].flatten(), force_scale * contact_forces[i][:2].flatten()]))

        vis[f'F{N}'] = list(itertools.chain.from_iterable(
            [r_OS_i[N - 1][:2].flatten(), force_scale * contact_forces[-1][:2].flatten()]))

        return vis

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
