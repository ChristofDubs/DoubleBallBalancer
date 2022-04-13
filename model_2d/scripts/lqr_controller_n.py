"""Script to compute LQR-controller gain for stabilizing the 2D N Ball Balancer, based on numerical parameter values.
"""
import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.signal import place_poles
import matplotlib.pyplot as plt

import pickle

from symbolic_dynamics_n import ang, omega, omega_dot, omega_cmd, all_constants

use_pole_placement = True

dyn_lin = pickle.load(open("linear_dynamics.p", "rb"))

# substitute parameters with numerical values
dyn_lin = dyn_lin.subs("l", 1.0)
dyn_lin = dyn_lin.subs("r_0", 4.0)
dyn_lin = dyn_lin.subs("r_1", 3.0)
dyn_lin = dyn_lin.subs("r_2", 2.0)
dyn_lin = dyn_lin.subs("m_l", 1.0)
dyn_lin = dyn_lin.subs("m_0", 1.0)
dyn_lin = dyn_lin.subs("m_1", 1.0)
dyn_lin = dyn_lin.subs("m_2", 1.0)
dyn_lin = dyn_lin.subs("theta_l", 1.0)
dyn_lin = dyn_lin.subs("theta_0", 1.0)
dyn_lin = dyn_lin.subs("theta_1", 1.0)
dyn_lin = dyn_lin.subs("theta_2", 1.0)
dyn_lin = dyn_lin.subs("g", 9.81)
dyn_lin = dyn_lin.subs("tau", 0.1)

assert(dyn_lin.jacobian(all_constants).is_zero_matrix)

# extract matrices such that M*x_ddot + D*x_dot + K*x = Bd*u
M = dyn_lin.jacobian(omega_dot)
D = dyn_lin.jacobian(omega)
K = dyn_lin.jacobian(ang)
Bd = dyn_lin.diff(omega_cmd)

# convert to first order x_ddot = A*x + B*u
dim = dyn_lin.shape[0]

A = np.zeros([2 * dim, 2 * dim], dtype=float)

A[:dim, dim:] = np.eye(dim)
A[dim:, :dim] = np.array(-M.LUsolve(K))
A[dim:, dim:] = np.array(-M.LUsolve(D))

B = np.zeros([2 * dim, 1], dtype=float)

B[dim:, :] = np.array(-M.LUsolve(Bd))

X = [B]
for _ in range(dim - 1):
    X.append(np.dot(A, X[-1]))

X = np.column_stack(X)

assert(np.linalg.matrix_rank(X) == dim)

# eigenvalues
print('open loop eigenvalues: \n{}'.format(np.linalg.eigvals(A)))


# LQR controller:
# https://en.wikipedia.org/wiki/Linear%E2%80%93quadratic_regulator
R = np.ones([1, 1], dtype=float) * 1
# not punishing alpha_N deviations will stabilize the system at an arbitrary alpha_N
# state: alpha_2, psi_0, psi_1, phi, alpha_2_dot, psi_0_dot, psi_1_dot, phi_dot
Q = np.diag(np.array([0, 0, 0, 0, 4, 8, 16, 4], dtype=float))

# solve continuous Riccati equation
P = solve_continuous_are(A, B, Q, R)

# compute controller gain
K = np.dot(np.linalg.inv(R), np.dot(B.T, P))

# compute eigenvalues of closed loop system
eig = np.linalg.eigvals(A - np.dot(B, K))
print('closed loop eigenvalues: \n{}'.format(eig))

# find minimal damping coefficient
zeta = [np.absolute(e.real) / np.absolute(e) for e in eig if e < 0]
print('minimal damping ratio: {}'.format(min(zeta)))

plt.figure()
plt.plot(eig.real, eig.imag, 'b*')
plt.xlabel('real')
plt.ylabel('imag')
plt.title('poles of closed loop system')
plt.axis('equal')
plt.show(block=True)
