"""Script to compute LQR-controller gain for stabilizing the 3D Double Ball Balancer, based on numerical parameter values.
"""
import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.signal import place_poles
import matplotlib.pyplot as plt

import pickle

from symbolic_dynamics_linearization import ang, omega, omega_dot, omega_cmd

use_pole_placement = True

dyn_lin = pickle.load(open("linear_dynamics.p", "rb"))

# substitute parameters with numerical values
dyn_lin = dyn_lin.subs("l", 1.0)
dyn_lin = dyn_lin.subs("m1", 1.0)
dyn_lin = dyn_lin.subs("m2", 1.0)
dyn_lin = dyn_lin.subs("m3", 1.0)
dyn_lin = dyn_lin.subs("r1", 3.0)
dyn_lin = dyn_lin.subs("r2", 2.0)
dyn_lin = dyn_lin.subs("theta1", 1.0)
dyn_lin = dyn_lin.subs("theta2", 1.0)
dyn_lin = dyn_lin.subs("theta3x", 1.0)
dyn_lin = dyn_lin.subs("theta3y", 1.0)
dyn_lin = dyn_lin.subs("theta3z", 1.0)
dyn_lin = dyn_lin.subs("g", 9.81)
dyn_lin = dyn_lin.subs("tau", 0.1)

# extract matrices such that M*x_ddot + D*x_dot + K*x = Bd*u
M = dyn_lin.jacobian(omega_dot)
D = dyn_lin.jacobian(omega)
K = dyn_lin.jacobian(ang)
Bd = dyn_lin.jacobian(omega_cmd)

# convert to first order x_ddot = A*x + B*u
A = np.zeros([16, 16], dtype=float)

A[:8, 8:] = np.eye(8)
A[8:, :8] = np.array(-M.LUsolve(K))
A[8:, 8:] = np.array(-M.LUsolve(D))

B = np.zeros([16, 2], dtype=float)

B[8:, :] = np.array(-M.LUsolve(Bd))

# eigenvalues
print('open loop eigenvalues: \n{}'.format(np.linalg.eigvals(A)))

# remove uncontrollable states of the linear system:
# angles: alpha_z, beta_z
# omega: w_1z, w_2z
# resulting state:
# [psi_x, psi_y, beta_x, beta_y, phi_x, phi_y, psi_x_dot, psi_y_dot, w_2x, w_2y, phi_x_dot, phi_y_dot]
delete_mask = (0, 5, 8, 13)

A = np.delete(A, delete_mask, 0)
A = np.delete(A, delete_mask, 1)

B = np.delete(B, delete_mask, 0)

if use_pole_placement:
    # [beta, phi, psi, beta_dot, phi_dot, psi_dot]
    poles_2D = np.array([-2.07789478e+01 + 0.j, -1.23128756e-15 + 0.j, -1.48613821e+00 + 1.18812958j, -
                         1.48613821e+00 - 1.18812958j, -5.80068041e-01 + 0.27468173j, -5.80068041e-01 - 0.27468173j])

    p = np.array([poles_2D[2],
                  poles_2D[2],
                  poles_2D[0],
                  poles_2D[0],
                  poles_2D[1],
                  poles_2D[1],
                  poles_2D[5],
                  poles_2D[5],
                  poles_2D[3],
                  poles_2D[3],
                  poles_2D[4],
                  poles_2D[4]])

    K = place_poles(A, B, p, method='YT').gain_matrix

else:
    # LQR controller:
    # https://en.wikipedia.org/wiki/Linear%E2%80%93quadratic_regulator
    R = np.eye(2, dtype=float) * 1
    q_diag = np.array([0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 0, 0])
    Q = np.diag(q_diag)

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
