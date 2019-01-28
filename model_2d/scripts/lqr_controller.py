"""Script to compute LQR-controller gain for stabilizing the 2D Double Ball Balancer, based on numerical parameter values.
"""
import sympy as sp
import numpy as np
from scipy.linalg import solve_continuous_are
import matplotlib.pyplot as plt

# angles
alpha, beta, phi, psi = sp.symbols('alpha beta phi psi')
ang = sp.Matrix([beta, phi, psi])

# angular velocities
alpha_dot, beta_dot, phi_dot, psi_dot = sp.symbols('alpha_d beta_dot phi_dot psi_dot')
omega = sp.Matrix([beta_dot, phi_dot, psi_dot])

# angular accelerations
beta_dd, phi_dd, psi_dd = sp.symbols('beta_dd phi_dd psi_dd')
omega_dot = sp.Matrix([beta_dd, phi_dd, psi_dd])

# input
omega_cmd = sp.symbols('omega_cmd')

# parameter
l, m1, m2, m3, r1, r2, tau, theta1, theta2, theta3 = (1, 1, 1, 1, 3, 2, 0.1, 1, 1, 1)

# constants
g = 9.81

dyn_lin = sp.Matrix([[(beta_dd * (r1**2 * (l * m3 * r2 + m1 * r2**2 + m2 * r2**2 + m3 * r2**2 + theta2) + r2**2 * theta1) - psi_dd * (r1 + r2) * (r1**2 * (2 * l * m3 + m1 * r2 + 2 * m2 * r2 + 2 * m3 * r2) + r2 * theta1) + r1**2 * (g * l * m3 * phi + phi_dd * (l**2 * m3 + l * m3 * r2 + theta3))) /
                      r1**2], [(r1 + r2) * (-beta_dd * r2 * (r1**2 * (m1 + 2 * m2 + 2 * m3) + theta1) + psi_dd * (r1 + r2) * (r1**2 * (m1 + 4 * m2 + 4 * m3) + theta1) - r1**2 * (g * psi * (m2 + m3) + 2 * l * m3 * phi_dd)) / r1**2], [(-beta_dot - omega_cmd + phi_dot + tau * (-beta_dd + phi_dd)) / tau]])

# extract matrices such that M*x_ddot + D*x_dot + K*x = Bd*u
M = dyn_lin.jacobian(omega_dot)
D = dyn_lin.jacobian(omega)
K = dyn_lin.jacobian(ang)
Bd = dyn_lin.diff(omega_cmd, 1)

# convert to first order x_ddot = A*x + B*u
A = np.zeros([6, 6], dtype=float)

A[:3, 3:] = np.eye(3)
A[3:, :3] = np.array(-M.inv() * K)
A[3:, 3:] = np.array(-M.inv() * D)

B = np.zeros([6, 1], dtype=float)

B[3:] = np.array(-M.inv() * Bd)

# eigenvalues
print('open loop eigenvalues: \n{}'.format(np.linalg.eigvals(A)))

# LQR controller:
# https://en.wikipedia.org/wiki/Linear%E2%80%93quadratic_regulator
R = np.ones([1, 1], dtype=float) * 1
# not punishing beta deviations will stabilize the system at an arbitrary beta
Q = np.diag(np.array([0, 0, 0, 4, 8, 4], dtype=float))

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
