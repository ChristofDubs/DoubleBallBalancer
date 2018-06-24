"""Script to compute LQR-controller gain for stabilizing the 2D Double Ball Balancer, based on numerical parameter values.
"""
import sympy as sp
import numpy as np
from scipy.linalg import solve_continuous_are
import matplotlib.pyplot as plt
import copy


# angles
alpha_z, beta_x, beta_y, beta_z, phi_x, phi_y, phi_z, psi_x, psi_y = sp.symbols(
    'alpha_z beta_x beta_y beta_z phi_x phi_y phi_z psi_x psi_y')

# angular velocities
phi_x_dot, phi_y_dot, phi_z_dot = sp.symbols('phi_x_dot phi_y_dot phi_z_dot')
psi_x_dot, psi_y_dot = sp.symbols('psi_x_dot psi_y_dot')
w_1z = sp.symbols('w_1z')
w_2x, w_2y, w_2z = sp.symbols('w_2x w_2y w_2z')
w_3x, w_3y, w_3z = sp.symbols('w_3x w_3y w_3z')


# angular accelerations
w_1_dot_z, w_3_dot_x, w_3_dot_y, w_3_dot_z, psi_x_ddot, psi_y_ddot, w_2_dot_x, w_2_dot_y, w_2_dot_z = sp.symbols(
    'w_1_dot_z w_3_dot_x w_3_dot_y w_3_dot_z psi_x_ddot psi_y_ddot w_2_dot_x w_2_dot_y w_2_dot_z')


omega_dot = sp.Matrix([w_1_dot_z, psi_x_ddot, psi_y_ddot, w_2_dot_x,
                       w_2_dot_y, w_2_dot_z, w_3_dot_x, w_3_dot_y, w_3_dot_z])
omega = sp.Matrix([w_1z, psi_x_dot, psi_y_dot, w_2x, w_2y, w_2z, w_3x, w_3y, w_3z])
ang = sp.Matrix([alpha_z, psi_x, psi_y, beta_x, beta_y, beta_z, phi_x, phi_y, phi_z])

# inputs
Tx, Ty, Tz = sp.symbols('Tx Ty Tz')
T = sp.Matrix([Tx, Ty, Tz])

# parameter
l, m1, m2, m3, r1, r2, theta1, theta2, theta3x, theta3y, theta3z = (
    1.0, 1.0, 1.0, 1.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0)

# constants
g = 9.81

# linearized dynamics
dyn_lin = sp.Matrix([[theta1 * w_1_dot_z],
                     [l * m3 * w_3_dot_x * (-2 * r1 - 2 * r2) + psi_x * (-g * m2 * (r1 + r2) - g * m3 * (r1 + r2)) + psi_x_ddot * (m1 * (r1 + r2)**2 + m2 * (-2 * r1 - 2 * r2)**2 + m3 * (-2 * r1 - 2 * r2)**2 + theta1 * (r1 + r2)**2 / r1**2) + w_2_dot_x * (-m1 * r2 * (r1 + r2) + m2 * r2 * (-2 * r1 - 2 * r2) + m3 * r2 * (-2 * r1 - 2 * r2) - r2 * theta1 * (r1 + r2) / r1**2)],
                     [-l * m3 * w_3_dot_y * (2 * r1 + 2 * r2) + psi_y * (-g * m2 * (r1 + r2) - g * m3 * (r1 + r2)) + psi_y_ddot * (m1 * (r1 + r2)**2 + m2 * (2 * r1 + 2 * r2)**2 + m3 * (2 * r1 + 2 * r2)**2 + theta1 * (r1 + r2)**2 / r1**2) + w_2_dot_y * (-m1 * r2 * (r1 + r2) - m2 * r2 * (2 * r1 + 2 * r2) - m3 * r2 * (2 * r1 + 2 * r2) - r2 * theta1 * (r1 + r2) / r1**2)],
                     [Tx + l * m3 * r2 * w_3_dot_x + psi_x_ddot * (-m1 * r2 * (r1 + r2) + m2 * r2 * (-2 * r1 - 2 * r2) + m3 * r2 * (-2 * r1 - 2 * r2) - r2 * theta1 * (r1 + r2) / r1**2) + w_2_dot_x * (m1 * r2**2 + m2 * r2**2 + m3 * r2**2 + theta2 + r2**2 * theta1 / r1**2)],
                     [Ty + l * m3 * r2 * w_3_dot_y + psi_y_ddot * (-m1 * r2 * (r1 + r2) - m2 * r2 * (2 * r1 + 2 * r2) - m3 * r2 * (2 * r1 + 2 * r2) - r2 * theta1 * (r1 + r2) / r1**2) + w_2_dot_y * (m1 * r2**2 + m2 * r2**2 + m3 * r2**2 + theta2 + r2**2 * theta1 / r1**2)],
                     [Tz + theta2 * w_2_dot_z],
                     [-Tx + g * l * m3 * phi_x + l * m3 * psi_x_ddot * (-2 * r1 - 2 * r2) + l * m3 * r2 * w_2_dot_x + w_3_dot_x * (l**2 * m3 + theta3x)],
                     [-Ty + g * l * m3 * phi_y - l * m3 * psi_y_ddot * (2 * r1 + 2 * r2) + l * m3 * r2 * w_2_dot_y + w_3_dot_y * (l**2 * m3 + theta3y)],
                     [-Tz + theta3z * w_3_dot_z]])


# extract matrices such that M*x_ddot + D*x_dot + K*x = Bd*u
M = dyn_lin.jacobian(omega_dot)
D = dyn_lin.jacobian(omega)
K = dyn_lin.jacobian(ang)
Bd = dyn_lin.jacobian(T)

# convert to first order x_ddot = A*x + B*u
A = np.zeros([18, 18], dtype=float)

A[:9, 9:] = np.eye(9)
A[9:, :9] = np.array(-M.LUsolve(K))
A[9:, 9:] = np.array(-M.LUsolve(D))

B = np.zeros([18, 3], dtype=float)

B[9:, :] = np.array(-M.LUsolve(Bd))

# eigenvalues
print('open loop eigenvalues: \n{}'.format(np.linalg.eigvals(A)))

# LQR controller:
# https://en.wikipedia.org/wiki/Linear%E2%80%93quadratic_regulator
R = np.eye(3, dtype=float) * 0.05

# remove uncontrollable / irrelevant states for stabilization:
# angles: alpha_z, beta_x, beta_y, beta_z, phi_z
# omega: w_1z, w_2z,w_3z
# resulting state:
# [psi_x, psi_y, phi_x, phi_y, psi_x_dot, psi_y_dot, w_2x, w_2y, w_3x, w_3y]
delete_mask = (0, 3, 4, 5, 8, 9, 14, 17)

A = np.delete(A, delete_mask, 0)
A = np.delete(A, delete_mask, 1)

B = np.delete(B, delete_mask, 0)

Q = np.diag(np.array([1, 1, 0.5, 0.5, 2, 2, 1, 1, 8, 8], dtype=float))

# eigenvalues
print('open loop eigenvalues: \n{}'.format(np.linalg.eigvals(A)))

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
