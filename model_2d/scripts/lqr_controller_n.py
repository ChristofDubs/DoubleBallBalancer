"""Script to compute LQR-controller gain for stabilizing the 2D N Ball Balancer, based on numerical parameter values.
"""
import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.signal import place_poles
import matplotlib.pyplot as plt

import pickle

from symbolic_dynamics_n import ang, omega, omega_dot, omega_cmd, all_constants

def generateParameterSubstitutionList(N: int):
    # substitute parameters with numerical values
    sub_list = [("g", 9.81), ("tau", 0.1), ("r_l", 1.0), ("m_l", 1.0), ("theta_l", 1.0)]
    for i in range(N):
        sub_list.append(("r_{}".format(i), 1))
        # sub_list.append(("r_{}".format(i), N + 1 - i))
        sub_list.append(("m_{}".format(i), 1))
        sub_list.append(("theta_{}".format(i), 1))
    return sub_list

def computeControllerGains(N: int, verbose: bool=False):

    dyn_lin = pickle.load(open(f"linear_dynamics_{N}.p", "rb"))

    dyn_lin = dyn_lin.subs(generateParameterSubstitutionList(N))

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

    # check controlability matrix
    X = [B]
    for _ in range(dim - 1):
        X.append(np.dot(A, X[-1]))

    X = np.column_stack(X)

    assert(np.linalg.matrix_rank(X) == dim)

    # eigenvalues
    if verbose:
        print('open loop eigenvalues: \n{}'.format(np.linalg.eigvals(A)))

    # LQR controller:
    # https://en.wikipedia.org/wiki/Linear%E2%80%93quadratic_regulator
    R = np.ones([1, 1], dtype=float) * 1
    # not punishing alpha_N deviations will stabilize the system at an arbitrary alpha_N
    # state: alpha_N-1, phi, psi_0, ...,  psi_N, alpha_N-1_dot, phi_dot, psi_0_dot, ... , psi_N-1_dot
    Q = np.diag(np.concatenate([np.zeros(N + 1), np.array([4, 8]), np.array([4 * (2**x) for x in range(N - 1)])]))

    # solve continuous Riccati equation
    P = solve_continuous_are(A, B, Q, R)

    # compute controller gain
    K = np.dot(np.linalg.inv(R), np.dot(B.T, P))

    if verbose:
        # compute eigenvalues of closed loop system
        eig = np.linalg.eigvals(A - np.dot(B, K))
        print('closed loop eigenvalues: \n{}'.format(eig))

        # find minimal damping coefficient
        zeta = [np.absolute(e.real) / np.absolute(e) for e in eig if e < 0]
        print('minimal damping ratio: {}'.format(min(zeta)))
    
    return K, eig

if __name__ == '__main__':

    K, eig = computeControllerGains(2, True)

    plt.figure()
    plt.plot(eig.real, eig.imag, 'b*')
    plt.xlabel('real')
    plt.ylabel('imag')
    plt.title('poles of closed loop system')
    plt.axis('equal')
    plt.show(block=True)
