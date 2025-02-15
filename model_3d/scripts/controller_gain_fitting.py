"""Script to fit controller gains

1) For each trajectory {x, u} at a constant ball forward velocity omega_2_y, find least-square K such that u = K*x
2) For all elements K_i of K, fit a function K_i(omega_2_y) to match K_i of all trajectories

author: Christof Dubs
"""
import context


from model_3d.dynamic_model import ModelState
from model_3d.controller import projectModelState

import numpy as np
import pickle

import glob
import matplotlib.pyplot as plt

data = []

for file in glob.glob('data/data*'):
    with open(file, 'rb') as handle:
        us, state_vec, _ = pickle.load(handle)

    A = np.array([projectModelState(state)[0] for state in state_vec[:-1]])

    # K = gains that reproduce the control signals u given the projected model state: u = K*x
    K = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, us))

    # plt.figure()
    # plt.plot(us, 'b*', label="desired u")
    # plt.plot(np.dot(A, K), 'r-', label='linear fit')
    # plt.legend()
    omega_2_y = projectModelState(state_vec[0])[1][3]
    data.append([omega_2_y, K])

us = [u for u, _ in data]
# A = np.array([[1, u, u**2, u**3, u**4] for u in us])
A = np.array([[1, u**2, u**3, u**4, u**5, u**6] for u in us])
# A = np.array([[1, u**2, u**4] for u in us])
# A = np.array([[1, u, u**2] for u in us])

for i in range(len(data[0][1])):
    b = np.array([K[i][0] for _, K in data])
    x = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, b))

    print('coefficients x that map f(u)*x = K_i')
    print(x)

    plt.figure()
    plt.plot(us, b, 'b*')
    ref = np.linspace(0, 2.5, 31)
    # plt.plot(ref, x[4]*ref**4 + x[3]*ref**3 + x[2]*ref**2 + x[1]*ref + x[0])
    # plt.plot(ref, x[3]*ref**4 + x[2]*ref**3 + x[1]*ref**2 + x[0])
    plt.plot(ref, x[5] * ref**6 + x[4] * ref**5 + x[3] * ref**4 + x[2] * ref**3 + x[1] * ref**2 + x[0], 'r-')
    # plt.plot(ref, x[2]*np.cos(ref) + x[1]*ref**2 + x[0])
    # plt.plot(ref, x[2]*ref**4 + x[1]*ref**2 + x[0])
    # plt.plot(ref, x[2]*ref**2 + x[1]*ref + x[0])
    plt.xlabel('omega_2_y')
    plt.ylabel(f'gain {i}')
plt.show(block=True)
