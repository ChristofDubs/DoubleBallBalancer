import matplotlib.pyplot as plt
import numpy as np

import pickle

import glob
from scipy.optimize import minimize

import context
from model_3d.dynamic_model import ModelState
from model_3d.controller import projectModelState
from model_2d.dynamics_2 import StateIndex as StateIndex2D

BETA_IDX = StateIndex2D.ALPHA_1_IDX
PHI_IDX = StateIndex2D.PHI_IDX
PSI_IDX = StateIndex2D.PSI_0_IDX
BETA_DOT_IDX = StateIndex2D.ALPHA_DOT_1_IDX

processed_data = 'data/all_turn_data.pickle'

data = []

try:
    with open(processed_data, 'rb') as handle:
        data = np.array(pickle.load(handle))
        print(f'loaded data from {processed_data}')
except FileNotFoundError:
    print(f'{processed_data} not found; regenerating from turn data')
    for file in glob.glob('data/turn_data_*'):
        with open(file, 'rb') as handle:
            print(f'processing {file}')
            beta_cmd, omega_x_cmd_offset, state_vec = pickle.load(handle)

        num_data = len(state_vec)

        projected = [projectModelState(state) for state in state_vec[3 * num_data // 4:-1]]

        x = np.mean([p[0] for p in projected], axis=0)
        y = np.mean([p[1] for p in projected], axis=0)
        z = np.mean([p[2] for p in projected], axis=0)

        data.append([y[BETA_DOT_IDX], z[1], x[BETA_IDX], x[PHI_IDX], x[PSI_IDX], omega_x_cmd_offset])

    with open(processed_data, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'stored data in {processed_data}')

# generate symmetric data (negative omega_y_cmd / negative omega_x_command_offset)
data = np.row_stack([np.dot(data, np.diag(x)) for x in [[1, 1, 1, 1, 1, 1], [
                    1, -1, -1, -1, -1, -1], [-1, -1, 1, 1, 1, 1], [-1, 1, -1, -1, -1, -1]]])

omega_y = data[:, 0]
omega_z = data[:, 1]
beta_x = data[:, 2]
phi_x = data[:, 3]
psi_x = data[:, 4]
omega_x_cmd = data[:, 5]

phi_x_motor = beta_x - phi_x


# find function omega_x_cmd = f(omega_y, phi_x_motor)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

ax.scatter(omega_y, phi_x_motor, omega_x_cmd, marker='o', label="measured data")

A = np.column_stack([a * b for a in [phi_x_motor]
                     for b in [np.ones(omega_y.shape), np.abs(omega_y), omega_y ** 2]])
b = omega_x_cmd

x = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, b))

print(x)

ax.scatter(omega_y, phi_x_motor, np.dot(A, x), marker='^', label="function approximation")

ax.set_xlabel('angular y vel [rad/s]')
ax.set_ylabel('phi x motor [rad]')
ax.set_xlabel('omega_x command [rad/s]')

ax.legend()

# find function phi_x_motor_max = f(omega_y)
max_phi_x_motor = {}
for i in range(len(omega_y)):
    bucket = np.round(10 * omega_y[i]) / 10
    if bucket not in max_phi_x_motor or np.abs(phi_x_motor[i]) > max_phi_x_motor[bucket]:
        max_phi_x_motor[bucket] = np.abs(phi_x_motor[i])


omega_y_buckets = np.array(list(max_phi_x_motor.keys()))
A = np.abs(np.column_stack([np.ones(omega_y_buckets.shape), omega_y_buckets,
                            omega_y_buckets**2, omega_y_buckets**3, omega_y_buckets**4]))
b = np.array(list(max_phi_x_motor.values()))
x = None


def fun(x):
    diff = np.dot(A, x) - b
    return np.linalg.norm(diff) + 100 * np.dot(diff[diff > 0], diff[diff > 0])


# find lower bound
res = minimize(fun, 0.7 * np.linalg.solve(np.dot(A.T, A), np.dot(A.T, b)))

print(res.x)

plt.figure()
plt.plot(omega_y_buckets, b, 'b*', label="measured data")

omega_y_buckets = np.linspace(-2, 2, 101)
A = np.abs(np.column_stack([np.ones(omega_y_buckets.shape), omega_y_buckets,
                            omega_y_buckets**2, omega_y_buckets**3, omega_y_buckets**4]))
plt.plot(omega_y_buckets, np.dot(A, res.x), 'r', label='lower bound fit')

plt.xlabel('angular y vel [rad/s]')
plt.ylabel('phi x motor [rad]')
plt.legend()

# find function phi_x_motor = f(omega_y, omega_z)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

ax.scatter(omega_z / omega_y, omega_y, phi_x_motor, marker='o', label="measured data")
A = np.column_stack([a * b for a in [omega_z / omega_y, (omega_z / omega_y) * np.abs(omega_z / omega_y)]
                     for b in [np.ones(omega_y.shape), omega_y ** 2]])
b = phi_x_motor

x = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, b))

print(x)

ax.scatter(omega_z / omega_y, omega_y, np.dot(A, x), marker='^', label="function approximation")

ax.set_xlabel('angular vel ratio z/y [-]')
ax.set_ylabel('angular y vel [rad/s]')
ax.set_zlabel('phi x motor [rad]')

ax.legend()

plt.show(block=True)
