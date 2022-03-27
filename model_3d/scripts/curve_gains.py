import matplotlib.pyplot as plt
import numpy as np

import pickle

import glob
from scipy.optimize import minimize

import context
from model_3d.dynamic_model import ModelState
from model_3d.controller import projectModelState
from model_2d.definitions import BETA_IDX, PHI_IDX, PSI_IDX, BETA_DOT_IDX, PHI_DOT_IDX, PSI_DOT_IDX

data = []

for file in glob.glob('disabled_turn_data_*'):
    with open(file, 'rb') as handle:
        beta_cmd, omega_x_cmd_offset, state_vec = pickle.load(handle)

    num_data = len(state_vec)

    projected = [projectModelState(state) for state in state_vec[3 * num_data // 4:-1]]

    x = np.mean([p[0] for p in projected], axis=0)
    y = np.mean([p[1] for p in projected], axis=0)
    z = np.mean([p[2] for p in projected], axis=0)

    data.append([y[BETA_DOT_IDX], z[1], x[BETA_IDX], x[PHI_IDX], x[PSI_IDX], omega_x_cmd_offset])

    # with open('all_turn_data.pickle', 'wb') as handle:
    #     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open('all_turn_data.pickle', 'rb') as handle:
    data2 = pickle.load(handle)

data = np.array(data2)

# generate symmetric data (negative omega_y_cmd / negative omega_x_command_offset)
data = np.row_stack([np.dot(data, np.diag(x)) for x in [[1, 1, 1, 1, 1, 1], [
                    1, -1, -1, -1, -1, -1], [-1, -1, 1, 1, 1, 1], [-1, 1, -1, -1, -1, -1]]])

omega_y = data[:, 0]
omega_z = data[:, 1]
beta_x = data[:, 2]
phi_x = data[:, 3]
psi_x = data[:, 4]
omega_x_cmd = data[:, 5]

total_x = beta_x - phi_x

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

ax.scatter(omega_y, total_x, omega_x_cmd, marker='o')

A = np.column_stack([a * b for a in [total_x]
                     for b in [np.ones(omega_y.shape), np.abs(omega_y), omega_y ** 2]])
b = omega_x_cmd

x = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, b))

print(x)

ax.scatter(omega_y, total_x, np.dot(A, x), marker='^')

max_total_x = {}
for i in range(len(omega_y)):
    bucket = np.round(10 * omega_y[i]) / 10
    if bucket not in max_total_x or np.abs(total_x[i]) > max_total_x[bucket]:
        max_total_x[bucket] = np.abs(total_x[i])


omega_y_buckets = np.array(list(max_total_x.keys()))
A = np.abs(np.column_stack([np.ones(omega_y_buckets.shape), omega_y_buckets,
                            omega_y_buckets**2, omega_y_buckets**3, omega_y_buckets**4]))
b = np.array(list(max_total_x.values()))
x = None


def fun(x):
    diff = np.dot(A, x) - b
    return np.linalg.norm(diff) + 100 * np.dot(diff[diff > 0], diff[diff > 0])


# find lower bound
res = minimize(fun, 0.7 * np.linalg.solve(np.dot(A.T, A), np.dot(A.T, b)))

print(res.x)

plt.figure()
plt.plot(omega_y_buckets, b, 'b*')

omega_y_buckets = np.linspace(-2, 2, 101)
A = np.abs(np.column_stack([np.ones(omega_y_buckets.shape), omega_y_buckets,
                            omega_y_buckets**2, omega_y_buckets**3, omega_y_buckets**4]))
plt.plot(omega_y_buckets, np.dot(A, res.x), 'r')

plt.show(block=True)
