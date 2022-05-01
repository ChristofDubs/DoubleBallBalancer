"""Find an optimal trajectories for 2D Double Ball Balancer
"""
import matplotlib.pyplot as plt
import numpy as np

import context

from model_2d.dynamic_model import ModelParam, DynamicModel
from model_2d.controller_2 import Controller as Ctrl
from model_2d.definitions import *
from crocoddyl_controller import Controller

# create parameter struct
param = ModelParam()
param.l = 1.0
param.m1 = 1.0
param.m2 = 1.0
param.m3 = 1.0
param.r1 = 3.0
param.r2 = 2.0
param.tau = 0.100
param.theta1 = 1.0
param.theta2 = 1.0
param.theta3 = 1.0

# initial state
x0 = np.zeros(STATE_SIZE)
# x0[2] = 0.05
x0[0] = 0

# instantiate model
model = DynamicModel(param, x0)

# instantiate controller
controller = Controller(param)

# simulation time step
dt = 0.05

# commands
beta_cmd = 2.0

ctrl_mode = BETA_DOT_IDX
us, state_vec = controller.compute_ctrl_input(model.x, beta_cmd, ctrl_mode)
sim_time_vec = np.array(range(state_vec.shape[0])) * dt

plt.figure()
if ctrl_mode == BETA_IDX:
    plt.plot(sim_time_vec, state_vec[:, BETA_IDX], label='beta')
plt.plot(sim_time_vec, state_vec[:, PHI_IDX], label='phi')
plt.plot(sim_time_vec, state_vec[:, PSI_IDX], label='psi')
plt.xlabel('time [s]')
plt.ylabel('angles [rad]')
plt.legend()
plt.title('angles')

plt.figure()
plt.plot(sim_time_vec, state_vec[:, BETA_DOT_IDX], label='beta_dot')
plt.plot(sim_time_vec, state_vec[:, PHI_DOT_IDX], label='phi_dot')
plt.plot(sim_time_vec, state_vec[:, PSI_DOT_IDX], label='psi_dot')
plt.plot(sim_time_vec[:-1], us, label='motor_cmd')
plt.xlabel('time [s]')
plt.ylabel('omega [rad]')
plt.legend()
plt.title('omega')
plt.show(block=True)
