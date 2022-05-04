"""Find an optimal trajectories for 2D Double Ball Balancer
"""
import matplotlib.pyplot as plt
import numpy as np

import context

from model_2d.param import getDefaultParam
from model_2d.dynamics_2 import DynamicModel, StateIndex
from crocoddyl_controller import Controller

# create parameter struct
param = getDefaultParam(2)

# initial state
x0 = np.zeros(StateIndex.NUM_STATES)
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

ctrl_mode = controller.VELOCITY_MODE
us, state_vec = controller.compute_ctrl_input(model.x, beta_cmd, ctrl_mode)
sim_time_vec = np.array(range(state_vec.shape[0])) * dt

plt.figure()
for x in StateIndex:
    if x.value > StateIndex.NUM_STATES // 2:
        break
    if x.value != StateIndex.ALPHA_1_IDX or ctrl_mode == controller.ANGLE_MODE:
        plt.plot(sim_time_vec, state_vec[:, x.value], label=x.name)
plt.xlabel('time [s]')
plt.ylabel('angles [rad]')
plt.legend()
plt.title('angles')

plt.figure()
for x in StateIndex:
    if x.value < StateIndex.NUM_STATES // 2 or x.value == StateIndex.NUM_STATES:
        continue
    plt.plot(sim_time_vec, state_vec[:, x.value], label=x.name)
plt.plot(sim_time_vec[:-1], us, label='motor_cmd')
plt.xlabel('time [s]')
plt.ylabel('omega [rad]')
plt.legend()
plt.title('omega')
plt.show(block=True)
