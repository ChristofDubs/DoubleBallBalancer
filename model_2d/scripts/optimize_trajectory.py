"""Find optimal trajectories for 2D Double Ball Balancer
"""
import context  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from crocoddyl_controller import Controller

from model_2d.dynamics_2 import DynamicModel, StateIndex
from model_2d.param import getDefaultParam

# create parameter struct
param = getDefaultParam(2)

# initial state
x0 = np.zeros(StateIndex.NUM_STATES)
# x0[StateIndex.PHI_IDX] = 0.05

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
for x in list(StateIndex)[:StateIndex.NUM_STATES // 2]:
    if x.value != StateIndex.ALPHA_1_IDX or ctrl_mode == controller.ANGLE_MODE:
        plt.plot(sim_time_vec, state_vec[:, x.value], label=x.name)
plt.xlabel('time [s]')
plt.ylabel('angles [rad]')
plt.legend()
plt.title('angles')

plt.figure()
for x in list(StateIndex)[StateIndex.NUM_STATES // 2: StateIndex.NUM_STATES]:
    plt.plot(sim_time_vec, state_vec[:, x.value], label=x.name)
plt.plot(sim_time_vec[:-1], us, label='motor_cmd')
plt.xlabel('time [s]')
plt.ylabel('omega [rad]')
plt.legend()
plt.title('omega')
plt.show(block=True)
