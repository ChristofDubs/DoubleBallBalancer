"""Simple test script for 2D Double Ball Balancer
"""
import matplotlib.pyplot as plt
import numpy as np
import time

from dynamic_model_2d import ModelParam, DynamicModel
from controller_2d import Controller
from definitions_2d import *

print_sim_time = False
plot_visualization = True
plot_states = True

# create parameter struct
param = ModelParam()
param.l = 1
param.m1 = 1
param.m2 = 1
param.m3 = 1
param.r1 = 3
param.r2 = 2
param.tau = 0.100
param.theta1 = 1
param.theta2 = 1
param.theta3 = 1

# initial state
x0 = np.zeros(STATE_SIZE)

# instantiate model
model = DynamicModel(param, x0)

# instantiate controller
controller = Controller(param)

# simulation time step
dt = 0.05

# commands
beta_cmd = 16 * np.pi

# prepare simulation
max_sim_time = 30
sim_time = 0
sim_time_vec = [sim_time]
state_vec = [model.x]
input_vec = []
start_time = time.time()

# simulate until system is irrecoverable or max_sim_time reached
while not model.is_irrecoverable() and sim_time < max_sim_time:
    if plot_visualization:
        plt.figure(0)

        # get visualization
        vis = model.get_visualization()

        # plot
        plt.cla()
        plt.plot(*vis['lower_ball'])
        plt.plot(*vis['upper_ball'])
        plt.plot(*vis['lever_arm'])
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.axis('equal')
        plt.show(block=False)
        time_passed = time.time() - start_time
        plt.pause(max(dt - time_passed, 0.001))

        start_time = time.time()

    # get control input
    u = controller.compute_ctrl_input(model.x, beta_cmd)

    # simulate one time step
    model.simulate_step(dt, u)
    sim_time += dt

    # save states as matrix, sim_time and inputs as lists
    state_vec = np.concatenate([state_vec, [model.x]])
    sim_time_vec.append(sim_time)
    input_vec.append(u)

    if print_sim_time:
        print('sim_time: {0:.3f} s'.format(sim_time))

if plot_states:
    plt.figure()
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
    plt.plot(sim_time_vec[:-1], input_vec, label='motor_cmd')
    plt.xlabel('time [s]')
    plt.ylabel('omega [rad]')
    plt.legend()
    plt.title('omega')
    plt.show(block=True)
