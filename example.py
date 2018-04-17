"""Simple test script for 2D Double Ball Balancer
"""
from dynamic_model_2d import ModelParam, DynamicModel
import matplotlib.pyplot as plt
import numpy as np
import time

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
x0 = np.zeros(DynamicModel.STATE_SIZE)
x0[DynamicModel.PSI_IDX] = 0.2

# instantiate model
model = DynamicModel(param, x0)

# simulation time step
dt = 0.05

# define control law


def lqr_control_law(x):
    K = np.array([[2.67619260e-15, 1.03556079e+01, -4.73012271e+01,
                   3.23606798e+00, 6.05877477e-01, -3.53469304e+01]])
    return -np.dot(K, x)


# prepare simulation
max_sim_time = 10
sim_time = 0
sim_time_vec = [sim_time]
state_vec = [model.x]
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
    u = lqr_control_law(model.x)

    # simulate one time step
    model.simulate_step(dt, u)
    sim_time += dt

    # save states as matrix
    state_vec = np.concatenate([state_vec, [model.x]])
    sim_time_vec.append(sim_time)

    if print_sim_time:
        print('sim_time: {0:.3f} s'.format(sim_time))

if plot_states:
    plt.figure()
    plt.plot(sim_time_vec, state_vec[:, model.BETA_IDX], label='beta')
    plt.plot(sim_time_vec, state_vec[:, model.PHI_IDX], label='phi')
    plt.plot(sim_time_vec, state_vec[:, model.PSI_IDX], label='psi')
    plt.xlabel('time [s]')
    plt.ylabel('angles [rad]')
    plt.legend()
    plt.title('angles')

    plt.figure()
    plt.plot(sim_time_vec, state_vec[:, model.BETA_DOT_IDX], label='beta_dot')
    plt.plot(sim_time_vec, state_vec[:, model.PHI_DOT_IDX], label='phi_dot')
    plt.plot(sim_time_vec, state_vec[:, model.PSI_DOT_IDX], label='psi_dot')
    plt.xlabel('time [s]')
    plt.ylabel('omega [rad]')
    plt.legend()
    plt.title('omega')
    plt.show(block=True)
