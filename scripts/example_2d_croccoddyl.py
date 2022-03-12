"""Simple test script for 2D Double Ball Balancer
"""
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse

import context

from model_2d.dynamic_model import ModelParam, DynamicModel
from model_2d.crocoddyl_controller import Controller
from model_2d.controller import Controller as Ctrl
from model_2d.definitions import *

parser = argparse.ArgumentParser(description="Test 2D double ball balancer")
parser.add_argument("-a", "--no-animation", help="disable animation", action="store_true")
parser.add_argument(
    "-c",
    "--contact-forces",
    help="enable visualization of contact forces in the animation",
    action="store_true")
parser.add_argument("-p", "--no-plot", help="disable plotting states", action="store_true")
parser.add_argument("-v", "--verbose", help="print simulation time", action="store_true")
args = parser.parse_args()

enable_plot = not args.no_plot
enable_animation = not args.no_animation
enable_contact_forces = args.contact_forces

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

# prepare simulation
max_sim_time = 5
sim_time = 0
sim_time_vec = [sim_time]
state_vec = [model.x]
input_vec = []
start_time = time.time()
u = 0
contact_forces = None

ctrl_mode = BETA_DOT_IDX
us, state_vec = controller.compute_ctrl_input(model.x, beta_cmd, ctrl_mode)
sim_time_vec = np.array(range(state_vec.shape[0])) * dt

contoller2 = Ctrl(param)
u2 =  [contoller2.compute_ctrl_input(x, 0) for x in state_vec]


input_vec = us
i = 0

# simulate until system is irrecoverable or max_sim_time reached
while  i<len(sim_time_vec)-1:
    # get control input
    # u = us[i]
    # i+=1

    # u = controller.compute_ctrl_input(model.x, beta_cmd)[0]

    # calculate contact forces
    contact_forces = model.compute_contact_forces(x = state_vec[i], omega_cmd=us[i])
    print(np.abs(np.cross(contact_forces[1], model._compute_e_S1S2(state_vec[i]))) / np.dot(contact_forces[1], model._compute_e_S1S2(state_vec[i])))


    if False and enable_animation:
        plt.figure(0)

        # get visualization
        vis = model.get_visualization(x = state_vec[i], omega_cmd=us[i], contact_forces=contact_forces)

        # plot
        plt.cla()
        plt.plot(*vis['lower_ball'])
        plt.plot(*vis['upper_ball'])
        plt.plot(*vis['lever_arm'])
        if True or enable_contact_forces:
            plt.arrow(*vis['F1'], head_width=0.1, color='red')
            plt.arrow(*vis['F12'], head_width=0.1, color='red')
            plt.arrow(*vis['F23'], head_width=0.1, color='red')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.axis('equal')
        plt.show(block=False)
        time_passed = time.time() - start_time
        # plt.pause(max(dt - time_passed, 0.001))
        plt.pause( 0.01)

        start_time = time.time()

    i+= 1
    # simulate one time step
    # model.simulate_step(dt, u)
    # sim_time += dt

    # # save states as matrix, sim_time and inputs as lists
    # state_vec = np.concatenate([state_vec, [model.x]])
    # sim_time_vec.append(sim_time)
    # input_vec.append(u)

    if args.verbose:
        print('sim_time: {0:.3f} s'.format(sim_time))

if enable_plot:
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
    plt.plot(sim_time_vec[:-1], input_vec, label='motor_cmd')
    plt.plot(sim_time_vec[:-1], u2[:-1], label='motor_cmd default ctrl')
    plt.xlabel('time [s]')
    plt.ylabel('omega [rad]')
    plt.legend()
    plt.title('omega')
    plt.show(block=True)
