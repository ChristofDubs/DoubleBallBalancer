"""Simple test script for 2D Double Ball Balancer
"""
from dynamic_model_2d import ModelParam, DynamicModel
import matplotlib.pyplot as plt
import numpy as np

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
x0 = np.array([0, 0.8, 0, 0, 0, 0])

# instantiate model
model = DynamicModel(param, x0)

# simulation time step
dt = 0.05

# speed command
u = -1

plt.figure()

# simulate until system is irrecoverable
while not model.is_irrecoverable():
    # simulate
    model.simulate_step(dt, u)

    # get visualization
    vis = model.get_visualization()

    # plot
    plt.cla()
    plt.plot(*vis['lower_ball'])
    plt.plot(*vis['upper_ball'])
    plt.plot(*vis['lever_arm'])
    plt.axis('equal')
    plt.show(block=False)
    plt.pause(dt)
