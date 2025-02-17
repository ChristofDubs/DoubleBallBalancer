"""Simple test script for 2D Double Ball Balancer
"""

import argparse
import time

import context  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

from model_2d.dynamics_2 import StateIndex
from model_2d.param import getDefaultParam

N = 2

exec(f"from model_2d.dynamics_{N} import DynamicModel, StateIndex")
exec(f"from model_2d.controller_{N} import Controller")

parser = argparse.ArgumentParser(description="Test 2D double ball balancer")
parser.add_argument("-a", "--no-animation", help="disable animation", action="store_true")
parser.add_argument(
    "-c", "--contact-forces", help="enable visualization of contact forces in the animation", action="store_true"
)
parser.add_argument("-p", "--no-plot", help="disable plotting states", action="store_true")
parser.add_argument("-v", "--verbose", help="print simulation time", action="store_true")
args = parser.parse_args()

enable_plot = not args.no_plot
enable_animation = not args.no_animation
enable_contact_forces = args.contact_forces

# create parameter struct
model_param = getDefaultParam(N)

# initial state
x0 = np.zeros(StateIndex.NUM_STATES)

# instantiate model
model = DynamicModel(model_param, x0)  # noqa: F821

# instantiate controller
controller = Controller(model_param)  # noqa: F821

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
u = 0
contact_forces = None

# simulate until system is irrecoverable or max_sim_time reached
while model.is_recoverable(contact_forces=contact_forces, omega_cmd=u) and sim_time < max_sim_time:
    # get control input
    u = controller.compute_ctrl_input(model.x, beta_cmd, mode=controller.ANGLE_MODE)

    # calculate contact forces
    contact_forces = model.compute_contact_forces(omega_cmd=u)

    if enable_animation:
        plt.figure(0)

        # get visualization
        vis = model.get_visualization(contact_forces=contact_forces)

        # plot
        plt.cla()
        for i in range(N + 1):
            plt.plot(*vis[str(i)])
            if enable_contact_forces:
                plt.arrow(*vis[f"F{i}"], head_width=0.1, color="red")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.axis("equal")
        plt.show(block=False)
        time_passed = time.time() - start_time
        plt.pause(max(dt - time_passed, 0.001))

        start_time = time.time()

    # simulate one time step
    model.simulate_step(dt, u)
    sim_time += dt

    # save states as matrix, sim_time and inputs as lists
    state_vec = np.concatenate([state_vec, [model.x]])
    sim_time_vec.append(sim_time)
    input_vec.append(u)

    if args.verbose:
        print("sim_time: {0:.3f} s".format(sim_time))

if enable_plot:
    plt.figure()
    for x in list(StateIndex)[: StateIndex.NUM_STATES // 2]:
        plt.plot(sim_time_vec, state_vec[:, x.value], label=x.name)
    plt.xlabel("time [s]")
    plt.ylabel("angles [rad]")
    plt.legend()
    plt.title("angles")

    plt.figure()
    for x in list(StateIndex)[StateIndex.NUM_STATES // 2 : StateIndex.NUM_STATES]:
        plt.plot(sim_time_vec, state_vec[:, x.value], label=x.name)
    plt.plot(sim_time_vec[:-1], input_vec, label="motor_cmd")
    plt.xlabel("time [s]")
    plt.ylabel("omega [rad]")
    plt.legend()
    plt.title("omega")
    plt.show(block=True)
