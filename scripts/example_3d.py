"""Simple test script for 3D Double Ball Balancer
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for resolving projection='3d'
import numpy as np
import time
import copy
import argparse

import context

from model_3d.dynamic_model import ModelParam, DynamicModel, ModelState
from model_3d.controller import Controller

parser = argparse.ArgumentParser(description="Test double ball balancer")
parser.add_argument("-a", "--no-animation", help="disable animation", action="store_true")
parser.add_argument(
    "-c",
    "--contact-forces",
    help="enable visualization of contact forces in the animation",
    action="store_true")
parser.add_argument("-g", "--gif", help="create gif of animation", action="store_true")
parser.add_argument("-p", "--no-plot", help="disable plotting states", action="store_true")
parser.add_argument("-v", "--verbose", help="print simulation time", action="store_true")
args = parser.parse_args()

enable_plot = not args.no_plot
enable_animation = not args.no_animation
enable_contact_forces = args.contact_forces

# create parameter struct
param = ModelParam()
param.l = 1.0
param.r1 = 3.0
param.r2 = 2.0

# initial state
x0 = ModelState()
# x0.q2 = np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])
x0.phi_x = 1

# instantiate model
model = DynamicModel(param, x0)
controller = Controller(param)

# simulation time step
dt = 0.05

# commands
beta_cmd = 8 * np.pi

# prepare simulation
max_sim_time = 20
sim_time = 0
sim_time_vec = [sim_time]
state_vec = [copy.copy(model.state)]
start_time = time.time()
omega_cmd = np.zeros(2)
contact_forces = None

if enable_animation or args.gif:
    fig = plt.figure(0)
    ax = fig.add_subplot(111, projection='3d')
    if args.gif:
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        import imageio
        canvas = FigureCanvas(fig)
        images = []

# simulate until system is irrecoverable or max_sim_time reached
while not model.is_irrecoverable(
        contact_forces=contact_forces,
        omega_cmd=omega_cmd) and sim_time < max_sim_time:
    # get control input
    omega_cmd = controller.compute_ctrl_input(model.state, beta_cmd, controller.ANGLE_MODE)

    if enable_animation or args.gif:
        # get visualization
        if enable_contact_forces:
            contact_forces = model.compute_contact_forces(omega_cmd=omega_cmd)

        vis = model.get_visualization(
            contact_forces=contact_forces,
            visualize_contact_forces=enable_contact_forces)

        # plot
        plt.cla()

        ax.plot_wireframe(*vis['lower_ball'], color='b', linewidth=0.5)
        ax.plot_wireframe(*vis['upper_ball'], color='r', linewidth=0.5)
        ax.plot_wireframe(*vis['lever_arm'], color='g')

        if enable_contact_forces:
            ax.quiver(*vis['F1'], color='m')
            ax.quiver(*vis['F12'], color='m')
            ax.quiver(*vis['F23'], color='m')

        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')

        ball_pos = model.state.pos
        range = param.r1 + param.r2
        ax.set_xlim3d(ball_pos[0] - range, ball_pos[0] + range)
        ax.set_ylim3d(ball_pos[1] - range, ball_pos[1] + range)
        ax.set_zlim3d(0, 2 * range)

        if enable_animation:
            plt.show(block=False)
            time_passed = time.time() - start_time
            plt.pause(max(dt - time_passed, 0.001))
            start_time = time.time()

    # simulate one time step
    model.simulate_step(dt, omega_cmd)
    sim_time += dt

    if enable_plot:
        # save states as matrix, sim_time and inputs as lists
        state_vec.append(copy.copy(model.state))
        sim_time_vec.append(sim_time)

    if args.verbose:
        print('sim_time: {0:.3f} s'.format(sim_time))

    if args.gif:
        # convert figure to numpy image:
        # https://stackoverflow.com/questions/21939658/matplotlib-render-into-buffer-access-pixel-data
        fig.canvas.draw()

        buf = fig.canvas.tostring_rgb()
        ncols, nrows = fig.canvas.get_width_height()
        image = np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols, 3)

        images.append(image)

if args.gif:
    imageio.mimsave('doc/img/3d_demo.gif', images, fps=1 / dt, palettesize=64, subrectangles=True)

if enable_plot:
    plt.figure()
    plt.plot(sim_time_vec, [state.psi_x for state in state_vec], label='psi_x')
    plt.plot(sim_time_vec, [state.phi_x for state in state_vec], label='phi_x')
    plt.xlabel('time [s]')
    plt.ylabel('angles [rad]')
    plt.legend()
    plt.title('x angles')

    plt.figure()
    plt.plot(sim_time_vec, [state.psi_y for state in state_vec], label='psi_y')
    plt.plot(sim_time_vec, [state.phi_y for state in state_vec], label='phi_y')
    plt.xlabel('time [s]')
    plt.ylabel('angles [rad]')
    plt.legend()
    plt.title('y angles')

    plt.figure()
    plt.plot(sim_time_vec, [state.psi_y_dot for state in state_vec], label='psi_y_dot')
    plt.plot(sim_time_vec, [state.omega_2[1] for state in state_vec], label='omega_2y')
    plt.plot(sim_time_vec, [state.phi_y_dot for state in state_vec], label='phi_y_dot')
    plt.xlabel('time [s]')
    plt.ylabel('omega [rad]')
    plt.legend()
    plt.title('omega')
    plt.show(block=True)
