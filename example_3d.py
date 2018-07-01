"""Simple test script for 3D Double Ball Balancer
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
import copy

from dynamic_model_3d import ModelParam, DynamicModel, ModelState
from controller_3d import Controller

print_sim_time = False
plot_visualization = True
plot_states = True
create_gif = False

# create parameter struct
param = ModelParam()
param.l = 1
param.r1 = 3
param.r2 = 2

# initial state
x0 = ModelState()
# x0.q2 = np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])

# instantiate model
model = DynamicModel(param, x0)
controller = Controller()

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

if plot_visualization or create_gif:
    fig = plt.figure(0)
    ax = fig.add_subplot(111, projection='3d')
    if create_gif:
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        import imageio
        canvas = FigureCanvas(fig)
        images = []

# simulate until system is irrecoverable or max_sim_time reached
while not model.is_irrecoverable() and sim_time < max_sim_time:
    if plot_visualization or create_gif:
        # get visualization
        vis = model.get_visualization()

        # plot
        plt.cla()

        ax.plot_wireframe(*vis['lower_ball'], color='b', linewidth=0.5)
        ax.plot_wireframe(*vis['upper_ball'], color='r', linewidth=0.5)
        ax.plot_wireframe(*vis['lever_arm'], color='g')

        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')

        ball_pos = model.state.pos
        range = param.r1 + param.r2
        ax.set_xlim3d(ball_pos[0] - range, ball_pos[0] + range)
        ax.set_ylim3d(ball_pos[1] - range, ball_pos[1] + range)
        ax.set_zlim3d(0, 2 * range)

        if plot_visualization:
            plt.show(block=False)
            time_passed = time.time() - start_time
            plt.pause(max(dt - time_passed, 0.001))
            start_time = time.time()

    # simulate one time step
    model.simulate_step(dt, controller.compute_ctrl_input(model.state, beta_cmd))
    sim_time += dt

    if plot_states:
        # save states as matrix, sim_time and inputs as lists
        state_vec.append(copy.copy(model.state))
        sim_time_vec.append(sim_time)

    if print_sim_time:
        print('sim_time: {0:.3f} s'.format(sim_time))

    if create_gif:
        # convert figure to numpy image:
        # https://stackoverflow.com/questions/35355930/matplotlib-figure-to-image-as-a-numpy-array
        fig.canvas.draw()
        image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        images.append(image)

if create_gif:
    imageio.mimsave('doc/img/3d_demo.gif', images)

if plot_states:
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
