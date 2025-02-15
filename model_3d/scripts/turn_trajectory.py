"""generate a turning trajectories for 3D Double Ball Balancer
"""
import matplotlib.pyplot as plt
import numpy as np

import context

import copy

import pickle
from model_2d.dynamics_2 import StateIndex as StateIndex2D
from model_3d.dynamic_model import DynamicModel, ModelParam, ModelState
from model_3d.controller import Controller, projectModelState, VELOCITY_MODE


# create parameter struct
param = ModelParam()
param.l = 1.0
param.r1 = 3.0
param.r2 = 2.0

for beta_cmd in [0.1, 0.2, 0.4, 0.7, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]:
    stop_iteration = False
    for scale in range(1, 11):
        if stop_iteration:
            continue

        # initial state
        x0 = ModelState()

        x0.omega_2 = np.array([0, beta_cmd, 0])
        x0.phi_y_dot = -x0.omega_2[1]

        # instantiate model
        model = DynamicModel(param, x0)
        controller = Controller(param)

        # simulation time step
        dt = 0.05

        # commands
        control_mode = VELOCITY_MODE

        # prepare simulation
        max_sim_time = 50 * max(1, 2.0 * np.abs(beta_cmd))
        sim_time = 0
        sim_time_vec = [sim_time]
        state_vec = [copy.copy(model.state)]
        omega_cmd = np.zeros(2)
        contact_forces = None

        omega_x_cmd_offset = 0.05 * scale

        print(f"generating trajectory for {beta_cmd:.3} {omega_x_cmd_offset:.3}")

        # simulate until system is irrecoverable or max_sim_time reached
        while sim_time < max_sim_time:
            if model.is_irrecoverable(contact_forces=contact_forces, omega_cmd=omega_cmd):
                stop_iteration = True
                break

            # get control input
            omega_cmd = controller.compute_ctrl_input(model.state, beta_cmd, control_mode, 0.0)
            omega_cmd[0] += omega_x_cmd_offset * min(1, sim_time / (0.3 * max_sim_time))

            # simulate one time step
            model.simulate_step(dt, omega_cmd)
            sim_time += dt

            # save states as matrix, sim_time and inputs as lists
            state_vec.append(copy.copy(model.state))
            sim_time_vec.append(sim_time)

        stop_iteration = stop_iteration or np.abs(projectModelState(state_vec[-1])[0][StateIndex2D.ALPHA_1_IDX]) > 1

        if not stop_iteration:
            with open(f'data/turn_data_{beta_cmd:.3}_{omega_x_cmd_offset:.3}.pickle', 'wb') as handle:
                pickle.dump([beta_cmd, omega_x_cmd_offset, state_vec], handle, protocol=pickle.HIGHEST_PROTOCOL)

        # plt.figure()
        # plt.plot(sim_time_vec, [state.psi_x for state in state_vec], label='psi_x')
        # plt.plot(sim_time_vec, [state.phi_x for state in state_vec], label='phi_x')
        # plt.xlabel('time [s]')
        # plt.ylabel('angles [rad]')
        # plt.legend()
        # plt.title('x angles')

        # plt.figure()
        # plt.plot(sim_time_vec, [state.psi_y for state in state_vec], label='psi_y')
        # plt.plot(sim_time_vec, [state.phi_y for state in state_vec], label='phi_y')
        # plt.xlabel('time [s]')
        # plt.ylabel('angles [rad]')
        # plt.legend()
        # plt.title('y angles')

        # plt.figure()
        # plt.plot(sim_time_vec, [state.psi_y_dot for state in state_vec], label='psi_y_dot')
        # plt.plot(sim_time_vec, [state.omega_2[1] for state in state_vec], label='omega_2y')
        # plt.plot(sim_time_vec, [projectModelState(state)[2][1] for state in state_vec], label='omega_2z')
        # plt.plot(sim_time_vec, [state.phi_y_dot for state in state_vec], label='phi_y_dot')
        # plt.xlabel('time [s]')
        # plt.ylabel('omega [rad]')
        # plt.legend()
        # plt.title('omega')

        # plt.figure()
        # plt.plot([state.pos[1] for state in state_vec], [state.pos[0] for state in state_vec])
        # plt.xlabel('x [m]')
        # plt.ylabel('y [m]')
        # plt.title('position')
        # plt.show(block=True)
