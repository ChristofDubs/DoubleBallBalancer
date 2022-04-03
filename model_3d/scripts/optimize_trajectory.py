"""Find an optimal trajectories for 3D Double Ball Balancer
"""
import matplotlib.pyplot as plt
import numpy as np

import context

import pickle
from model_3d.dynamic_model import ModelParam, ModelState
from crocoddyl_controller import Controller, VELOCITY_MODE


# create parameter struct
param = ModelParam()
param.l = 1.0
param.r1 = 3.0
param.r2 = 2.0

# initial state
x0 = ModelState()

# commands
for beta_cmd in [0.7, 0.8, 1.5, 1.7, 1.8, 2.0, 2.2]:

    x0.omega_2 = np.array([0, beta_cmd, 0])

    x0.phi_y_dot = -beta_cmd
    x0.phi_x = 0.01

    controller = Controller(param)

    # simulation time step
    dt = 0.05

    us, state_vec = controller.compute_ctrl_input(x0, beta_cmd, VELOCITY_MODE)
    sim_time_vec = np.array(range(len(state_vec))) * dt

    with open('data_{}.pickle'.format(beta_cmd), 'wb') as handle:
        pickle.dump([us, state_vec, sim_time_vec], handle, protocol=pickle.HIGHEST_PROTOCOL)


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
