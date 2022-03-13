"""Controller class for controlling 3D Double Ball Balancer


The controller then tries to find control inputs for the lateral controller to optimize the trajectory

"""
import numpy as np
from numpy import sin, cos
from pyrotation import Quaternion

from model_3d.dynamic_model import DynamicModel, ModelState, STATE_SIZE

from model_3d.controller import Controller as LinearController
from model_3d.controller import projectModelState

import crocoddyl


class ActionModel(crocoddyl.ActionModelAbstract):
    """Notes about action model
    
    - To achieve smooth inputs, add the input as a state, and change that state by the external input u.
      That way, changes in input can be penalized (rather than the actual input)

    - This model uses the model_3d.controller as a forward controller,
      and optimizes the input of the lateral control only

    - Since the states used in cost residuals must be 0 at the desired state, the state used internally is offset by
      the desired state (not needed for when laterally stabilizing at motor x angle 0) 

    """
    def __init__(self, param):
        crocoddyl.ActionModelAbstract.__init__(self, crocoddyl.StateVector(STATE_SIZE + 2), 1, 7)  # nu = 1; nr = 7
        self.unone = np.zeros(self.nu)
        self.des = ModelState().x
        self.r = 0
        self.mode = 0

        self.param = param
        self.costWeightsState = np.array([10, 10, 0, 4, 8, 4])
        self.costWeightsInput = [10]
        self.model = DynamicModel(param)
        self.dt = 0.05
        self.linear_controller = LinearController(param)

    def calc(self, data, x, u=None):
        if u is None:
            u = self.unone

        data.xnext[-2] = x[-2] + u
        uy = self.linear_controller.compute_ctrl_input(ModelState(x[:-2] + self.des), self.r, self.mode)[1]
        data.xnext[-1] = uy

        if self.model.is_irrecoverable(state=ModelState(
                x[:-2] + self.des, skip_checks=True), omega_cmd=data.xnext[-2:]):
            data.xnext = x * np.nan
            data.cost = np.nan
            data.r = np.ones(self.nr) * np.nan
        else:
            state = ModelState(x[:-2] + self.des + self.model._x_dot(x[:-2] + self.des,
                                                                     0, data.xnext[-2:]) * self.dt, skip_checks=True)
            state.normalize_quaternions()
            data.xnext[:-2] = state.x - self.des

            data.r[:6] = self.costWeightsState * projectModelState(state)[0]
            data.r[-self.nu:] = self.costWeightsInput * u
            data.cost = .5 * sum(data.r**2)
        return data.xnext, data.cost

    def setSetpoint(self, des, r, mode):
        self.des = des
        self.r = r
        self.mode = mode


class Controller:
    def __init__(self, param):
        self.model = ActionModel(param)
        self.pred_model = crocoddyl.ActionModelNumDiff(self.model, True)
        self.terminal_model = ActionModel(param)
        self.terminal_model.costWeightsState *= 5
        self.terminal_model = crocoddyl.ActionModelNumDiff(self.terminal_model, True)
        self.controller = LinearController(param)

    def compute_ctrl_input(self, x0, r, mode):
        des = ModelState()

        if mode == self.controller.VELCITY_MODE:
            des.omega_2 = np.array([0, r, 0])
            des.phi_y_dot = -r

        des = des.x
        x0 = x0.x

        self.pred_model.model.setSetpoint(des, r, mode)
        self.terminal_model.model.setSetpoint(des, r, mode)

        T = int(20 / 0.05)  # number of knots

        problem = crocoddyl.ShootingProblem(np.concatenate(
            [x0 - des, np.array([0, -r])]), [self.pred_model] * T, self.terminal_model)

        # use linear controller for an initial solution of {x, u}
        xs = []
        us = []
        xs.append(np.concatenate([x0 - des, np.array([0, -r])]))

        for _ in range(T):
            state = ModelState(xs[-1][:-2] + des)
            u = self.controller.compute_ctrl_input(state, r, mode)

            next_state = ModelState(
                state.x +
                self.pred_model.model.model._x_dot(
                    state.x,
                    0,
                    u) *
                self.pred_model.model.dt,
                skip_checks=True)
            next_state.normalize_quaternions()

            us.append(np.array([u[0] - xs[-1][-2]]))
            xs.append(np.concatenate([next_state.x - des, u]))

        # Creating the DDP solver for this OC problem, defining a logger
        ddp = crocoddyl.SolverDDP(problem)
        ddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])

        # Solving it with the DDP algorithm
        ddp.solve(xs, us, 1)

        return np.cumsum(ddp.us, axis=0), [ModelState(x + des, True) for x in np.array(ddp.xs)[:, :-2]]
