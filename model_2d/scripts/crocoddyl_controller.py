"""Controller class for controlling 2D Double Ball Balancer
"""
import numpy as np

from .definitions import *
from .dynamic_model import DynamicModel

import crocoddyl


class ActionModel(crocoddyl.ActionModelAbstract):
    def __init__(self, param):
        crocoddyl.ActionModelAbstract.__init__(self, crocoddyl.StateVector(7), 1, 7)  # nu = 1; nr = 7
        self.unone = np.zeros(self.nu)
        self.des = np.zeros(self.nr)
        self.u_des = np.zeros(self.nu)

        self.param = param
        self.costWeightsState = [10, 0, 0, 4, 8, 4]
        self.costWeightsInput = [10]
        self.model = DynamicModel(param)
        self.dt = 0.05

    def calc(self, data, x, u=None):
        if u is None:
            u = self.unone

        if self.model.is_irrecoverable(x=x[:-1], omega_cmd=u):
            data.xnext[:] = x * np.nan
            data.cost = np.nan
            data.r = np.ones(self.nr) * np.nan
        else:
            data.xnext[-1] = x[-1] + u
            data.xnext[:-1] = x[:-1] + self.model._x_dot(x[:-1], 0, data.xnext[-1]) * self.dt
            data.r[:-1] = self.costWeightsState * (data.xnext[:-1] - self.des)
            data.r[-1] = self.costWeightsInput[0] * u
            data.cost = .5 * sum(data.r**2)
        return data.xnext, data.cost

    def setSetpoint(self, x, mode):
        self.des = x
        if mode == BETA_DOT_IDX:
            self.costWeightsState[0] = 0


class Controller:
    def __init__(self, param):
        model = ActionModel(param)
        self.pred_model = crocoddyl.ActionModelNumDiff(model, True)
        self.terminal_model = ActionModel(param)
        self.terminal_model.costWeightsState = [5 * x for x in self.terminal_model.costWeightsState]
        self.terminal_model = crocoddyl.ActionModelNumDiff(self.terminal_model, True)

    def compute_ctrl_input(self, x0, r, mode=BETA_IDX):
        des = np.zeros(np.shape(x0))
        des[mode] = r
        self.pred_model.model.setSetpoint(des, mode)
        self.terminal_model.model.setSetpoint(des, mode)

        model = self.pred_model

        T = int(20 / 0.05)  # number of knots
        problem = crocoddyl.ShootingProblem(np.concatenate([x0, np.array([0])]), [model] * T, self.terminal_model)

        # Creating the DDP solver for this OC problem, defining a logger
        ddp = crocoddyl.SolverDDP(problem)
        ddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])

        # Solving it with the DDP algorithm
        ddp.solve([], [], 5)

        return np.cumsum(ddp.us), np.array(ddp.xs)[:, :-1]
