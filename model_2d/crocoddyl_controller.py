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
        # self.u_lb = np.array([-2])
        # self.u_ub = np.array([2])
        # self.has_control_limits = True

    def calc(self, data, x, u=None):
        if u is None: u = self.unone

        if self.model.is_irrecoverable(x=x[:-1] + self.des, omega_cmd=u):
            data.xnext[:] = x * np.nan
            data.cost = np.nan
            data.r = np.ones(self.nr) * np.nan
        else:
            data.xnext[-1] = x[-1] + u
            data.xnext[:-1] = x[:-1] + self.model._x_dot(x[:-1] + self.des, 0, data.xnext[-1] + self.u_des) * self.dt
            data.r[:-1] = self.costWeightsState*(data.xnext[:-1])
            data.r[-1] = self.costWeightsInput[0]*u
            data.cost = .5* sum(data.r**2)
        return data.xnext, data.cost

    def setSetpoint(self, x, u_des, mode):
        self.des = x
        self.u_des = u_des
        if mode == BETA_DOT_IDX:
            self.costWeightsState[0] = 0

    # def calcDiff(self, data, x, u=None):
        # Advance user might implement the derivatives
        # pass


class Controller:
    def __init__(self, param):
        model = ActionModel(param)
        self.pred_model = crocoddyl.ActionModelNumDiff(model, True)
        # self.model.u_lb = np.array([-0.5])
        # self.model.u_ub = np.array([0.5])
        self.terminal_model = ActionModel(param)
        self.terminal_model.costWeightsState = [5 * x for x in self.terminal_model.costWeightsState]
        self.terminal_model = crocoddyl.ActionModelNumDiff(self.terminal_model, True)
        # self.terminal_model.u_lb = self.model.u_lb
        # self.terminal_model.u_ub =   self.model.u_ub

    def compute_ctrl_input(self, x0, r, mode=BETA_IDX):
        des = np.zeros(np.shape(x0))
        des[mode] = r
        u_des = r if mode == BETA_DOT_IDX else 0
        self.pred_model.model.setSetpoint(des, u_des, mode)
        self.terminal_model.model.setSetpoint(des, u_des, mode)

        # data = self.model.createData()

        model = self.pred_model

        T = int(20/0.05)  # number of knots
        problem = crocoddyl.ShootingProblem(np.concatenate([x0-des, np.array([0])-u_des]), [model] * T, self.terminal_model)

        # Creating the DDP solver for this OC problem, defining a logger
        # ddp = crocoddyl.SolverBoxDDP(problem)
        ddp = crocoddyl.SolverDDP(problem)
        ddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])

        # Solving it with the DDP algorithm
        ddp.solve()

        return np.cumsum(ddp.us)+u_des, np.array(ddp.xs)[:,:-1] + des