"""Controller class for controlling 3D Double Ball Balancer
"""
import numpy as np
from numpy import sin, cos
from pyrotation import Quaternion

from .dynamic_model import DynamicModel, ModelState, OMEGA_2_Y_IDX, PHI_Y_DOT_IDX
from model_2d.definitions import BETA_IDX, PHI_IDX, PSI_IDX, BETA_DOT_IDX, PHI_DOT_IDX, PSI_DOT_IDX

from .controller import Controller as LinearController

import crocoddyl


class ActionModel(crocoddyl.ActionModelAbstract):
    def __init__(self, param):
        crocoddyl.ActionModelAbstract.__init__(self, crocoddyl.StateVector(22+2), 1, 13)  # nu = 2; nr = 14
        self.unone = np.zeros(self.nu)
        self.des = ModelState().x
        self.r = 0
        self.mode = 0

        self.param = param
        self.costWeightsState = [0, 10, 0, 4, 8, 4]
        self.costWeightsInput = [10]
        self.model = DynamicModel(param)
        self.dt = 0.05
        self.linear_controller = LinearController(param)

    def calc(self, data, x, u=None):
        if u is None: u = self.unone

        if self.model.is_irrecoverable(state=ModelState(x[:-2] + self.des, skip_checks=True), omega_cmd=[u,0]):
            data.xnext[:] = x * np.nan
            data.cost = np.nan
            data.r = np.ones(self.nr) * np.nan
        else:
            data.xnext[-2] = x[-2] + u
            uy = self.linear_controller.compute_ctrl_input(ModelState(x[:-2] + self.des), self.r, self.mode)[1]
            data.xnext[-1] = uy
            state = ModelState(x[:-2] + self.des + self.model._x_dot(x[:-2] + self.des, 0, data.xnext[-2:]) * self.dt, skip_checks=True)
            state.normalize_quaternions()
            data.xnext[:-2] = state.x - self.des

            [phi_x, phi_y] = state.phi
            [phi_x_dot, phi_y_dot] = state.phi_dot
            [psi_x, psi_y] = state.psi
            [psi_x_dot, psi_y_dot] = state.psi_dot
            [w_2x, w_2y, w_2z] = state.omega_2

            R_IB2 = Quaternion(state.q2).rotation_matrix()

            # construct horizontal ball frame
            z = np.array([0, 0, 1])
            x = np.cross(R_IB2[:, 1], z)
            x *= 1 / np.linalg.norm(x)
            y = np.cross(z, x)

            R_IB2h = np.column_stack([x, y, z])

            # express psi vector in B2h frame
            I_e_S1S2 = np.array([cos(psi_x) * sin(psi_y), -sin(psi_x), cos(psi_x) * cos(psi_y)])
            B2h_e_S1S2 = np.dot(R_IB2h.T, I_e_S1S2)

            # extract psi angles wrt B2h frame
            B2h_psi_x = np.arcsin(-B2h_e_S1S2[1])
            B2h_psi_y = np.arcsin(B2h_e_S1S2[0])

            # express angular velocities induced by psi rates in B2h frame
            I_omega_psi = np.array([psi_x_dot * cos(psi_y), psi_y_dot, -psi_x_dot * sin(psi_y)])
            B2h_omega_psi = np.dot(R_IB2h.T, I_omega_psi)

            # extract psi rates wrt B2h frame
            B2h_psi_x_dot = B2h_omega_psi[0]
            B2h_psi_y_dot = B2h_omega_psi[1]

            # express upper ball velocity in B2h frame
            B2h_omega_IB2 = np.dot(R_IB2h.T, np.dot(R_IB2, state.omega_2))

            # extract upper ball velocity wrt B2h frame
            B2h_omega_2x = B2h_omega_IB2[0]
            B2h_omega_2y = B2h_omega_IB2[1]

            # express lever arm directional vector in B2h frame
            R_IB3 = state.q3.rotation_matrix()
            I_e_S2S3 = np.dot(R_IB3, np.array([0, 0, -1]))
            B2h_e_S2S3 = np.dot(R_IB2h.T, I_e_S2S3)

            # extract lever arm angles wrt B2h frame
            B2h_phi_x = np.arcsin(B2h_e_S2S3[1])
            B2h_phi_y = np.arcsin(-B2h_e_S2S3[0])

            # express lever arm angular velocity in B2h frame
            B3_omega_IB3 = np.array([phi_x_dot +
                                    w_2x *
                                    cos(phi_y) -
                                    w_2z *
                                    sin(phi_y), phi_y_dot *
                                    cos(phi_x) +
                                    w_2x *
                                    sin(phi_x) *
                                    sin(phi_y) +
                                    w_2y *
                                    cos(phi_x) +
                                    w_2z *
                                    sin(phi_x) *
                                    cos(phi_y), -
                                    phi_y_dot *
                                    sin(phi_x) +
                                    w_2x *
                                    sin(phi_y) *
                                    cos(phi_x) -
                                    w_2y *
                                    sin(phi_x) +
                                    w_2z *
                                    cos(phi_x) *
                                    cos(phi_y)])

            B2h_omega_IB3 = np.dot(R_IB2h.T, np.dot(R_IB3, B3_omega_IB3))

            # extract lever arm angular velocity wrt B2h frame
            B2h_phi_x_dot = B2h_omega_IB3[0]
            B2h_phi_y_dot = B2h_omega_IB3[1]

            x = np.zeros(6)

            # rotate along principal motor axis (y-axis)
            x[BETA_IDX] = B2h_phi_y - phi_y
            x[PHI_IDX] = B2h_phi_y
            x[PSI_IDX] = B2h_psi_y

            x[BETA_DOT_IDX] = B2h_omega_2y - self.des[OMEGA_2_Y_IDX]
            x[PHI_DOT_IDX] = B2h_phi_y_dot
            x[PSI_DOT_IDX] = B2h_psi_y_dot

            data.r[:6] = self.costWeightsState*x

            # stabilize lateral axis
            x[BETA_IDX] = B2h_phi_x - phi_x
            x[PHI_IDX] = B2h_phi_x
            x[PSI_IDX] = B2h_psi_x

            x[BETA_DOT_IDX] = B2h_omega_2x
            x[PHI_DOT_IDX] = B2h_phi_x_dot
            x[PSI_DOT_IDX] = B2h_psi_x_dot

            data.r[6:-self.nu] = self.costWeightsState*x

            data.r[-self.nu:] = self.costWeightsInput*u
            data.cost = .5* sum(data.r**2)
        return data.xnext, data.cost

    def setSetpoint(self, des, r, mode):
        self.des = des
        self.r = r
        self.mode = mode
        # if mode == BETA_DOT_IDX:
            # self.costWeightsState[0] = 0

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
        self.controller = LinearController(param)

    def compute_ctrl_input(self, x0, r, mode):
        des_state = ModelState()

        des = des_state.x

        if mode == self.controller.VELCITY_MODE:
            des[OMEGA_2_Y_IDX] = r
            des[PHI_Y_DOT_IDX] = -r


        elif mode == self.controller.ANGLE_MODE:
            pass

        x0[OMEGA_2_Y_IDX] = r
        x0[PHI_Y_DOT_IDX] = -r

        x0[10] = 0.01

        # des[mode] = r
        self.pred_model.model.setSetpoint(des, r, mode)
        self.terminal_model.model.setSetpoint(des, r, mode)

        model = self.pred_model

        T = int(40/0.05)  # number of knots
        problem = crocoddyl.ShootingProblem(np.concatenate([x0-des, np.array([0,-r])]), [model] * T, self.terminal_model)

        xs = []
        us = []
        xs.append(np.concatenate([x0-des, np.array([0,-r])]))  

        for _ in range(T):
            state = ModelState(xs[-1][:-2]+des)
            u = self.controller.compute_ctrl_input(state, r, mode)

            next_state = ModelState(state.x + self.pred_model.model.model._x_dot(state.x, 0, u) * self.pred_model.model.dt, skip_checks=True)
            next_state.normalize_quaternions()

            us.append(np.array([(u - xs[-1][-2:])[0]]))
            xs.append(np.concatenate([next_state.x - des, u]))

        # Creating the DDP solver for this OC problem, defining a logger
        ddp = crocoddyl.SolverDDP(problem)
        ddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])

        # Solving it with the DDP algorithm
        ddp.solve(xs,us,3)

        return np.cumsum(ddp.us, axis=0), [ModelState(x + des, True) for x in np.array(ddp.xs)[:,:-2]]