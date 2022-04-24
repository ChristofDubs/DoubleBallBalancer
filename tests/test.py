import unittest

import numpy as np

import context

from model_2d.controller import Controller
from model_2d.definitions import *
from model_2d.dynamic_model import ModelParam

from model_2d.controller_3 import Controller3
from model_2d.param import getDefaultParam
from model_2d.dynamics_3 import StateIndex


delta = 1e-6


class TestController(unittest.TestCase):
    param = ModelParam()
    param.l = 1.0
    param.m1 = 1.0
    param.m2 = 1.0
    param.m3 = 1.0
    param.r1 = 3.0
    param.r2 = 2.0
    param.tau = 0.100
    param.theta1 = 1.0
    param.theta2 = 1.0
    param.theta3 = 1.0

    controller = Controller(param)

    K = np.array([0.447213596e+00, 1.03556079e+01, -4.73012271e+01,
                  3.683281576e+00, 6.05877477e-01, -3.53469304e+01])

    def test_gains(self):
        # state gain
        for i in range(STATE_SIZE):
            x0 = np.zeros(STATE_SIZE)
            x0[i] = delta
            u = self.controller.compute_ctrl_input(x0, 0)
            self.assertAlmostEqual(u / delta, -self.K[i])

        # input gain
        x0 = np.zeros(STATE_SIZE)

        u = self.controller.compute_ctrl_input(x0, delta)
        self.assertAlmostEqual(u / delta, self.K[0])


class TestController3(unittest.TestCase):
    controller = Controller3()
    K = np.array([-0.447213596, 26.62771567, -199.45731763, -327.80147739, -
                  1.6832815768, 9.73717718, -133.08516459, -137.62380736])

    def test_gains(self):
        # state gain
        for i in range(StateIndex.NUM_STATES):
            x0 = np.zeros(StateIndex.NUM_STATES)
            x0[i] = delta
            u = self.controller.compute_ctrl_input(x0, 0)
            self.assertAlmostEqual(u / delta, -self.K[i])

        # input gain
        x0 = np.zeros(StateIndex.NUM_STATES)

        u = self.controller.compute_ctrl_input(x0, delta)
        self.assertAlmostEqual(u / delta, self.K[0])


if __name__ == '__main__':
    unittest.main()
