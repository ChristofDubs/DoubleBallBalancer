import unittest

import numpy as np

import controller_2d as ctrl
from definitions_2d import *

lqr_controller = ctrl.LQRController()
stacked_controller = ctrl.Controller()
delta = 1e-6


class TestController(unittest.TestCase):
    def test_gains(self):
        # state gain
        for i in range(STATE_SIZE):
            x0 = np.zeros(STATE_SIZE)
            x0[i] = delta
            u1 = lqr_controller.compute_ctrl_input(x0, 0)
            u2 = stacked_controller.compute_ctrl_input(x0, 0)

            k1 = u1 / delta
            k2 = u2 / delta

            self.assertAlmostEqual(k1, k2)

        # input gain
        x0 = np.zeros(STATE_SIZE)
        u1 = lqr_controller.compute_ctrl_input(x0, delta)
        u2 = stacked_controller.compute_ctrl_input(x0, delta)

        k1 = u1 / delta
        k2 = u2 / delta

        self.assertAlmostEqual(k1, k2)


if __name__ == '__main__':
    unittest.main()
