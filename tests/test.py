import unittest

import context  # noqa: F401
import numpy as np

from model_2d.controller_2 import Controller as Controller2
from model_2d.controller_3 import Controller as Controller3
from model_2d.dynamics_2 import StateIndex as StateIndex2
from model_2d.dynamics_3 import StateIndex as StateIndex3
from model_2d.param import getDefaultParam

delta = 1e-6


class TestController(unittest.TestCase):
    controller = Controller2(getDefaultParam(2))

    K = np.array([0.447213596e+00, 1.03556079e+01, -4.73012271e+01,
                  3.683281576e+00, 6.05877477e-01, -3.53469304e+01])

    def test_gains(self):
        # state gain
        for i in range(StateIndex2.NUM_STATES):
            x0 = np.zeros(StateIndex2.NUM_STATES)
            x0[i] = delta
            u = self.controller.compute_ctrl_input(x0, 0)
            self.assertAlmostEqual(u / delta, -self.K[i])

        # input gain
        x0 = np.zeros(StateIndex2.NUM_STATES)

        u = self.controller.compute_ctrl_input(x0, delta)
        self.assertAlmostEqual(u / delta, self.K[0])


class TestController3(unittest.TestCase):
    controller = Controller3(getDefaultParam(3))
    K = np.array([-0.447213596, 26.62771567, -199.45731763, -327.80147739, -
                  1.6832815768, 9.73717718, -133.08516459, -137.62380736])

    def test_gains(self):
        # state gain
        for i in range(StateIndex3.NUM_STATES):
            x0 = np.zeros(StateIndex3.NUM_STATES)
            x0[i] = delta
            u = self.controller.compute_ctrl_input(x0, 0)
            self.assertAlmostEqual(u / delta, -self.K[i])

        # input gain
        x0 = np.zeros(StateIndex3.NUM_STATES)

        u = self.controller.compute_ctrl_input(x0, delta)
        self.assertAlmostEqual(u / delta, self.K[0])


if __name__ == '__main__':
    unittest.main()
