import unittest

import numpy as np

from controller_2d import Controller
from definitions_2d import *

controller = Controller()
delta = 1e-6

K = np.array([0.447213596e+00, 1.03556079e+01, -4.73012271e+01,
              3.683281576e+00, 6.05877477e-01, -3.53469304e+01])


class TestController(unittest.TestCase):
    def test_gains(self):
        # state gain
        for i in range(STATE_SIZE):
            x0 = np.zeros(STATE_SIZE)
            x0[i] = delta
            u = controller.compute_ctrl_input(x0, 0)
            self.assertAlmostEqual(u / delta, -K[i])

        # input gain
        x0 = np.zeros(STATE_SIZE)
        u = controller.compute_ctrl_input(x0, delta)
        self.assertAlmostEqual(u / delta, K[0])


if __name__ == '__main__':
    unittest.main()
