import unittest

import numpy as np

from rotation import Quaternion, quat_from_angle_axis, quat_from_angle_vector, quat_from_roll_pitch_yaw, rot_from_angle_axis, rot_from_angle_vector, rot_x, rot_y, rot_z


class TestRotation(unittest.TestCase):
    def test_rot_from_angle_vector(self):
        angle_vec = np.array([-1, 0.5, 0.4])

        rot = rot_from_angle_vector(angle_vec)

        # test rotation axis: should be unaffected by rotation
        axis_rot = np.dot(rot, angle_vec)
        self.assertAlmostEqual(np.linalg.norm(axis_rot - angle_vec), 0)

        # test axis perpendicular to rotation axis: should be rotated by angle
        vec = np.cross(angle_vec, np.array([1, 0, 0]))

        vec_rot = np.dot(rot, vec)

        # rotation angle
        rot_angle = np.arccos(np.dot(vec, vec_rot) /
                              (np.linalg.norm(vec) * np.linalg.norm(vec_rot)))

        # reconstructed rotation axis
        axis_rot = np.cross(vec, vec_rot)

        # reconstructed angle vector
        angle_vec_2 = rot_angle * axis_rot / np.linalg.norm(axis_rot)

        # check angle vector
        self.assertAlmostEqual(np.linalg.norm(angle_vec_2 - angle_vec), 0)

    def test_rot_from_angle_axis(self):
        angle_vec = np.array([-1.6, 0.35, -0.7])

        rot1 = rot_from_angle_vector(angle_vec)
        rot2 = rot_from_angle_axis(np.linalg.norm(angle_vec), angle_vec * 4.89485)

        self.check_rot_equal(rot1, rot2)

    def test_rot_x(self):
        angle = 2.1921
        rot1 = rot_x(angle)
        rot2 = rot_from_angle_axis(angle, np.array([1, 0, 0]))

        self.check_rot_equal(rot1, rot2)

    def test_rot_y(self):
        angle = 2.1921
        rot1 = rot_y(angle)
        rot2 = rot_from_angle_axis(angle, np.array([0, 1, 0]))

        self.check_rot_equal(rot1, rot2)

    def test_rot_z(self):
        angle = 2.1921
        rot1 = rot_z(angle)
        rot2 = rot_from_angle_axis(angle, np.array([0, 0, 1]))

        self.check_rot_equal(rot1, rot2)

    def test_quat_from_angle_axis(self):
        axis = np.array([-1, 5, 3.4])
        angle = 2.4832

        q = quat_from_angle_axis(angle, axis)

        axis *= 1 / np.linalg.norm(axis)

        self.assertAlmostEqual(q.w, np.cos(0.5 * angle))
        self.assertAlmostEqual(q.x, axis[0] * np.sin(0.5 * angle))
        self.assertAlmostEqual(q.y, axis[1] * np.sin(0.5 * angle))
        self.assertAlmostEqual(q.z, axis[2] * np.sin(0.5 * angle))

    def quat_from_angle_vector(self):
        omega = np.array([-1, 5, 3.4])

        q1 = quat_from_angle_vector(omega)
        q2 = quat_from_angle_axis(np.linalg.norm(omega), omega * 1.95256)

        self.check_quat_equal(q1, q2)

    def test_inverse(self):
        axis = np.array([-1, 5, 3.4])
        angle = 2.4832

        q1 = quat_from_angle_axis(-angle, axis)
        q2 = quat_from_angle_axis(angle, axis).inverse()

        self.check_quat_equal(q1, q2)

    def test_multiplication_absolute(self):
        q1 = Quaternion(np.array([0.46093515, 0.19205631, -0.80663652, 0.31621304]))
        q2 = Quaternion(np.array([0.26935601, -0.87047979, 0.30220431, 0.27997291]))

        q3 = q1 * q2

        # expected result: -0.724576 - 0.660298 i + 0.141607 j + 0.137635 k
        # source:
        # https://www.wolframalpha.com/input/?i=Quaternion(+0.46093515,+0.19205631,++-0.80663652,++0.31621304)+*+Quaternion(0.26935601,+-0.87047979,++0.30220431,++0.27997291)
        q4 = Quaternion(np.array([0.446575, - 0.670901, - 0.407003, -0.429897]))

        self.check_quat_equal(q3, q4, 6)

    def test_multiplication_simple(self):
        axis = np.array([-1, 5, 3.4])
        angle1 = 2.4832
        angle2 = 0.456
        angle3 = angle1 + angle2

        q1 = quat_from_angle_axis(angle1, axis)
        q2 = quat_from_angle_axis(angle2, axis)
        q3 = quat_from_angle_axis(angle3, axis)

        q4 = q1 * q2

        self.check_quat_equal(q3, q4)

    def test_left_right_multiplication(self):
        q1 = Quaternion(np.array([0.46093515, 0.19205631, -0.80663652, 0.31621304]))
        q2 = Quaternion(np.array([0.26935601, -0.87047979, 0.30220431, 0.27997291]))

        q3 = q1 * q2

        q4 = Quaternion(np.dot(q1.get_left_mult_matrix(), q2.q))
        self.check_quat_equal(q3, q4)

        q5 = Quaternion(np.dot(q2.get_right_mult_matrix(), q1.q))
        self.check_quat_equal(q3, q5)

    def test_q_dot(self):
        q_IB = Quaternion(np.array([0.46093515, 0.19205631, -0.80663652, 0.31621304]))
        B_omega_IB = np.array([1, 2, 3])
        dt = 1e-9
        q_BBnew = quat_from_angle_vector(B_omega_IB * dt)

        q_IB_new = q_IB * q_BBnew

        q_dot = (q_IB_new.q - q_IB.q) / dt

        q_dot_ref = q_IB.q_dot(B_omega_IB, frame='body')

        self.assertAlmostEqual(q_dot_ref[0], q_dot[0], 5)
        self.assertAlmostEqual(q_dot_ref[1], q_dot[1], 5)
        self.assertAlmostEqual(q_dot_ref[2], q_dot[2], 5)
        self.assertAlmostEqual(q_dot_ref[3], q_dot[3], 5)

        I_omega_IB = np.dot(q_IB.rotation_matrix(), B_omega_IB)
        q_dot_ref = q_IB.q_dot(I_omega_IB, frame='inertial')

        self.assertAlmostEqual(q_dot_ref[0], q_dot[0], 5)
        self.assertAlmostEqual(q_dot_ref[1], q_dot[1], 5)
        self.assertAlmostEqual(q_dot_ref[2], q_dot[2], 5)
        self.assertAlmostEqual(q_dot_ref[3], q_dot[3], 5)

    def test_roll_pitch_yaw(self):
        roll = 1.8526989
        pitch = -0.75699
        yaw = -2.89421

        q1 = quat_from_angle_vector(np.array([0, 0, yaw])) * quat_from_angle_vector(
            np.array([0, pitch, 0])) * quat_from_angle_vector(np.array([roll, 0, 0]))
        q2 = quat_from_roll_pitch_yaw(roll, pitch, yaw)

        self.check_quat_equal(q1, q2)

        rpy = q1.get_roll_pitch_yaw()

        self.assertAlmostEqual(roll, rpy[0])
        self.assertAlmostEqual(pitch, rpy[1])
        self.assertAlmostEqual(yaw, rpy[2])

    # helper functions (all function that don't start with "test" are not executed as tests)
    def check_quat_equal(self, q1, q2, n_digits=7):
        # since negating all entries of a quaternion leads to an identical
        # quaternion (angle += 2pi), we have to check first whether they point to
        # the same direction or not
        sgn = np.sign(np.dot(q1.q, q2.q))

        # check valid sign
        self.assertEqual(np.abs(sgn), 1)

        self.assertAlmostEqual(q1.w, sgn * q2.w, n_digits)
        self.assertAlmostEqual(q1.x, sgn * q2.x, n_digits)
        self.assertAlmostEqual(q1.y, sgn * q2.y, n_digits)
        self.assertAlmostEqual(q1.z, sgn * q2.z, n_digits)

    def check_rot_equal(self, rot1, rot2):
        # calculate difference in rotation (should be identity if rot1==rot2)
        eye = np.dot(rot1, rot2.T)
        # trace = 1 + 2*cos(angle) -> trace should be 3 for 0 angle (and det 1)
        self.assertAlmostEqual(np.trace(eye), 3)
        self.assertAlmostEqual(np.linalg.det(eye), 1)


if __name__ == '__main__':
    unittest.main()
