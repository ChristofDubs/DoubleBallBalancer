"""Script for running pybullet simulation of 3D Double Ball Balancer
"""
import pybullet as p
import time
import pybullet_data
import numpy as np
from numpy import sin, cos
from dataclasses import dataclass

import context

from pyrotation import Quaternion

from model_3d.dynamic_model import ModelParam, ModelState
from model_3d.controller import Controller, projectModelState, ANGLE_MODE, VELOCITY_MODE

ANGLE_MODE = ANGLE_MODE
VELOCITY_MODE = VELOCITY_MODE


class PyBulletSim:
    def __init__(self):
        self.start_time = time.time()

        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        p.setGravity(0, 0, -9.81)
        planeId = p.loadURDF("plane.urdf")
        p.changeDynamics(planeId, -1, rollingFriction=0, spinningFriction=0)

        # load models from URDF
        p.setAdditionalSearchPath("..")
        self.lower_ball_id = p.loadURDF(
            "model_3d/urdf/lower_ball.urdf",
            flags=p.URDF_USE_INERTIA_FROM_FILE)
        self.robot_id = p.loadURDF(
            "model_3d/urdf/robot.urdf",
            flags=p.URDF_USE_INERTIA_FROM_FILE)

        p.changeDynamics(self.lower_ball_id, -1, linearDamping=0, angularDamping=0)
        p.changeDynamics(self.robot_id, -1, linearDamping=0, angularDamping=0)

        # extract joint indices
        if p.getNumJoints(self.robot_id) != 2:
            print("robot does not have 2 joints!")
            exit()

        name_to_joint_idx = {}
        for i in range(p.getNumJoints(self.robot_id)):
            name_to_joint_idx.update({p.getJointInfo(self.robot_id, i)[1]: i})

        self.motor_x_idx = name_to_joint_idx[b'lateral_rotation_axis']
        self.motor_y_idx = name_to_joint_idx[b'primary_rotation_axis']

        # add texture for better visualization
        texUid = p.loadTexture("model_3d/urdf/media/circles.png")

        p.changeVisualShape(self.lower_ball_id, -1, textureUniqueId=texUid)
        p.changeVisualShape(self.robot_id, -1, textureUniqueId=texUid)

        self.cam_yaw_param_id = p.addUserDebugParameter("camera_yaw", -180, 180, 0)
        self.cam_pitch_param_id = p.addUserDebugParameter("camera_pitch", -89.9, 89.9, -30)
        self.cam_dist_param_id = p.addUserDebugParameter("camera_distance", 1, 50, 10)

        self.timestep_param_id = p.addUserDebugParameter("simulation_timestep", 1 / 240, 1 / 60, 1 / 60)
        self.rtf_param_id = p.addUserDebugParameter("realtime_factor", 0.1, 5, 2)

        # load controller
        param = ModelParam()
        param.l = 1.0
        param.r1 = 3.0
        param.r2 = 2.0

        self.controller = Controller(param)

        self.respawn()

    def respawn(self):
        lowerBallSpawnPos = [0, 0, self.controller.ctrl_2d.param.r1]
        robotSpawnPos = [0, 0, 2 * self.controller.ctrl_2d.param.r1 + self.controller.ctrl_2d.param.r2]
        defaultOrientation = p.getQuaternionFromEuler([0, 0, 0])
        p.resetBasePositionAndOrientation(self.lower_ball_id, lowerBallSpawnPos, defaultOrientation)
        p.resetBasePositionAndOrientation(self.robot_id, robotSpawnPos, defaultOrientation)
        p.resetJointState(self.robot_id, self.motor_x_idx, 0)
        p.resetJointState(self.robot_id, self.motor_y_idx, 0)

    def simulate_step(self, forward_cmd, forward_cmd_mode, steering_cmd):
        state = self.get_state()
        omega_cmd = list(
            self.controller.compute_ctrl_input(state, forward_cmd, forward_cmd_mode, steering_cmd))
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=[
                self.motor_x_idx,
                self.motor_y_idx],
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=omega_cmd,
            velocityGains=[0.1, 0.1])

        time_step = p.readUserDebugParameter(self.timestep_param_id)
        p.setTimeStep(time_step)

        p.stepSimulation()

        upper_ball_fwd_dir = 90 + projectModelState(state)[2][0] * 180 / np.pi
        p.resetDebugVisualizerCamera(
            cameraDistance=p.readUserDebugParameter(self.cam_dist_param_id),
            cameraYaw=upper_ball_fwd_dir + p.readUserDebugParameter(self.cam_yaw_param_id),
            cameraPitch=p.readUserDebugParameter(self.cam_pitch_param_id),
            cameraTargetPosition=[state.pos[0], state.pos[1], 5])

        realtime_factor = p.readUserDebugParameter(self.rtf_param_id)
        time_passed = time.time() - self.start_time
        sleep_time = time_step / realtime_factor - time_passed
        time.sleep(max(sleep_time, 0))
        self.start_time += time_passed
        return realtime_factor if sleep_time > 0 else time_step / time_passed

    def terminate(self):
        p.disconnect()

    def adjust_quaternion_convention(self, q_xyzw):
        return [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]

    def get_state(self):
        state = ModelState()

        contact = len(p.getContactPoints(bodyA=self.robot_id, bodyB=self.lower_ball_id)) > 0
        if not contact:
            return state

        upper_ball_pos, upper_ball_orientation = p.getBasePositionAndOrientation(self.robot_id)
        lower_ball_pos, lower_ball_orientation = p.getBasePositionAndOrientation(self.lower_ball_id)

        upper_ball_linear_vel, upper_ball_angular_vel = p.getBaseVelocity(self.robot_id)
        lower_ball_linear_vel, lower_ball_angular_vel = p.getBaseVelocity(self.lower_ball_id)

        state.pos = np.array(lower_ball_pos[0:2])
        state.omega_1_z = lower_ball_angular_vel[2]

        state.q1 = np.array(self.adjust_quaternion_convention(lower_ball_orientation))
        state.q2 = np.array(self.adjust_quaternion_convention(upper_ball_orientation))

        R_IB2 = Quaternion(state.q2).rotation_matrix()
        state.omega_2 = np.dot(R_IB2.T, np.array(upper_ball_angular_vel))

        state.phi_x, state.phi_x_dot, _, _ = p.getJointState(
            bodyUniqueId=self.robot_id, jointIndex=self.motor_x_idx)
        state.phi_y, state.phi_y_dot, _, _ = p.getJointState(
            bodyUniqueId=self.robot_id, jointIndex=self.motor_y_idx)

        # e_s1S2 = np.array([sin(psi_y) * cos(psi_x), -sin(psi_x), cos(psi_x) * cos(psi_y)])
        r_S1S2 = np.array(upper_ball_pos) - np.array(lower_ball_pos)
        d = np.linalg.norm(r_S1S2)
        e_S1S2 = r_S1S2 / d

        state.psi_x = -np.arcsin(e_S1S2[1])
        state.psi_y = np.arctan2(e_S1S2[0], e_S1S2[2])

        # v_e_s1S2 = psi_x_dot * np.array([-sin(psi_y) * sin(psi_x), -cos(psi_x), -sin(psi_x) * cos(psi_y)])
        # + psi_y_dot * np.array([cos(psi_y) * cos(psi_x), 0, -cos(psi_x) * sin(psi_y)])

        v_S1S2 = np.array(upper_ball_linear_vel) - np.array(lower_ball_linear_vel)
        v_e_s1S2 = v_S1S2 / d

        state.psi_x_dot = np.dot(v_e_s1S2, np.array(
            [-sin(state.psi_y) * sin(state.psi_x), -cos(state.psi_x), -sin(state.psi_x) * cos(state.psi_y)]))
        state.psi_y_dot = np.dot(v_e_s1S2, np.array(
            [cos(state.psi_y) * cos(state.psi_x), 0, -cos(state.psi_x) * sin(state.psi_y)]))

        return state


if __name__ == '__main__':
    sim = PyBulletSim()

    sim_time = 40

    for i in range(int(sim_time / p.readUserDebugParameter(sim.timestep_param_id))):
        sim.simulate_step(8 * np.pi, ANGLE_MODE, 0)

    sim.terminate()
