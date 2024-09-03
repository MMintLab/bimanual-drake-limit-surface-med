'''
    Final Draft code for bimanual arm control with camera and ATI gamma feedback.
'''
import rospy
from pydrake.all import (
    MultibodyPlant,
    RotationMatrix,
    RigidTransform
)
import numpy as np

from gamma import GammaManager
from camera import CameraManager
import sys
sys.path.append('..')
from load.sim_setup import load_iiwa_setup
from movement_lib import goto_joints, curr_joints, curr_desired_joints, close_arms, direct_joint_torque


class BimanualKuka:
    def __init__(self, scenario_file="../../config/bimanual_med_hardware_gamma.yaml", directives_file="../../config/bimanual_med_gamma.yaml", grasp_force = 10.0):
        # object pose SE2 in thanos/medusa end-effector frame
        self.camera_manager = CameraManager()
        
        # ATI gamma sensing
        self.gamma_manager = GammaManager()
        
        self.scenario_file = scenario_file
        self.directives_file = directives_file
        
        self._plant = MultibodyPlant(1e-3)
        load_iiwa_setup(self._plant, package_file='../../package.xml', directive_path=self.directives_file)
        self._plant.Finalize()
        self._plant_context = self._plant.CreateDefaultContext()
        
        self.grasp_force = grasp_force # grasp force to apply in +z direction (in end-effector frame)
        self.objfeedforward_force = 10.0 # feedforward force to apply in +z direction (in end-effector frame)
        
    def get_poses(self, q):
        self._plant.SetPositions(self._plant_context, self._plant.GetModelInstanceByName("iiwa_thanos"), q[:7])
        self._plant.SetPositions(self._plant_context, self._plant.GetModelInstanceByName("iiwa_medusa"), q[7:])
        thanos_pose = self._plant.GetFrameByName("thanos_finger").CalcPoseInWorld(self._plant_context)
        medusa_pose = self._plant.GetFrameByName("medusa_finger").CalcPoseInWorld(self._plant_context)
        return thanos_pose, medusa_pose
    
    # move arms back to complete zero
    def reset_arms(self):
        thanos_q = np.zeros(7)
        medusa_q = np.zeros(7)
        goto_joints(thanos_q, medusa_q, endtime=30.0, scenario_file=self.scenario_file, directives_file=self.directives_file)
        
    # move arms towards setup position
    def go_home(self):
        curr_q = curr_joints(scenario_file=self.scenario_file)
        home_q = [1.3507202126267908, 1.0306422492340395, 1.4623797469261657, 0.47340482371611614, -1.7470723329576021, 2.0943951023905236, 0.7999247983754085,
                  -1.4783288067442353, 1.644651108471463, 0.9608959783544273, -0.9229792723727335, -1.2343844303404459, -2.0943951023776375, 0.6230349606485529]
        home_q = np.array(home_q)
        
        joint_speed = 5.0 * np.pi / 180.0
        medusa_endtime = np.max(np.abs(home_q[:7] - curr_q[:7])) / joint_speed
        thanos_endtime = np.max(np.abs(home_q[7:] - curr_q[7:])) / joint_speed
        print("medusa_endtime: ", medusa_endtime)
        print("thanos_endtime: ", thanos_endtime)
        
        input("Press Enter to setup medusa arm.")
        goto_joints(curr_q[:7], home_q[7:], endtime=medusa_endtime, scenario_file=self.scenario_file, directives_file=self.directives_file)
        input("Press Enter to setup thanos arm.")
        goto_joints(home_q[:7], home_q[7:], endtime=thanos_endtime, scenario_file=self.scenario_file, directives_file=self.directives_file)
    def close_gripper(self, gap = 0.001):
        curr_des_q = curr_desired_joints(scenario_file=self.scenario_file)
        input("Press Enter to close gripper")
        des_q = close_arms(self._plant, self._plant_context, curr_des_q, gap=gap)
        input("Press Enter to grasp object")
        curr_q = curr_joints(scenario_file=self.scenario_file)
        thanos_pose, medusa_pose = self.get_poses(curr_q)
        
        wrench_thanos = thanos_pose.rotation().matrix() @ np.array([0, 0.0, self.grasp_force])
        wrench_medusa = medusa_pose.rotation().matrix() @ np.array([0, 0.0, self.grasp_force + self.objfeedforward_force])
        
        wrench_thanos = np.concatenate([np.zeros(3), wrench_thanos])
        wrench_medusa = np.concatenate([np.zeros(3), wrench_medusa])
        direct_joint_torque(des_q[:7], des_q[7:], wrench_thanos, wrench_medusa, endtime=5.0, scenario_file=self.scenario_file, directives_file=self.directives_file)
    
    def setup_robot(self):
        self.go_home()
        self.close_gripper()
        
if __name__ == "__main__":
    rospy.init_node("bimanual_kuka")
    bimanual_kuka = BimanualKuka()
    bimanual_kuka.setup_robot()
    rospy.spin()