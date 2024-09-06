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
from movement_lib import (
    goto_joints,
    curr_joints,
    curr_desired_joints,
    close_arms,
    direct_joint_torque,
    inhand_rotate_traj,
    generate_trajectory,
    follow_trajectory_apply_push,
    inhand_se2_traj,
    follow_traj_and_torque_gamma
)


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
        
        self.home_q = [1.0702422097407691, 0.79111135304063, 0.039522481390182704, -0.47337899137126993, -0.029476186840982563, 1.8773559661476429, 1.0891375237383238,
                    -0.6243724965777308, 1.8539706319471008, -1.419344148470764, -0.9229579763233258, 1.7124576303632164, -1.8588769537333005, 1.5895425219089256]
        self.home_q = np.array(self.home_q)
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
        
        joint_speed = 5.0 * np.pi / 180.0
        
        thanos_endtime = np.max(np.abs(self.home_q[:7] - curr_q[:7])) / joint_speed
        medusa_endtime = np.max(np.abs(self.home_q[7:] - curr_q[7:])) / joint_speed
        
        print("medusa_endtime: ", medusa_endtime)
        print("thanos_endtime: ", thanos_endtime)
        
        input("Press Enter to setup medusa arm.")
        goto_joints(curr_q[:7], self.home_q[7:], endtime=medusa_endtime, scenario_file=self.scenario_file, directives_file=self.directives_file)
        input("Press Enter to setup thanos arm.")
        goto_joints(self.home_q[:7], self.home_q[7:], endtime=thanos_endtime, scenario_file=self.scenario_file, directives_file=self.directives_file)
    def close_gripper(self, gap = 0.0005):
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
        self.gamma_manager.zero_sensor()
        
    def rotate_arms(self, rotation, rotate_steps = 30, rotate_time = 30.0):
        curr_des_q = curr_desired_joints(scenario_file=self.scenario_file)
        thanos_pose, medusa_pose = self.get_poses(curr_des_q)
        current_obj2medusa_se2 = self.camera_manager.get_medusa_se2()
        
        thanos_ee_piecewise, medusa_ee_piecewise, T = inhand_rotate_traj(rotation, rotate_steps, rotate_time, thanos_pose, medusa_pose, current_obj2medusa_se2)
        thanos_piecewise, medusa_piecewise, T = generate_trajectory(self._plant, curr_des_q, thanos_ee_piecewise, medusa_ee_piecewise, T, tsteps=100)
        
        input("Press Enter to rotate arms")
        follow_trajectory_apply_push(thanos_piecewise, medusa_piecewise, force=30.0, camera_manager=self.camera_manager, object_kg = 0.5, endtime = T, scenario_file=self.scenario_file, directives_file=self.directives_file)
        
    def se2_arms(self, desired_obj2arm_se2, medusa = True, se2_time = 10.0, force = 10.0, object_kg = 1.0, extra_z_force = 5.0, filter_vector_medusa = np.array([0,0,1,1,1,0]), filter_vector_thanos = np.array([0,0,1,1,1,0])):
        curr_des_q = curr_desired_joints(scenario_file=self.scenario_file)
        thanos_pose, medusa_pose = self.get_poses(curr_des_q)
        
        
        current_obj2arm_se2 = self.camera_manager.get_medusa_se2() if medusa else self.camera_manager.get_thanos_se2()
        thanos_ee_piecewise, medusa_ee_piecewise, T = inhand_se2_traj(thanos_pose, medusa_pose, current_obj2arm_se2, desired_obj2arm_se2, medusa=medusa, se2_time=se2_time)
        
        thanos_piecewise, medusa_piecewise, T = generate_trajectory(self._plant, curr_des_q, thanos_ee_piecewise, medusa_ee_piecewise, T, tsteps=100)
        
        print("current_obj2arm_se2: ", current_obj2arm_se2)
        input("Press Enter to move arms")
        self.gamma_manager.zero_sensor()
        print("Starting to follow trajectory")
        
        follow_traj_and_torque_gamma(thanos_piecewise, medusa_piecewise, self.camera_manager, self.gamma_manager, force=force, object_kg=object_kg, endtime = T, scenario_file=self.scenario_file, directives_file=self.directives_file,
                                     filter_vector_medusa=filter_vector_medusa, filter_vector_thanos=filter_vector_thanos, feedforward_z_force= extra_z_force)
        print("Finished following trajectory")
if __name__ == "__main__":
    rospy.init_node("bimanual_kuka")
    bimanual_kuka = BimanualKuka(scenario_file="../../config/bimanual_med_hardware_gamma.yaml", directives_file="../../config/bimanual_med_gamma.yaml")
    bimanual_kuka.setup_robot()
    bimanual_kuka.rotate_arms(np.pi/6)
    
    bimanual_kuka.se2_arms(np.array([0,0.03,np.pi]), medusa=False, se2_time=30.0, force = 10.0, object_kg = 1.0, extra_z_force=0.0)
    
    bimanual_kuka.rotate_arms(120 * np.pi/180)
    
    bimanual_kuka.se2_arms(np.array([0,-0.03,0.0]), medusa=True, se2_time=30.0, force = 10.0, object_kg = 1.0, extra_z_force=0.0)
    
    # medusa_pose = bimanual_kuka.camera_manager.get_medusa_se2()
    # medusa_pose[2] += np.pi/2
    # bimanual_kuka.se2_arms(medusa_pose, medusa=True, se2_time=30.0, force = 20.0, object_kg = 0.5, filter_vector_medusa = np.ones(6))
    
    # thanos_pose = bimanual_kuka.camera_manager.get_thanos_se2()
    # thanos_pose[2] += np.pi/2
    # bimanual_kuka.se2_arms(thanos_pose, medusa=False, se2_time=30.0, force = 20.0, object_kg = 0.5)
    
    bimanual_kuka.rotate_arms(-150 * np.pi/180 )
    print("Finished demo")
    rospy.spin()