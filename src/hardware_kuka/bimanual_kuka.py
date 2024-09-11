'''
    Final Draft code for bimanual arm control with camera and ATI gamma feedback.
'''
import rospy
from pydrake.all import (
    MultibodyPlant,
    RotationMatrix,
    RigidTransform,
    JacobianWrtVariable
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
    curr_torque_commanded,
    curr_desired_joints,
    close_arms,
    direct_joint_torque,
    inhand_rotate_traj,
    generate_trajectory,
    follow_trajectory_apply_push,
    inhand_se2_traj,
    follow_traj_and_torque_gamma,
    goto_joints_torque,
    follow_traj_and_torque_gamma_se2
)
from scipy.linalg import block_diag

class BimanualKuka:
    def __init__(self, scenario_file="../../config/bimanual_med_hardware_gamma.yaml", directives_file="../../config/bimanual_med_gamma.yaml", grasp_force = 10.0):
        # object pose SE2 in thanos/medusa end-effector frame
        self.camera_manager = CameraManager()
        
        # ATI gamma sensing
        self.gamma_manager = GammaManager(use_compensation=True)
        
        self.scenario_file = scenario_file
        self.directives_file = directives_file
        
        self._plant = MultibodyPlant(1e-3)
        load_iiwa_setup(self._plant, package_file='../../package.xml', directive_path=self.directives_file)
        self._plant.Finalize()
        self._plant_context = self._plant.CreateDefaultContext()
        
        self.grasp_force = grasp_force # grasp force to apply in +z direction (in end-effector frame)
        self.objfeedforward_force = 5.0 # feedforward force to apply in +z direction (in end-effector frame)
        
        self.home_q = [1.0540217588325416, 0.7809724707093576, 0.09313438428576386, -0.5122596993658023, -0.06815758341558487, 1.849912055108764, 1.1014584994181864, -0.633812090276105, 2.049101710112114, 1.2905853609868987, 0.9356053209918564, -1.0266903347613374, -1.6463252209521346, 1.521753948710382]
        self.home_q = np.array(self.home_q)
        
        self.des_q = curr_desired_joints(scenario_file=self.scenario_file)
        
        self.thanos_stiffness = np.ones(7) * 1000.0
        self.medusa_stiffness = np.ones(7) * 1000.0
    
    def get_torque_commanded(self):
        curr_torque = curr_torque_commanded(scenario_file=self.scenario_file)
        return curr_torque
    
    def get_gamma_wrench(self):
        curr_q = curr_joints(scenario_file=self.scenario_file)
        thanos_pose, medusa_pose = self.get_poses(curr_q)
        
        wrench_thanos = self.gamma_manager.get_thanos_wrench(thanos2world_rot=thanos_pose.rotation().matrix())
        wrench_medusa = self.gamma_manager.get_medusa_wrench(medusa2world_rot=medusa_pose.rotation().matrix())
        
        return wrench_thanos, wrench_medusa

    def get_gamma_joint_torque(self):
        wrench_thanos, wrench_medusa = self.get_gamma_wrench()
        
        #zero everything but force xy
        wrench_thanos[5] = 0
        wrench_medusa[5] = 0
        
        # get wrench in world frame
        curr_q = curr_joints(scenario_file=self.scenario_file)
        thanos_pose, medusa_pose = self.get_poses(curr_q)
        
        thanos_rot = block_diag(thanos_pose.rotation().matrix(), thanos_pose.rotation().matrix())
        medusa_rot = block_diag(medusa_pose.rotation().matrix(), medusa_pose.rotation().matrix())
        
        wrench_thanos = thanos_rot @ wrench_thanos
        wrench_medusa = medusa_rot @ wrench_medusa
        
        #get jacobians
        J_thanos = self._plant.CalcJacobianSpatialVelocity(self._plant_context, JacobianWrtVariable.kQDot, 
                                                           self._plant.GetFrameByName("iiwa_link_ee_kuka", self._plant.GetModelInstanceByName("iiwa_thanos")),
                                                           [0,0,0],
                                                           self._plant.world_frame(),
                                                           self._plant.world_frame())[:, :7]
        J_medusa = self._plant.CalcJacobianSpatialVelocity(self._plant_context, JacobianWrtVariable.kQDot,
                                                           self._plant.GetFrameByName("iiwa_link_ee_kuka", self._plant.GetModelInstanceByName("iiwa_medusa")),
                                                           [0,0,0],
                                                           self._plant.world_frame(),
                                                           self._plant.world_frame())[:, 7:]
        
        thanos_torque = J_thanos.T @ wrench_thanos
        medusa_torque = J_medusa.T @ wrench_medusa
        
        return thanos_torque, medusa_torque
    
    def zero_grasp_external_forces(self):
        thanos_torque, medusa_torque = self.get_gamma_joint_torque()
        
        thanos_delta_q = thanos_torque / self.thanos_stiffness
        medusa_delta_q = medusa_torque / self.medusa_stiffness
        
        delta_q = np.concatenate([thanos_delta_q, medusa_delta_q]) 
        return delta_q

    def goto_joints(self, thanos_q, medusa_q, joint_speed=5.0, endtime = None):
        if endtime is None:
            curr_q = curr_joints(scenario_file=self.scenario_file)
            thanos_endtime = np.max(np.abs(thanos_q - curr_q[:7])) / joint_speed
            medusa_endtime = np.max(np.abs(medusa_q - curr_q[7:])) / joint_speed
            endtime = np.max([thanos_endtime, medusa_endtime])
        goto_joints(thanos_q, medusa_q, endtime=endtime, scenario_file=self.scenario_file, directives_file=self.directives_file)
        
        
    def get_poses(self, q):
        self._plant.SetPositions(self._plant_context, self._plant.GetModelInstanceByName("iiwa_thanos"), q[:7])
        self._plant.SetPositions(self._plant_context, self._plant.GetModelInstanceByName("iiwa_medusa"), q[7:])
        thanos_pose = self._plant.GetFrameByName("thanos_finger").CalcPoseInWorld(self._plant_context)
        medusa_pose = self._plant.GetFrameByName("medusa_finger").CalcPoseInWorld(self._plant_context)
        return thanos_pose, medusa_pose
    
    def get_obj_relative_poses(self):
        current_obj2medusa_se2 = self.camera_manager.get_medusa_se2()
        current_obj2thanos_se2 = self.camera_manager.get_thanos_se2()
        
        return current_obj2thanos_se2, current_obj2medusa_se2
    
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
        self.des_q = self.home_q
        
    def close_gripper(self, gap = 0.01):
        curr_des_q = self.des_q
        
        input("Press Enter to close gripper")
        des_q = close_arms(self._plant, self._plant_context, curr_des_q, gap=gap)
        input("Press Enter to grasp object")
        wrench_thanos = np.concatenate([np.zeros(3), np.array([0, 0.0, self.grasp_force])])
        wrench_medusa = np.concatenate([np.zeros(3), np.array([0, 0.0, self.grasp_force + self.objfeedforward_force])])
        
        direct_joint_torque(des_q[:7], des_q[7:], wrench_thanos, wrench_medusa, endtime=5.0, scenario_file=self.scenario_file, directives_file=self.directives_file)
        
        gamma_wrench_thanos, gamma_wrench_medusa = self.get_gamma_wrench()
        print("gamma_wrench_thanos: ", gamma_wrench_thanos)
        print("gamma_wrench_medusa: ", gamma_wrench_medusa)
        
        # delta_q = self.zero_grasp_external_forces()
        # des_q[:7] += delta_q[:7]
        # des_q[7:] += delta_q[7:]
        
        # input("Press Enter to zero external forces")
        # goto_joints_torque(des_q[:7], des_q[7:], wrench_thanos, wrench_medusa, endtime=5.0, scenario_file=self.scenario_file, directives_file=self.directives_file)
        
        # des_q = curr_joints(scenario_file=self.scenario_file)
        # direct_joint_torque(des_q[:7], des_q[7:], wrench_thanos, wrench_medusa, endtime=5.0, scenario_file=self.scenario_file, directives_file=self.directives_file)
        
        
        self.des_q = des_q
        
    def setup_robot(self):
        self.go_home()
        self.close_gripper(gap=0.01)
        
    def rotate_arms(self, rotation, rotate_steps = 30, rotate_time = 30.0):
        curr_des_q = self.des_q
        thanos_pose, medusa_pose = self.get_poses(curr_des_q)
        current_obj2medusa_se2 = self.camera_manager.get_medusa_se2()
        
        thanos_ee_piecewise, medusa_ee_piecewise, T = inhand_rotate_traj(rotation, rotate_steps, rotate_time, thanos_pose, medusa_pose, current_obj2medusa_se2)
        thanos_piecewise, medusa_piecewise, T = generate_trajectory(self._plant, curr_des_q, thanos_ee_piecewise, medusa_ee_piecewise, T, tsteps=100)
        
        input("Press Enter to rotate arms")
        filter_vector_medusa = np.array([0,0,1,1,1,0])
        filter_vector_thanos = np.array([0,0,1,1,1,0])
        # follow_trajectory_apply_push(thanos_piecewise, medusa_piecewise, force=30.0, camera_manager=self.camera_manager, object_kg = 0.5, endtime = T, scenario_file=self.scenario_file, directives_file=self.directives_file)
        follow_traj_and_torque_gamma(thanos_piecewise, medusa_piecewise, self.camera_manager, self.gamma_manager, force=30.0, object_kg=0.5, endtime = T, scenario_file=self.scenario_file, directives_file=self.directives_file,
                                    filter_vector_medusa=filter_vector_medusa, filter_vector_thanos=filter_vector_thanos)
        des_thanos_q = thanos_piecewise.value(T).flatten()
        des_medusa_q = medusa_piecewise.value(T).flatten()
        des_q = np.concatenate([des_thanos_q, des_medusa_q])
        self.des_q = des_q
        
    def se2_arms(self, desired_obj2arm_se2, medusa = True, se2_time = 10.0, force = 10.0, object_kg = 1.0, filter_vector_medusa = np.array([0,0,1,1,1,0]), filter_vector_thanos = np.array([0,0,1,1,1,0])):
        curr_des_q = curr_desired_joints(scenario_file=self.scenario_file)
        thanos_pose, medusa_pose = self.get_poses(curr_des_q)
        
        
        current_obj2arm_se2 = self.camera_manager.get_medusa_se2() if medusa else self.camera_manager.get_thanos_se2()
        thanos_ee_piecewise, medusa_ee_piecewise, T = inhand_se2_traj(thanos_pose, medusa_pose, current_obj2arm_se2, desired_obj2arm_se2, medusa=medusa, se2_time=se2_time)
        
        thanos_piecewise, medusa_piecewise, T = generate_trajectory(self._plant, curr_des_q, thanos_ee_piecewise, medusa_ee_piecewise, T, tsteps=100)
        
        print("current_obj2arm_se2: ", current_obj2arm_se2)
        input("Press Enter to move arms")
        # self.gamma_manager.zero_sensor()
        # follow_traj_and_torque_gamma(thanos_piecewise, medusa_piecewise, self.camera_manager, self.gamma_manager, force=force, object_kg=object_kg, endtime = T, scenario_file=self.scenario_file, directives_file=self.directives_file,
        #                              filter_vector_medusa=filter_vector_medusa, filter_vector_thanos=filter_vector_thanos)
        follow_traj_and_torque_gamma_se2(thanos_piecewise, medusa_piecewise, self.gamma_manager, force=force, object_kg=object_kg, endtime = T, scenario_file=self.scenario_file, directives_file=self.directives_file,
                                     filter_vector_medusa=filter_vector_medusa, filter_vector_thanos=filter_vector_thanos, medusa=medusa)
        des_thanos_q = thanos_piecewise.value(T).flatten()
        des_medusa_q = medusa_piecewise.value(T).flatten()
        des_q = np.concatenate([des_thanos_q, des_medusa_q])
        self.des_q = des_q
if __name__ == "__main__":
    rospy.init_node("bimanual_kuka")
    bimanual_kuka = BimanualKuka(scenario_file="../../config/bimanual_med_hardware_gamma.yaml", directives_file="../../config/bimanual_med_gamma.yaml")
    rospy.sleep(0.1)
    
    wrench_thanos, wrench_medusa = bimanual_kuka.get_gamma_wrench()
    print("wrench_thanos: ", np.round(wrench_thanos,2))
    print("wrench_medusa: ", np.round(wrench_medusa,2))
    bimanual_kuka.setup_robot()
    
    
    
    
    #print wrench
    wrench_thanos, wrench_medusa = bimanual_kuka.get_gamma_wrench()
    print("wrench_thanos: ", np.round(wrench_thanos,2))
    print("wrench_medusa: ", np.round(wrench_medusa,2))

    
    # # get current pose
    thanos_pose, medusa_pose = bimanual_kuka.get_poses(bimanual_kuka.des_q)

    obj2thanos_se2, obj2medusa_se2 = bimanual_kuka.get_obj_relative_poses()
    
    print("obj2thanos_se2: ", obj2thanos_se2)
    print("obj2medusa_se2: ", obj2medusa_se2)
    
    
    bimanual_kuka.se2_arms(np.array([-0.03,0.00,np.pi]), medusa=False, se2_time=10.0, force = 10.0, object_kg = 0.5, filter_vector_medusa=np.array([1,1,1,1,1,0]), filter_vector_thanos=np.array([1,1,1,1,1,0]))
    
    wrench_thanos, wrench_medusa = bimanual_kuka.get_gamma_wrench()
    print("wrench_thanos: ", np.round(wrench_thanos,2))
    print("wrench_medusa: ", np.round(wrench_medusa,2))
    
    obj2thanos_se21, obj2medusa_se21 = bimanual_kuka.get_obj_relative_poses()
    
    # # get current pose
    thanos_pose1, medusa_pose1 = bimanual_kuka.get_poses(bimanual_kuka.des_q)
    
    # # get translation error
    thanos_error = thanos_pose.translation() - thanos_pose1.translation()
    medusa_error = medusa_pose.translation() - medusa_pose1.translation()
    print("thanos_error: ", thanos_error)
    print("medusa_error: ", medusa_error)
    
    print("obj2thanos_error: ", obj2thanos_se2 - obj2thanos_se21)
    print("obj2medusa_error: ", obj2medusa_se2 - obj2medusa_se21)
    
    
    # real test
    # bimanual_kuka.rotate_arms(30 * np.pi/180)
    # bimanual_kuka.se2_arms(np.array([-0.03,0.00,np.pi]), medusa=False, se2_time=10.0, force = 10.0, object_kg = 0.5, filter_vector_medusa=np.array([1,1,1,1,1,0]), filter_vector_thanos=np.array([1,1,1,1,1,0]))
    # bimanual_kuka.rotate_arms(120 * np.pi/180)
    # bimanual_kuka.se2_arms(np.array([0.0,0.00,np.pi/2]), medusa=True, se2_time=10.0, force = 10.0, object_kg = 0.5, filter_vector_medusa=np.array([0,0,1,1,1,0]), filter_vector_thanos=np.array([1,1,1,1,1,0]))
    # bimanual_kuka.rotate_arms(-150 * np.pi/180)
    
    print("Finished demo")
    rospy.spin()