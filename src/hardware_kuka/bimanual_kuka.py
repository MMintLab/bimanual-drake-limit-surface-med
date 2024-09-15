'''
    Final Draft code for bimanual arm control with camera and ATI gamma feedback.
'''
import rospy
from geometry_msgs.msg import Vector3
from pydrake.all import (
    MultibodyPlant,
    RotationMatrix,
    RigidTransform,
    JacobianWrtVariable,
    PiecewisePolynomial
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
from planning.ik_util import solveDualIK
from scipy.linalg import block_diag

class BimanualKuka:
    def __init__(self, scenario_file="../../config/bimanual_med_hardware_gamma.yaml", directives_file="../../config/bimanual_med_gamma.yaml", grasp_force = 35.0, obj_feedforward=5.0):
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
        
        self.home_q = [1.0540217588325416, 0.7809724707093576, 0.09313438428576386, -0.5122596993658023, -0.06815758341558487, 1.849912055108764, 1.1014584994181864, -0.633812090276105, 2.049101710112114, 1.2905853609868987, 0.9356053209918564, -1.0266903347613374, -1.6463252209521346, 1.521753948710382]
        self.angle = 0.0

        # self.home_q = [0.616143429570238, 1.6962545441640784, 1.1110463972964275, -1.013969579064842, 1.7121774471629296, 2.024581932313422, -1.6266687105457462, -0.8812609725550722, 0.9981531684014578, 1.7713755144493566, 0.3836090390795142, 1.1602213959287504, -2.024581932313422, -1.3021454001891604]
        # self.angle = 180.0 * np.pi / 180.0
        
        plant_context = self._plant.CreateDefaultContext()
        self._plant.SetPositions(plant_context, self._plant.GetModelInstanceByName("iiwa_thanos"), self.home_q[:7])
        thanos_pose = self._plant.GetFrameByName("thanos_finger").CalcPoseInWorld(plant_context)
        self.reset_y = thanos_pose.translation()[1]
        
        self.target_obj2left_se2_pub = rospy.Publisher("/thanos_target_se2", Vector3, queue_size=1)
        self.target_obj2right_se2_pub = rospy.Publisher("/medusa_target_se2", Vector3, queue_size=1)
        

        
        self.home_q = np.array(self.home_q)
        
        self.des_q = curr_desired_joints(scenario_file=self.scenario_file)
        
        self.thanos_stiffness = np.ones(7) * 1000.0
        self.medusa_stiffness = np.ones(7) * 1000.0
        
        self.object_feedforward_force = obj_feedforward
    
    def get_torque_commanded(self):
        curr_torque = curr_torque_commanded(scenario_file=self.scenario_file)
        return curr_torque
    
    def get_gamma_wrench(self):
        curr_q = curr_joints(scenario_file=self.scenario_file)
        thanos_pose, medusa_pose = self.get_poses(curr_q)
        
        wrench_thanos = self.gamma_manager.get_thanos_wrench(thanos2world_rot=thanos_pose.rotation().matrix())
        wrench_medusa = self.gamma_manager.get_medusa_wrench(medusa2world_rot=medusa_pose.rotation().matrix())
        
        return wrench_thanos, wrench_medusa

    def get_gamma_joint_torque(self, filter_medusa = np.ones(6), filter_thanos = np.ones(6)):
        wrench_thanos, wrench_medusa = self.get_gamma_wrench()
        
        #zero everything but force xy
        if self.angle < np.pi/2:
            wrench_thanos[5] += self.grasp_force
            wrench_medusa[5] += self.grasp_force + self.object_feedforward_force
        else:
            wrench_thanos[5] += self.grasp_force + self.object_feedforward_force
            wrench_medusa[5] += self.grasp_force
        
        wrench_medusa = np.diag(filter_medusa) @ wrench_medusa
        wrench_thanos = np.diag(filter_thanos) @ wrench_thanos
        
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
    
    def zero_grasp_external_forces(self,filter_medusa = np.ones(6), filter_thanos = np.ones(6)):
        thanos_torque, medusa_torque = self.get_gamma_joint_torque(filter_medusa=filter_medusa, filter_thanos=filter_thanos)
        
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
        wrench_medusa = np.concatenate([np.zeros(3), np.array([0, 0.0, self.grasp_force + self.object_feedforward_force])])
        
        direct_joint_torque(des_q[:7], des_q[7:], wrench_thanos, wrench_medusa, endtime=5.0, scenario_file=self.scenario_file, directives_file=self.directives_file)
                
        delta_q = self.zero_grasp_external_forces()
        des_q[:7] += delta_q[:7]
        des_q[7:] += delta_q[7:]
        
        input("Press Enter to zero external forces")
        goto_joints_torque(des_q[:7], des_q[7:], wrench_thanos, wrench_medusa, endtime=5.0, scenario_file=self.scenario_file, directives_file=self.directives_file)
        
        gamma_wrench_thanos, gamma_wrench_medusa = self.get_gamma_wrench()
        print("gamma_wrench_thanos: ", gamma_wrench_thanos)
        print("gamma_wrench_medusa: ", gamma_wrench_medusa)
        
        
        # des_q = curr_joints(scenario_file=self.scenario_file)
        # direct_joint_torque(des_q[:7], des_q[7:], wrench_thanos, wrench_medusa, endtime=5.0, scenario_file=self.scenario_file, directives_file=self.directives_file)
        
        
        self.des_q = des_q
        
    def setup_robot(self, gap = 0.015):
        self.go_home()
        self.close_gripper(gap=gap)
        
    def rotate_arms(self, rotation, rotate_steps = 30, rotate_time = 30.0, grasp_force = 15.0, readjust_arms=True):
        curr_des_q = self.des_q
        thanos_pose, medusa_pose = self.get_poses(curr_des_q)
        current_obj2medusa_se2 = self.camera_manager.get_medusa_se2()
        
        thanos_ee_piecewise, medusa_ee_piecewise, T = inhand_rotate_traj(rotation, rotate_steps, rotate_time, thanos_pose, medusa_pose, current_obj2medusa_se2)
        thanos_piecewise, medusa_piecewise, T = generate_trajectory(self._plant, curr_des_q, thanos_ee_piecewise, medusa_ee_piecewise, T, tsteps=100)
        
        filter_vector_medusa = np.array([0,0,1,1,1,0])
        filter_vector_thanos = np.array([0,0,1,1,1,0])
        # follow_trajectory_apply_push(thanos_piecewise, medusa_piecewise, force=30.0, camera_manager=self.camera_manager, object_kg = 0.5, endtime = T, scenario_file=self.scenario_file, directives_file=self.directives_file)
        follow_traj_and_torque_gamma(thanos_piecewise, medusa_piecewise, self.camera_manager, self.gamma_manager, force=30.0, object_kg=0.5, endtime = T, scenario_file=self.scenario_file, directives_file=self.directives_file,
                                    filter_vector_medusa=filter_vector_medusa, filter_vector_thanos=filter_vector_thanos)
        self.angle += rotation
        des_thanos_q = thanos_piecewise.value(T).flatten()
        des_medusa_q = medusa_piecewise.value(T).flatten()
        des_q = np.concatenate([des_thanos_q, des_medusa_q])
        
        if readjust_arms:
            if self.angle < np.pi/2:
                wrench_thanos = np.concatenate([np.zeros(3), np.array([0, 0.0, 0])])
                wrench_medusa = np.concatenate([np.zeros(3), np.array([0, 0.0, grasp_force])])
            else:
                wrench_thanos = np.concatenate([np.zeros(3), np.array([0, 0.0, grasp_force])])
                wrench_medusa = np.concatenate([np.zeros(3), np.array([0, 0.0, 0])])
            
            goto_joints_torque(des_q[:7], des_q[7:], wrench_thanos, wrench_medusa, endtime=5.0, scenario_file=self.scenario_file, directives_file=self.directives_file)

        print("Sensing before se2")        
        gamma_wrench_thanos, gamma_wrench_medusa = self.get_gamma_wrench()
        print("gamma_wrench_thanos: ", gamma_wrench_thanos)
        print("gamma_wrench_medusa: ", gamma_wrench_medusa)
           
        # delta_q = self.zero_grasp_external_forces(filter_medusa=np.array([0,0,0,0,0,1]), filter_thanos=np.array([0,0,0,0,0,1]))
        # des_q[:7] += delta_q[:7]
        # des_q[7:] += delta_q[7:]
        
        self.des_q = des_q
        
    def se2_arms_open_loop(self, desired_obj2arm_se2, current_obj2arm_se2, medusa=True, se2_time=10.0, force = 10.0, object_kg = 1.0, filter_vector_medusa = np.array([0,0,1,1,1,0]), filter_vector_thanos = np.array([0,0,1,1,1,0])):
        curr_des_q = self.des_q
        thanos_pose, medusa_pose = self.get_poses(curr_des_q)
        thanos_ee_piecewise, medusa_ee_piecewise, T = inhand_se2_traj(thanos_pose, medusa_pose, current_obj2arm_se2, desired_obj2arm_se2, medusa=medusa, se2_time=se2_time)
        thanos_piecewise, medusa_piecewise, T = generate_trajectory(self._plant, curr_des_q, thanos_ee_piecewise, medusa_ee_piecewise, T, tsteps=100)
        follow_traj_and_torque_gamma_se2(desired_obj2arm_se2, thanos_piecewise, medusa_piecewise, self.camera_manager, self.gamma_manager, force=force, object_kg=object_kg, endtime = T, scenario_file=self.scenario_file, directives_file=self.directives_file,
                                     filter_vector_medusa=filter_vector_medusa, filter_vector_thanos=filter_vector_thanos, medusa=medusa)
        des_thanos_q = thanos_piecewise.value(T).flatten()
        des_medusa_q = medusa_piecewise.value(T).flatten()
        des_q = np.concatenate([des_thanos_q, des_medusa_q])
        self.des_q = des_q
        
    def se2_arms(self, desired_obj2arm_se2, medusa = True, se2_time = 10.0, force = 10.0, object_kg = 1.0, filter_vector_medusa = np.array([0,0,1,1,1,0]), filter_vector_thanos = np.array([0,0,1,1,1,0])):
        curr_des_q = self.des_q
        thanos_pose, medusa_pose = self.get_poses(curr_des_q)
        
        
        current_obj2arm_se2 = self.camera_manager.get_medusa_se2() if medusa else self.camera_manager.get_thanos_se2()
        
        thanos_ee_piecewise, medusa_ee_piecewise, T = inhand_se2_traj(thanos_pose, medusa_pose, current_obj2arm_se2, desired_obj2arm_se2, medusa=medusa, se2_time=se2_time)
        
        thanos_piecewise, medusa_piecewise, T = generate_trajectory(self._plant, curr_des_q, thanos_ee_piecewise, medusa_ee_piecewise, T, tsteps=100)
        
        # follow_traj_and_torque_gamma(thanos_piecewise, medusa_piecewise, self.camera_manager, self.gamma_manager, force=force, object_kg=object_kg, endtime = T, scenario_file=self.scenario_file, directives_file=self.directives_file,
        #                              filter_vector_medusa=filter_vector_medusa, filter_vector_thanos=filter_vector_thanos)
        follow_traj_and_torque_gamma_se2(desired_obj2arm_se2, thanos_piecewise, medusa_piecewise, self.camera_manager, self.gamma_manager, force=force, object_kg=object_kg, endtime = T, scenario_file=self.scenario_file, directives_file=self.directives_file,
                                     filter_vector_medusa=filter_vector_medusa, filter_vector_thanos=filter_vector_thanos, medusa=medusa)

        des_thanos_q = thanos_piecewise.value(T).flatten()
        des_medusa_q = medusa_piecewise.value(T).flatten()
        des_q = np.concatenate([des_thanos_q, des_medusa_q])
        self.des_q = des_q
    def move_back(self, endtime = 10.0):
        curr_q = curr_joints(scenario_file=self.scenario_file)
        current_thanos_q = curr_q[:7]
        current_medusa_q = curr_q[7:]
        thanos_pose, medusa_pose = self.get_poses(curr_q)
        current_obj2medusa_se2 = self.camera_manager.get_medusa_se2()
        
        z_offset = 0.005
        thanos2medusa = medusa_pose.inverse() @ thanos_pose
        current_obj2medusa = RigidTransform(RotationMatrix.MakeZRotation(current_obj2medusa_se2[2]), np.array([current_obj2medusa_se2[0], current_obj2medusa_se2[1], z_offset]))
        current_obj2world = medusa_pose @ current_obj2medusa
        
        # move back to original x
        desired_obj2world = RigidTransform(current_obj2world.rotation(), np.array([current_obj2world.translation()[0], self.reset_y, current_obj2world.translation()[2]]))
        desired_medusa2world = desired_obj2world @ current_obj2medusa.inverse()
        desired_thanos2world = desired_medusa2world @ thanos2medusa
        
        desired_q, _ = solveDualIK(self._plant, desired_thanos2world, desired_medusa2world, "thanos_finger", "medusa_finger", curr_q)
        desired_thanos_q = desired_q[:7]
        desired_medusa_q = desired_q[7:]
        
        print("thanos difference:", desired_thanos2world.translation() - thanos_pose.translation())
        print("medusa difference:", desired_medusa2world.translation() - medusa_pose.translation())
        
        ts = [0, endtime]
        thanos_qs = np.array([current_thanos_q, desired_thanos_q]).T
        medusa_qs = np.array([current_medusa_q, desired_medusa_q]).T
        thanos_q_traj = PiecewisePolynomial.FirstOrderHold(ts, thanos_qs)
        medusa_q_traj = PiecewisePolynomial.FirstOrderHold(ts, medusa_qs)
        
        filter_vector_medusa = np.array([0,0,1,1,1,0])
        filter_vector_thanos = np.array([0,0,1,1,1,0])
        # follow_trajectory_apply_push(thanos_piecewise, medusa_piecewise, force=30.0, camera_manager=self.camera_manager, object_kg = 0.5, endtime = T, scenario_file=self.scenario_file, directives_file=self.directives_file)
        follow_traj_and_torque_gamma(thanos_q_traj, medusa_q_traj, self.camera_manager, self.gamma_manager, force=30.0, object_kg=0.5, endtime = endtime, scenario_file=self.scenario_file, directives_file=self.directives_file,
                                    filter_vector_medusa=filter_vector_medusa, filter_vector_thanos=filter_vector_thanos)
        
        des_q = np.concatenate([desired_thanos_q, desired_medusa_q])
        self.des_q = des_q
        
if __name__ == "__main__":
    rospy.init_node("bimanual_kuka")
    bimanual_kuka = BimanualKuka(scenario_file="../../config/bimanual_med_hardware_gamma.yaml", directives_file="../../config/bimanual_med_gamma.yaml", grasp_force=20.0)
    rospy.sleep(0.1)
    desired_obj2left_se2 = np.array([0.0, 0.02, np.pi])
    desired_obj2right_se2 = np.array([0.00, -0.02, 0.0])
    
    bimanual_kuka.setup_robot(gap=0.012)
    
    # bimanual_kuka.se2_arms(np.array([0.0,0.03,np.pi]), medusa=False, se2_time=20.0, force = 0.0, object_kg = 1.5, filter_vector_medusa=np.array([1,1,1,1,1,0]), filter_vector_thanos=np.array([1,1,1,1,1,1]))
    # bimanual_kuka.move_back(endtime=10.0)

    # input("Press Enter to rotate arms")
    # bimanual_kuka.rotate_arms(45 * np.pi/180)
    # input("Press Enter to move arms")
    # bimanual_kuka.se2_arms(np.array([0.0,0.03,np.pi]), medusa=False, se2_time=20.0, force = 0.0, object_kg = 3.0, filter_vector_medusa=np.array([1,1,1,1,1,0]), filter_vector_thanos=np.array([1,1,1,1,1,1]))
    # input("Press Enter to rotate arms")
    # bimanual_kuka.rotate_arms(-45 * np.pi/180)

    # bimanual_kuka.rotate_arms(20 * np.pi/180, rotate_time = 20)
    # bimanual_kuka.se2_arms(np.array([0.0,0.03,np.pi]), medusa=False, se2_time=20.0, force = 0.0, object_kg = 2.0, filter_vector_medusa=np.array([1,1,1,1,1,0]), filter_vector_thanos=np.array([1,1,1,1,1,1]))
    # bimanual_kuka.rotate_arms(70 * np.pi/180, rotate_time = 20)
    # bimanual_kuka.move_back(endtime=20.0)
    # bimanual_kuka.rotate_arms(-90 * np.pi/180, rotate_time = 20)
    
    # real test
    # run 2.5 for even
    # run 3.5 for uneven
    
    # bimanual_kuka.rotate_arms(20 * np.pi/180, rotate_time = 15, grasp_force=20.0)
    # bimanual_kuka.se2_arms(desired_obj2left_se2, medusa=False, se2_time=20.0, force = 0.0, object_kg = 2.0, filter_vector_medusa=np.array([1,1,1,1,1,0]), filter_vector_thanos=np.array([0,0,1,1,1,1]))
    # bimanual_kuka.se2_arms(desired_obj2right_se2, medusa=True, se2_time=20.0, force = 0.0, object_kg = 2.0, filter_vector_medusa=np.array([1,1,1,1,1,0]), filter_vector_thanos=np.array([0,0,1,1,1,1]))
    
    bimanual_kuka.rotate_arms(20 * np.pi/180, rotate_time = 15, grasp_force=25.0)
    bimanual_kuka.se2_arms(desired_obj2left_se2, medusa=False, se2_time=20.0, force = 0.0, object_kg = 3.0, filter_vector_medusa=np.array([1,1,1,1,1,0]), filter_vector_thanos=np.array([0,0,1,1,1,1]))
    bimanual_kuka.rotate_arms(70 * np.pi/180, rotate_time = 15, grasp_force=25.0, readjust_arms=False)
    
    # bimanual_kuka.rotate_arms(70 * np.pi/180, rotate_time = 15, grasp_force=20.0, readjust_arms=False)
    # bimanual_kuka.move_back(endtime=15.0)
    bimanual_kuka.rotate_arms(70 * np.pi/180, rotate_time = 15, grasp_force=25.0, readjust_arms=False)
    # bimanual_kuka.rotate_arms(140 * np.pi/180, rotate_time = 30, grasp_force=20.0)
    bimanual_kuka.se2_arms(desired_obj2right_se2, medusa=True, se2_time=20.0, force = 0.0, object_kg = 3.0, filter_vector_medusa=np.array([0,0,1,1,1,1]), filter_vector_thanos=np.array([1,1,1,1,1,0]))
    
    # bimanual_kuka.rotate_arms(-70 * np.pi/180, rotate_time = 30, grasp_force=20.0, readjust_arms=False)
    # bimanual_kuka.move_back(endtime=15.0)
    bimanual_kuka.rotate_arms(-160 * np.pi/180, rotate_time = 30, grasp_force=25.0)
    
    print("Finished demo")
    current_obj2left_se2, current_obj2right_se2 = bimanual_kuka.get_obj_relative_poses()
    print("error obj2left: ", desired_obj2left_se2 - current_obj2left_se2)
    print("error obj2right: ", desired_obj2right_se2 - current_obj2right_se2)
    rospy.spin()
    