#!/usr/bin/env python
import rospy
from bimanual_kuka import BimanualKuka
import numpy as np
import os
# ASSUME: start with medusa on bottom at 0 degrees and thanos at 180 degrees
from planning.drake_inhand_planner2 import DualLimitSurfaceParams, inhand_planner
PATH_GOALS = [
    (np.array([0.0, 0.02, np.pi]), np.array([0.0, -0.02, 0.0])),
    (np.array([0.0, -0.02, np.pi]), np.array([0.0, 0.02, 0.0])),
    (np.array([0.02, 0.015, np.pi]), np.array([0.02, 0.015, -np.pi/2])),
    (np.array([0.0, 0.0, -np.pi/2]), np.array([0.0, 0.0, np.pi/2])),
]

#mm for tables
def algo_open_loop(bimanual_kuka: BimanualKuka, goal_thanos, goal_medusa, angle = 30):
    #get current se2 positions of end-effector
    current_thanos = np.array([0,0,np.pi])
    current_medusa = np.array([0,0,0])
    
    dls_params = DualLimitSurfaceParams(mu_A = 0.75, r_A = 0.04, N_A = 20.0, mu_B = 0.75, r_B = 0.04, N_B = 20.0)
    horizon = 7
    obj2left, obj2right, vs = inhand_planner(current_thanos, current_medusa, goal_thanos, goal_medusa, dls_params, steps = horizon, angle = 60, palm_radius=0.04, kv = 20.0)
    input("Press Enter to keep going")
    desired_obj2left_se2s = []
    desired_obj2right_se2s = []
    for i in range(1,horizon):
        if i % 2 == 1:
            desired_obj2left_se2s.append(obj2left[:,i])
        else:
            desired_obj2right_se2s.append(obj2right[:,i] * np.array([-1,1,-1]))
    
    # rotate to from horizontal to vertical
    bimanual_kuka.rotate_arms(90 * np.pi / 180, readjust_arms=False, rotate_time = 15.0)
    for desired_obj2left_se2, desired_obj2right_se2 in zip(desired_obj2left_se2s, desired_obj2right_se2s):
        bimanual_kuka.rotate_arms(-(90-angle) * np.pi / 180, rotate_time = 15.0, grasp_force = 30.0)
        bimanual_kuka.se2_arms_open_loop(desired_obj2left_se2, current_thanos, medusa=False, se2_time = 10.0, force = 0.0, object_kg = 3.5, filter_vector_medusa=np.array([1,1,1,1,1,0]), filter_vector_thanos=np.array([1,1,1,1,1,1]))
        bimanual_kuka.rotate_arms((90-angle) * np.pi / 180, readjust_arms=False, rotate_time=15.0, grasp_force = 30.0) #back to vertical
        bimanual_kuka.move_back(endtime=5.0)
        
        bimanual_kuka.rotate_arms((90-angle) * np.pi / 180, rotate_time = 15.0, grasp_force = 30.0)
        bimanual_kuka.se2_arms_open_loop(desired_obj2right_se2, current_medusa, medusa=True, se2_time = 10.0, force = 0.0, object_kg = 3.5, filter_vector_medusa=np.array([1,1,1,1,1,1]), filter_vector_thanos=np.array([1,1,1,1,1,0]))
        bimanual_kuka.rotate_arms(-(90-angle) * np.pi / 180, readjust_arms=False, rotate_time = 15.0, grasp_force=30.0) #back to vertical
        bimanual_kuka.move_back(endtime=5.0)
        
        current_thanos = desired_obj2left_se2
        current_medusa = desired_obj2right_se2
        
    qthanos = bimanual_kuka.camera_manager.get_thanos_se2()
    qmedusa = bimanual_kuka.camera_manager.get_medusa_se2()
    return qthanos, qmedusa
        
if __name__ == '__main__':
    rospy.init_node("algo_open_loop_node")
    bimanual_kuka = BimanualKuka(scenario_file="../../config/bimanual_med_hardware_gamma.yaml", directives_file="../../config/bimanual_med_gamma.yaml", obj_feedforward=5.0)
    rospy.sleep(0.1)
    bimanual_kuka.setup_robot(gap=0.012)
    
    path_num = 0
    angle = 60
    qgoal_thanos = PATH_GOALS[path_num][0]
    qgoal_medusa = PATH_GOALS[path_num][1]
    
    # create fork, parent runs open loop, child runs open loop
    qthanos, qmedusa = algo_open_loop(bimanual_kuka, qgoal_thanos, qgoal_medusa, angle = angle)
    
    print("current se2 position of thanos: ", qthanos)
    print("current se2 position of medusa: ", qmedusa)
    print("goal se2 position of thanos: ", qgoal_thanos)
    print("goal se2 position of medusa: ", qgoal_medusa)
    
    print("error thanos: ", np.linalg.norm(qthanos[:2] - qgoal_thanos[:2]))
    print("error medusa: ", np.linalg.norm(qmedusa[:2] - qgoal_medusa[:2]))
    
    data = np.stack((qthanos, qmedusa, qgoal_thanos, qgoal_medusa))
    # save data into data folder ./data
    
    #make folder if not exists
    if not os.path.exists(f"data/algo/open_loop/circle/even"):
        os.makedirs(f"data/algo/open_loop/circle/even")
    np.save(f"data/algo/open_loop/circle/even/algo_angle_{angle}_path_{path_num}_MSE.npy", data)
    
    bimanual_kuka.rotate_arms(-90 * np.pi / 180, readjust_arms=False)
    print("Done")
    rospy.spin()