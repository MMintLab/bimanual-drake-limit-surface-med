#!/usr/bin/env python
import rospy
from bimanual_kuka import BimanualKuka
import numpy as np
import os
# ASSUME: start with medusa on bottom at 0 degrees and thanos at 180 degrees
PATH_GOALS = [
    (np.array([0.0, 0.02, np.pi]), np.array([0.0, -0.02, 0.0])),
    (np.array([0.0, -0.02, np.pi]), np.array([0.0, 0.02, 0.0])),
    (np.array([0.0, -0.02, np.pi + np.pi/4]), np.array([0.0, 0.02, -np.pi/4])),
    (np.array([0.0, 0.0, -np.pi/2]), np.array([0.0, 0.0, np.pi/2])),
]
ONLY_ANGLES_ALLOWED = [20, 30, 45, 60]
def naive_closed_loop(bimanual_kuka: BimanualKuka, qgoal_thanos, qgoal_medusa, angle = 30):
    assert 0 <= angle <= 180, "Angle must be between 0 and 180 degrees"
    #solve thanos goal, then medusa goal
    bimanual_kuka.rotate_arms(angle * np.pi / 180, grasp_force = 25.0)
    bimanual_kuka.se2_arms(qgoal_thanos, medusa=False, se2_time = 20.0, force = 0.0, object_kg = 3.0, filter_vector_medusa=np.array([1,1,1,1,1,0]), filter_vector_thanos=np.array([0,0,1,1,1,1]))
    bimanual_kuka.se2_arms(qgoal_medusa, medusa=True, se2_time = 20.0, force = 0.0, object_kg = 3.0, filter_vector_medusa=np.array([0,0,1,1,1,1]), filter_vector_thanos=np.array([1,1,1,1,1,0]))
    bimanual_kuka.rotate_arms(-angle * np.pi / 180, readjust_arms=False)
    
    # get the current se2 positions of end-effector
    qthanos = bimanual_kuka.camera_manager.get_thanos_se2()
    qmedusa = bimanual_kuka.camera_manager.get_medusa_se2()
    return qthanos, qmedusa

if __name__ == '__main__':
    rospy.init_node("naive_closed_loop_node")
    bimanual_kuka = BimanualKuka(scenario_file="../../config/bimanual_med_hardware_gamma.yaml", directives_file="../../config/bimanual_med_gamma.yaml", obj_feedforward=5.0)
    rospy.sleep(0.1)
    bimanual_kuka.setup_robot(gap=0.012)
    
    path_num = 0
    angle = ONLY_ANGLES_ALLOWED[1]
    
    qgoal_thanos = PATH_GOALS[path_num][0]
    qgoal_medusa = PATH_GOALS[path_num][1]
    
    # create fork, parent runs closed loop, child runs open loop
    qthanos, qmedusa = naive_closed_loop(bimanual_kuka, qgoal_thanos, qgoal_medusa, angle = angle)
    
    print("current se2 position of thanos: ", qthanos)
    print("current se2 position of medusa: ", qmedusa)
    print("goal se2 position of thanos: ", qgoal_thanos)
    print("goal se2 position of medusa: ", qgoal_medusa)
    
    print("error thanos: ", np.linalg.norm(qthanos[:2] - qgoal_thanos[:2]))
    print("error medusa: ", np.linalg.norm(qmedusa[:2] - qgoal_medusa[:2]))
    
    data = np.stack((qthanos, qmedusa, qgoal_thanos, qgoal_medusa))
    # save data into data folder ./data
    
    #make folder if not exists
    object = "square"
    
    if not os.path.exists(f"data/naive/closed_loop/{object}/even"):
        os.makedirs(f"data/naive/closed_loop/{object}/even")
    np.save(f"data/naive/closed_loop/{object}/even/naive_angle_{angle}_path_{path_num}_MSE.npy", data)
    
    print("Done")
    rospy.spin()