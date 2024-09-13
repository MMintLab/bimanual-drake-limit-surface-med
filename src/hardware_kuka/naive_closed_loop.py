#!/usr/bin/env python
import rospy
from bimanual_kuka import BimanualKuka
import numpy as np
import os
# ASSUME: start with medusa on bottom at 0 degrees and thanos at 180 degrees
PATH_GOALS = [
    (np.array([0.0, 0.03, np.pi]), np.array([0.0, -0.03, 0.0])),
    (np.array([0.0, 0.03, np.pi]), np.array([0.0, -0.03, 0.0])),
    (np.array([0.0, 0.03, np.pi]), np.array([0.0, -0.03, 0.0])),
    (np.array([0.0, 0.03, np.pi]), np.array([0.0, -0.03, 0.0])),
]
def naive_closed_loop(bimanual_kuka: BimanualKuka, qgoal_thanos, qgoal_medusa, angle = 30):
    assert 0 <= angle <= 180, "Angle must be between 0 and 180 degrees"
    #solve thanos goal, then medusa goal
    bimanual_kuka.rotate_arms(angle * np.pi / 180)
    bimanual_kuka.se2_arms(qgoal_thanos, medusa=False, se2_time = 20.0, force = 0.0, object_kg = 2.0, filter_vector_medusa=np.array([1,1,1,1,1,0]), filter_vector_thanos=np.array([1,1,1,1,1,1]))
    bimanual_kuka.rotate_arms((180 - 2*angle)  * np.pi / 180)
    bimanual_kuka.se2_arms(qgoal_medusa, medusa=True, se2_time = 20.0, force = 0.0, object_kg = 2.0, filter_vector_medusa=np.array([1,1,1,1,1,1]), filter_vector_thanos=np.array([1,1,1,1,1,0]))
    bimanual_kuka.rotate_arms(-(180 - angle) * np.pi / 180)
    
    # get the current se2 positions of end-effector
    qthanos = bimanual_kuka.camera_manager.get_thanos_se2()
    qmedusa = bimanual_kuka.camera_manager.get_medusa_se2()
    return qthanos, qmedusa

if __name__ == '__main__':
    rospy.init_node("naive_closed_loop_node")
    bimanual_kuka = BimanualKuka(scenario_file="../../config/bimanual_med_hardware_gamma.yaml", directives_file="../../config/bimanual_med_gamma.yaml", obj_feedforward=5.0)
    rospy.sleep(0.1)
    bimanual_kuka.setup_robot()
    
    is_uneven = False
    obj_classification = 'even' if not is_uneven else 'uneven'
    path_num = 0
    angle = 30
    qgoal_thanos = PATH_GOALS[path_num][0]
    qgoal_medusa = PATH_GOALS[path_num][1]
    
    # create fork, parent runs closed loop, child runs open loop
    qthanos, qmedusa = naive_closed_loop(bimanual_kuka, qgoal_thanos, qgoal_medusa, angle = 30)
    
    print("current se2 position of thanos: ", qthanos)
    print("current se2 position of medusa: ", qmedusa)
    print("goal se2 position of thanos: ", qgoal_thanos)
    print("goal se2 position of medusa: ", qgoal_medusa)
    
    data = np.stack((qthanos, qmedusa, qgoal_thanos, qgoal_medusa))
    # save data into data folder ./data
    
    #make folder if not exists
    if not os.path.exists(f"data/naive/closed_loop/circle/{obj_classification}"):
        os.makedirs(f"data/naive/closed_loop/circle/{obj_classification}")
    np.save(f"data/naive/closed_loop/circle/{obj_classification}/naive_angle_{angle}_path_{path_num}_MSE.npy", data)
    
    rospy.spin()