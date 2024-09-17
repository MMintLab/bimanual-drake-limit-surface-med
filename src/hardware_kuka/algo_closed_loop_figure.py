#!/usr/bin/env python
import rospy
from bimanual_kuka import BimanualKuka
import numpy as np
import os
from geometry_msgs.msg import Vector3
# ASSUME: start with medusa on bottom at 0 degrees and thanos at 180 degrees
from planning.drake_inhand_planner2 import DualLimitSurfaceParams, inhand_planner
import multiprocessing as mp
PATH_GOALS = [
    (np.array([0.0, 0.02, np.pi]), np.array([0.0, -0.02, 0.0])),
    (np.array([0.0, -0.02, np.pi]), np.array([0.0, -0.02, 0.0])),
    (np.array([0.0, 0.00, np.pi]), np.array([0.0, 0.02, np.pi/4])),
    (np.array([0.0, 0.0, -np.pi/2]), np.array([0.0, 0.0, np.pi/2])),
]

#mm for tables
def algo_closed_loop(bimanual_kuka: BimanualKuka, idx_pathgoals, angle = 30):
    #get current se2 positions of end-effector
    current_thanos = np.array([0,0,np.pi])
    current_medusa = np.array([0,0,0])
    
    goal_thanos = PATH_GOALS[idx_pathgoals][0]
    goal_medusa = PATH_GOALS[idx_pathgoals][1]
    
    dls_params = DualLimitSurfaceParams(mu_A = 0.75, r_A = 0.04, N_A = 20.0, mu_B = 0.75, r_B = 0.04, N_B = 20.0)
    
    if idx_pathgoals == 0 or idx_pathgoals == 2:
        horizon = 7
    else:
        horizon = 3
    obj2left, obj2right, vs = inhand_planner(current_thanos, current_medusa, goal_thanos, goal_medusa, dls_params, steps = horizon, angle = 60, palm_radius=0.04, kv = 20.0)
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
        bimanual_kuka.rotate_arms(-(90-angle) * np.pi / 180, grasp_force = 30.0, rotate_time = 15.0, readjust_arms=False)
        bimanual_kuka.se2_arms(desired_obj2left_se2, medusa=False, se2_time = 10.0, force = 15.0, object_kg = 0.5, filter_vector_medusa=np.array([1,1,1,1,1,0]), filter_vector_thanos=np.array([1,1,1,1,1,0]))
        bimanual_kuka.rotate_arms((90-angle) * np.pi / 180, readjust_arms=False, rotate_time=15.0) #back to vertical
        bimanual_kuka.move_back(endtime=10.0)
        
        bimanual_kuka.rotate_arms((90-angle) * np.pi / 180, grasp_force = 30.0, rotate_time = 15.0, readjust_arms=False)
        bimanual_kuka.se2_arms(desired_obj2right_se2, medusa=True, se2_time = 10.0, force = 15.0, object_kg = 0.5, filter_vector_medusa=np.array([1,1,1,1,1,0]), filter_vector_thanos=np.array([1,1,1,1,1,0]))
        bimanual_kuka.rotate_arms(-(90-angle) * np.pi / 180, readjust_arms=False, rotate_time = 15.0) #back to vertical
        bimanual_kuka.move_back(endtime=10.0)
        
    qthanos = bimanual_kuka.camera_manager.get_thanos_se2()
    qmedusa = bimanual_kuka.camera_manager.get_medusa_se2()
    
    try:
        bimanual_kuka.rotate_arms(-90 * np.pi / 180, readjust_arms=False)
    except:
        pass
    
    return qthanos, qmedusa

class CollectCamera:
    def __init__(self):
        self.thanos_se2_sub = rospy.Subscriber("/thanos_se2_pose", Vector3, self.thanos_se2_callback, queue_size=1)
        self.medusa_se2_sub = rospy.Subscriber("/medusa_se2_pose", Vector3, self.medusa_se2_callback, queue_size=1)
        
        self.t_start = rospy.Time.now()
        self.obj2thanos_list = []
        self.obj2thanos_t_list = []
        self.obj2medusa_list = []
        self.obj2medusa_t_list = []
    def thanos_se2_callback(self, data: Vector3):
        obj2thanos = np.array([data.x, data.y, data.z])
        self.obj2thanos_list.append(obj2thanos)
        self.obj2thanos_t_list.append(rospy.Time.now() - self.t_start)
    def medusa_se2_callback(self, data: Vector3):
        obj2medusa = np.array([data.x, data.y, data.z])
        self.obj2medusa_list.append(obj2medusa)
        self.obj2medusa_t_list.append(rospy.Time.now() - self.t_start)

def algo_closed_loop_mp(bimanual_kuka: BimanualKuka, idx_pathgoals, angle = 30):
    # run algo_closed_loop as a Process
    q = mp.Queue()
    
    camera = CollectCamera()
    def fn(bimanual_kuka, idx_pathgoals, angle, q):
        results = algo_closed_loop(bimanual_kuka, idx_pathgoals, angle)
        # results = np.array([0,0,0]), np.array([0,0,0])
        q.put(results)
        
    p = mp.Process(target=fn, args=(bimanual_kuka, idx_pathgoals, angle, q))
    p.start()
    p.join()
    # qthanos, qmedusa = q.get()
    
    qthanos_traj = np.stack(camera.obj2thanos_list)
    qmedusa_traj = np.stack(camera.obj2medusa_list)
    qthanos_t = np.array(camera.obj2thanos_t_list)
    qmedusa_t = np.array(camera.obj2medusa_t_list)
    return qthanos_traj, qmedusa_traj, qthanos_t, qmedusa_t
    
if __name__ == '__main__':
    rospy.init_node("algo_closed_loop_node_figure")
    
    bimanual_kuka = BimanualKuka(scenario_file="../../config/bimanual_med_hardware_gamma.yaml", directives_file="../../config/bimanual_med_gamma.yaml", obj_feedforward=5.0)
    rospy.sleep(0.1)
    bimanual_kuka.setup_robot(gap=0.012)
    
    path_num = 0
    angle = 45
    
    qgoal_thanos = PATH_GOALS[path_num][0]
    qgoal_medusa = PATH_GOALS[path_num][1]
    
    # create fork, parent runs closed loop, child runs open loop
    
    qthanos_traj, qmedusa_traj, qthanos_t, qmedusa_t = algo_closed_loop_mp(bimanual_kuka, path_num, angle = angle)
    
    qthanos = np.array([0,0,np.pi])
    qmedusa = np.array([0,0,0])
    #make folder if not exists
    if not os.path.exists(f"data/algo/closed_loop/figure"):
        os.makedirs(f"data/algo/closed_loop/figure")
    np.save(f"data/algo/closed_loop/figure/algo_angle_{angle}_path_{path_num}_thanos_traj.npy", qthanos_traj)
    np.save(f"data/algo/closed_loop/figure/algo_angle_{angle}_path_{path_num}_medusa_traj.npy", qmedusa_traj)
    np.save(f"data/algo/closed_loop/figure/algo_angle_{angle}_path_{path_num}_thanos_t.npy", qthanos_t)
    np.save(f"data/algo/closed_loop/figure/algo_angle_{angle}_path_{path_num}_medusa_t.npy", qmedusa_t)
    np.save(f"data/algo/closed_loop/figure/algo_angle_{angle}_path_{path_num}_goal_thanos.npy", qgoal_thanos)
    np.save(f"data/algo/closed_loop/figure/algo_angle_{angle}_path_{path_num}_goal_medusa.npy", qgoal_medusa)
    np.save(f"data/algo/closed_loop/figure/algo_angle_{angle}_path_{path_num}_current_thanos.npy", qthanos)
    np.save(f"data/algo/closed_loop/figure/algo_angle_{angle}_path_{path_num}_current_medusa.npy", qmedusa)
    
    # plot the data
    
    # plot initial and desired configuration
    # 3d plot
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('THETA')
    ax.plot(qthanos[0], qthanos[1], qthanos[2], 'ro', label='current obj2thanos')
    ax.plot(qmedusa[0], qmedusa[1], qmedusa[2], 'bo', label='current obj2medusa')
    ax.plot(qgoal_thanos[0], qgoal_thanos[1], qgoal_thanos[2], 'r*', label='goal obj2thanos')
    ax.plot(qgoal_medusa[0], qgoal_medusa[1], qgoal_medusa[2], 'b*', label='goal obj2medusa')
    
    ax.plot(qthanos_traj[:,0], qthanos_traj[:,1], qthanos_traj[:,2], 'r', label='obj2thanos traj')
    ax.plot(qmedusa_traj[:,0], qmedusa_traj[:,1], qmedusa_traj[:,2], 'b', label='obj2medusa traj')
    ax.legend()
    plt.show()
    
    
    
    print("Done")
    rospy.spin()