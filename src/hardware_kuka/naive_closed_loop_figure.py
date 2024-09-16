#!/usr/bin/env python
import rospy
from bimanual_kuka import BimanualKuka
import numpy as np
import os
from geometry_msgs.msg import Vector3
import multiprocessing as mp
# ASSUME: start with medusa on bottom at 0 degrees and thanos at 180 degrees
PATH_GOALS = [
    (np.array([0.0, 0.02, np.pi]), np.array([0.0, -0.02, 0.0])),
    (np.array([0.0, -0.02, np.pi]), np.array([0.0, 0.02, 0.0])),
    (np.array([0.02, 0.015, np.pi]), np.array([0.02, 0.015, -np.pi/2])),
    (np.array([0.0, 0.0, -np.pi/2]), np.array([0.0, 0.0, np.pi/2])),
]
def naive_closed_loop(bimanual_kuka: BimanualKuka, qgoal_thanos, qgoal_medusa, angle = 30):
    assert 0 <= angle <= 180, "Angle must be between 0 and 180 degrees"
    #solve thanos goal, then medusa goal
    bimanual_kuka.rotate_arms(angle * np.pi / 180)
    bimanual_kuka.se2_arms(qgoal_thanos, medusa=False, se2_time = 20.0, force = 0.0, object_kg = 2.0, filter_vector_medusa=np.array([1,1,1,1,1,0]), filter_vector_thanos=np.array([1,1,1,1,1,1]))
    bimanual_kuka.se2_arms(qgoal_medusa, medusa=True, se2_time = 20.0, force = 0.0, object_kg = 2.0, filter_vector_medusa=np.array([1,1,1,1,1,1]), filter_vector_thanos=np.array([1,1,1,1,1,0]))
    bimanual_kuka.rotate_arms(-angle * np.pi / 180, readjust_arms=False)
    
    # get the current se2 positions of end-effector
    qthanos = bimanual_kuka.camera_manager.get_thanos_se2()
    qmedusa = bimanual_kuka.camera_manager.get_medusa_se2()
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

def naive_closed_loop_mp(bimanual_kuka: BimanualKuka, qgoal_thanos, qgoal_medusa, angle = 30):
    q = mp.Queue()
    
    camera = CollectCamera()
    def fn(bimanual_kuka, goal_thanos, goal_medusa, angle, q):
        q.put(naive_closed_loop(bimanual_kuka, goal_thanos, goal_medusa, angle))
        
    p = mp.Process(target=fn, args=(bimanual_kuka, qgoal_thanos, qgoal_medusa, angle, q))
    p.start()
    p.join()
    # qthanos, qmedusa = q.get()
    
    qthanos_traj = np.array(camera.obj2thanos_list)
    qmedusa_traj = np.array(camera.obj2medusa_list)
    qthanos_t = np.array(camera.obj2thanos_t_list)
    qmedusa_t = np.array(camera.obj2medusa_t_list)
    return qthanos_traj, qmedusa_traj, qthanos_t, qmedusa_t

if __name__ == '__main__':
    rospy.init_node("naive_closed_loop_node")
    bimanual_kuka = BimanualKuka(scenario_file="../../config/bimanual_med_hardware_gamma.yaml", directives_file="../../config/bimanual_med_gamma.yaml", obj_feedforward=5.0)
    rospy.sleep(0.1)
    bimanual_kuka.setup_robot(gap=0.012)
    
    path_num = 2
    angle = 60
    qgoal_thanos = PATH_GOALS[path_num][0]
    qgoal_medusa = PATH_GOALS[path_num][1]
    
    # create fork, parent runs closed loop, child runs open loop
    qthanos_traj, qmedusa_traj, qthanos_t, qmedusa_t = naive_closed_loop_mp(bimanual_kuka, qgoal_thanos, qgoal_medusa, angle = angle)
    
    # save data into data folder ./data
    
    qthanos = np.array([0,0,np.pi])
    qmedusa = np.array([0,0,0])
    
    #make folder if not exists
    #make folder if not exists
    if not os.path.exists(f"data/naive/closed_loop/figure"):
        os.makedirs(f"data/naive/closed_loop/figure")
    np.save(f"data/naive/closed_loop/figure/naive_angle_{angle}_path_{path_num}_thanos_traj.npy", qthanos_traj)
    np.save(f"data/naive/closed_loop/figure/naive_angle_{angle}_path_{path_num}_medusa_traj.npy", qmedusa_traj)
    np.save(f"data/naive/closed_loop/figure/naive_angle_{angle}_path_{path_num}_thanos_t.npy", qthanos_t)
    np.save(f"data/naive/closed_loop/figure/naive_angle_{angle}_path_{path_num}_medusa_t.npy", qmedusa_t)
    np.save(f"data/naive/closed_loop/figure/naive_angle_{angle}_path_{path_num}_qgoal_thanos.npy", qgoal_thanos)
    np.save(f"data/naive/closed_loop/figure/naive_angle_{angle}_path_{path_num}_qgoal_medusa.npy", qgoal_medusa)
    np.save(f"data/naive/closed_loop/figure/naive_angle_{angle}_path_{path_num}_qthanos.npy", qthanos)
    np.save(f"data/naive/closed_loop/figure/naive_angle_{angle}_path_{path_num}_qmedusa.npy", qmedusa)
    # plot the data
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