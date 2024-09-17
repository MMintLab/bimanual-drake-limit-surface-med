import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
# get times new roman font
if __name__ == '__main__':
    
    naive_obj2left_data = np.load("data/figure/open_loop/naive/obj2left_data.npy", allow_pickle=True)
    naive_obj2right_data = np.load("data/figure/open_loop/naive/obj2right_data.npy", allow_pickle=True)
    algo_obj2left_data = np.load("data/figure/open_loop/algo/obj2left_data.npy", allow_pickle=True)
    algo_obj2right_data = np.load("data/figure/open_loop/algo/obj2right_data.npy", allow_pickle=True)
    
    desired_obj2left = np.load("data/figure/open_loop/naive/desired_obj2left.npy", allow_pickle=True)
    desired_obj2right = np.load("data/figure/open_loop/naive/desired_obj2right.npy", allow_pickle=True)
    
    # convert from meters to mm
    desired_obj2left[:2] = desired_obj2left[:2] * 1000
    desired_obj2right[:2] = desired_obj2right[:2] * 1000
    naive_obj2left_data[:,:2] = naive_obj2left_data[:,:2] * 1000
    naive_obj2right_data[:,:2] = naive_obj2right_data[:,:2] * 1000
    algo_obj2left_data[:,:2] = algo_obj2left_data[:,:2] * 1000
    algo_obj2right_data[:,:2] = algo_obj2right_data[:,:2] * 1000
    
    
    start_obj2left = naive_obj2left_data[0]
    start_obj2right = naive_obj2right_data[0]
    
    # naive_obj2left_data[:,2] = np.rad2deg(naive_obj2left_data[:,2])
    
    fig = plt.figure()
    #make subplot
    ax1 = fig.add_subplot(2,1,1,projection='3d')
    ax2 = fig.add_subplot(2,1,2,projection='3d')
    
    #plot start as red circle
    ax1.plot(start_obj2left[0], start_obj2left[1], start_obj2left[2], 'ko', label='start pose')
    ax1.plot(naive_obj2left_data[:,0], naive_obj2left_data[:,1], naive_obj2left_data[:,2], 'r', label='baseline traj')
    ax1.plot(algo_obj2left_data[:,0], algo_obj2left_data[:,1], algo_obj2left_data[:,2], 'b', label='DLS traj')
    
    #make star bigger
    ax1.scatter(desired_obj2left[0], desired_obj2left[1], desired_obj2left[2], s=100, c='g', marker='*', label='target pose')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('THETA (rad)')
    ax1.set_xlim(0,55)
    ax1.set_ylim(0,55)
    ax1.set_zlim(0,-1.5)
    ax1.set_zticks([0, -0.5, -1.0, -1.5])
    ax1.set_title("Object Top")
    
    ax2.plot(start_obj2right[0], start_obj2right[1], start_obj2right[2], 'ko', label='start obj2right')
    ax2.plot(naive_obj2right_data[:,0], naive_obj2right_data[:,1], naive_obj2right_data[:,2], 'r', label='baseline traj')
    ax2.plot(algo_obj2right_data[:,0], algo_obj2right_data[:,1], algo_obj2right_data[:,2], 'b', label='our traj')
    
    ax2.scatter(desired_obj2right[0], desired_obj2right[1], desired_obj2right[2], s=100, c='g', marker='*', label='desired obj2right')
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_zlabel('THETA (rad)')
    ax2.set_aspect('auto')
    ax2.set_xlim(0,45)
    ax2.set_ylim(-25,25)
    ax2.set_zlim(0,-6.2)
    ax2.set_title("Object Bottom")
    leg = ax1.legend(loc='upper right')
    bb = leg.get_bbox_to_anchor().transformed(ax1.transAxes.inverted()) 
    xOffset = 1.5
    bb.x0 += xOffset
    bb.x1 += xOffset
    leg.set_bbox_to_anchor(bb, transform = ax1.transAxes)
    # ax2.legend(loc='upper right')
    
    #legend
    
    plt.show()
    pass