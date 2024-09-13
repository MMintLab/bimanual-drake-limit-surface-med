import rospy
import numpy as np
from camera import CameraManager
if __name__ == '__main__':
    rospy.init_node("calc_camera_error")
    camera_manager = CameraManager()
    rospy.sleep(0.1)
    
    desired_obj2left_se2 = np.array([0.0, 0.02, -np.pi])
    desired_obj2right_se2 = np.array([0.0, -0.02, 0.0])
    
    obj2left_se2 = camera_manager.get_thanos_se2()
    obj2right_se2 = camera_manager.get_medusa_se2()
    
    print("Error: ", np.round(desired_obj2left_se2 - obj2left_se2,4))
    print("Error: ", np.round(desired_obj2right_se2 - obj2right_se2,4))
    