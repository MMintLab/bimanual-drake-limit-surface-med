#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from copy import deepcopy

def camerainfo2parameters(data: CameraInfo):
    intrinsics = data.K
    distortion = data.D
    rotation = np.array(data.R).reshape((3,3))
    projection = np.array(list(data.P)).reshape((3,4))
    size = (data.width, data.height)
    return CameraParameters(intrinsics, distortion, rotation, projection, size)
class CameraParameters:
    def __init__(self, K, D, R, P, size, alpha = 0.0):
        self.intrinsics = np.array(K).reshape((3,3)) # 3x3 intrinsics matrix
        self.dist = np.array(D) # 5x1 distortion coefficients
        self.R = R # 3x3 rotation matrix
        self.P = P # 3x4 projection matrix
        self.size = size # (width, height)
        
        assert 0.0 <= alpha <= 1.0 # alpha=0: all pixels valid, alpha=1: no pixels lost
        ncm, _ = cv2.getOptimalNewCameraMatrix(self.intrinsics, self.dist, self.size, alpha)
        for j in range(3):
            for i in range(3):
                self.P[j,i] = ncm[j,i]
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(self.intrinsics, self.dist, self.R, self.P, self.size, cv2.CV_32FC1)        
        
    def undistort(self, img):
        return cv2.remap(img, self.mapx, self.mapy, cv2.INTER_LINEAR)
    
class CameraUndistorter:
    def __init__(self):
        self.bridge = CvBridge()
        self.gelslim_left_camera_info : CameraInfo = rospy.wait_for_message("/panda_1_gelslim_left/camera_info", CameraInfo)
        self.gelslim_left_info : CameraParameters = camerainfo2parameters(self.gelslim_left_camera_info)
        
        self.gelslim_right_camera_info : CameraInfo = rospy.wait_for_message("/panda_1_gelslim_right/camera_info", CameraInfo)
        self.gelslim_right_info : CameraParameters = camerainfo2parameters(self.gelslim_right_camera_info)
        
        self.gelslim_left_sub  = rospy.Subscriber("/panda_1_gelslim_left/image_raw", Image, self.gelslim_left_callback, queue_size=1)
        self.gelslim_right_sub = rospy.Subscriber("/panda_1_gelslim_right/image_raw", Image, self.gelslim_right_callback, queue_size=1)
        
        self.gelslim_left_pub = rospy.Publisher("/panda_1_gelslim_left/undistorted/image_raw", Image, queue_size=1)
        self.gelslim_right_pub = rospy.Publisher("/panda_1_gelslim_right/undistorted/image_raw", Image, queue_size=1)
        self.gelslim_left_info_pub = rospy.Publisher("/panda_1_gelslim_left/undistorted/camera_info", CameraInfo, queue_size=1)
        self.gelslim_right_info_pub = rospy.Publisher("/panda_1_gelslim_right/undistorted/camera_info", CameraInfo, queue_size=1)
        

    def gelslim_left_callback(self, data: Image):
        #undistort image
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        undistorted = self.bridge.cv2_to_imgmsg(self.gelslim_left_info.undistort(cv_image), "bgr8")
        undistorted.header.stamp = data.header.stamp
        
        
        data_info = deepcopy(self.gelslim_left_camera_info)
        data_info.header.stamp = data.header.stamp
        
        self.gelslim_left_pub.publish(undistorted)
        self.gelslim_left_info_pub.publish(data_info)
        
    def gelslim_right_callback(self, data: Image):
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        undistorted = self.bridge.cv2_to_imgmsg(self.gelslim_right_info.undistort(cv_image), "bgr8")
        undistorted.header.stamp = data.header.stamp
        
        
        data_info = deepcopy(self.gelslim_right_camera_info)
        data_info.header.stamp = data.header.stamp
        
        self.gelslim_right_pub.publish(undistorted)
        self.gelslim_right_info_pub.publish(data_info)
    

# This file is for republishing undistorted images
if __name__ == '__main__':
    rospy.init_node('undistort_camera')
    cm = CameraUndistorter()
    rospy.spin()