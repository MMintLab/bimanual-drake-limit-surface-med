#!/usr/bin/env python
import rospy
import cv2
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Vector3
from apriltag_ros.msg import AprilTagDetectionArray, AprilTagDetection
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import tf
import threading
from undistort_camera import camerainfo2parameters, CameraParameters
class TagVisualization:
    def __init__(self):
        self.medusa_info_intrinsics = camerainfo2parameters(rospy.wait_for_message("/panda_1_gelslim_left/camera_info", CameraInfo)).intrinsics
        self.thanos_info_intrinsics = camerainfo2parameters(rospy.wait_for_message("/panda_1_gelslim_right/camera_info", CameraInfo)).intrinsics
        
        self.bridge = CvBridge()
        self.medusa_cam_sub = rospy.Subscriber("/panda_1_gelslim_left/undistorted/tag_detections_image", Image, self.medusa_callback, queue_size=1)
        self.thanos_cam_sub = rospy.Subscriber("/panda_1_gelslim_right/undistorted/tag_detections_image", Image, self.thanos_callback, queue_size=1)
        self.medusa_cam_pub = rospy.Publisher("/panda_1_gelslim_left/pose_image", Image, queue_size=1)
        self.thanos_cam_pub = rospy.Publisher("/panda_1_gelslim_right/pose_image", Image, queue_size=1)
        
        self.medusa_tag_sub = rospy.Subscriber("/panda_1_gelslim_left/undistorted/tag_detections", AprilTagDetectionArray, self.medusa_tag_callback, queue_size=1)
        self.thanos_tag_sub = rospy.Subscriber("/panda_1_gelslim_right/undistorted/tag_detections", AprilTagDetectionArray, self.thanos_tag_callback, queue_size=1)
        
        self.thanos_pose_pub = rospy.Publisher("/thanos_se2_pose", Vector3, queue_size=1)
        self.medusa_pose_pub = rospy.Publisher("/medusa_se2_pose", Vector3, queue_size=1)
        
        self.medusa_se2 = None
        self.thanos_se2 = None
        
        self.medusa_rotation = 0
        self.thanos_rotation = -np.pi/4
        
        self.lock = threading.Lock()
    def medusa_callback(self, data: Image):
        cv_image = self.image_callback_fn(data, self.medusa_se2, self.medusa_info_intrinsics, rotation=self.medusa_rotation)
        self.medusa_cam_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        
    def medusa_tag_callback(self, data: AprilTagDetectionArray):
        se2z = self.tag_callback_fn(data)
        
        if se2z is not None:
            self.medusa_se2 = se2z
            
        if self.medusa_se2 is not None:
            x,y,z,yaw = self.medusa_se2
            rot = np.array([[np.cos(self.medusa_rotation), -np.sin(self.medusa_rotation)], [np.sin(self.medusa_rotation), np.cos(self.medusa_rotation)]])
            x,y = rot @ np.array([x,y])
            yaw = yaw + self.medusa_rotation
            self.medusa_pose_pub.publish(Vector3(x,y,yaw))
        
    def thanos_callback(self, data: Image):
        cv_image = self.image_callback_fn(data, self.thanos_se2, self.thanos_info_intrinsics, rotation = self.thanos_rotation)
        self.thanos_cam_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        
    def thanos_tag_callback(self, data: AprilTagDetectionArray):
        se2z = self.tag_callback_fn(data)
        if se2z is not None:
            self.thanos_se2 = se2z
        if self.thanos_se2 is not None:
            x,y,z,yaw = self.thanos_se2
            rot = np.array([[np.cos(self.thanos_rotation), -np.sin(self.thanos_rotation)], [np.sin(self.thanos_rotation), np.cos(self.thanos_rotation)]])
            x,y = rot @ np.array([x,y])
            yaw = yaw + self.thanos_rotation
            self.thanos_pose_pub.publish(Vector3(x,y,yaw))
        
    def image_callback_fn(self, data: Image, se2: tuple, intrinsics: np.array, rotation = 0):
        h,w = data.height, data.width
        cv_image = cv2.resize(self.bridge.imgmsg_to_cv2(data, "bgr8"), (w,h))
        
        rotated_x = np.array([np.cos(rotation), -np.sin(rotation)])
        rotated_y = np.array([np.sin(rotation), np.cos(rotation)])
        
        # draw rotated x-axis in center of image in red
        cv2.line(cv_image, (w//2, h//2), (w//2 + int(rotated_x[0]*100), h//2 + int(rotated_x[1]*100)), (0,0,255), 3)
        # draw rotated y-axis in center of image in green
        cv2.line(cv_image, (w//2, h//2), (w//2 + int(rotated_y[0]*100), h//2 + int(rotated_y[1]*100)), (0,255,0), 3)
        
        #draw blue circle in center of image
        cv2.circle(cv_image, (w//2, h//2), 5, (255,0,0), -1)
        
        if se2 is not None:
            x,y,z,yaw = se2
            
            x = x / z
            y = y / z
            
            pixel_xy = (intrinsics @ np.array([x,y,1]))[:2]
            pixel_x = pixel_xy[0]
            pixel_y = pixel_xy[1]
            # draw red circle at tag position
            cv2.circle(cv_image, (int(pixel_x), int(pixel_y)), 5, (0,0,255), -1)
            # draw red line in direction of tag yaw
            cv2.line(cv_image, (int(pixel_x), int(pixel_y)), (int(pixel_x + 100*np.cos(yaw)), int(pixel_y + 100*np.sin(yaw))), (0,0,255), 3)
                     
        return cv_image
    def tag_callback_fn(self, data: AprilTagDetectionArray):
        for detection in data.detections:
            # only detect tag 4
            id = detection.id[0]
            x = detection.pose.pose.pose.position.x
            y = detection.pose.pose.pose.position.y
            z = detection.pose.pose.pose.position.z
            quaternion = detection.pose.pose.pose.orientation
            quaternion_array = np.array([quaternion.x, quaternion.y, quaternion.z, quaternion.w])
            euler = tf.transformations.euler_from_quaternion(quaternion_array)
            yaw = euler[2]
            return (x,y,z,yaw)
        return None
            
if __name__ == '__main__':
    rospy.init_node('camera_node')
    tv = TagVisualization()
    rospy.spin()