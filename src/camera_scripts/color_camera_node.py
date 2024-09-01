import rospy
import cv2
from sensor_msgs.msg import Image, CameraInfo
from apriltag_ros.msg import AprilTagDetectionArray, AprilTagDetection
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import tf

#object yaw 0 visual
'''
    yaw = 0
        4
    7       5
        6
    
'''
class TagVisualization:
    def __init__(self):
        self.medusa_cam_sub = rospy.Subscriber("/panda_1_gelslim_left/tag_detections_image", Image, self.medusa_callback)
        # self.thanos_cam_sub = rospy.Subscriber("/panda_1_gelslim_right/image_undistorted", Image, self.thanos_callback)
        
        self.medusa_tag_sub = rospy.Subscriber("/panda_1_gelslim_left/tag_detections", AprilTagDetectionArray, self.medusa_tag_callback)
        # self.thanos_cam_sub = rospy.Subscriber("/panda_1_gelslim_right/tag_detections", AprilTagDetectionArray, self.thanos_tag_callback)
        self.bridge = CvBridge()
        
        self.id4_se2 = None
        self.id5_se2 = None
        self.id6_se2 = None
        self.id7_se2 = None
    def medusa_callback(self, data: Image):
        
        h,w = data.height, data.width
        cv_image = cv2.resize(self.bridge.imgmsg_to_cv2(data, "bgr8"), (w,h))
        
        rotation = -np.pi/3 + np.pi/12
        rotated_x = np.array([np.cos(rotation), -np.sin(rotation)])
        rotated_y = np.array([np.sin(rotation), np.cos(rotation)])
        
        # draw rotated x-axis in center of image in red
        cv2.line(cv_image, (w//2, h//2), (w//2 + int(rotated_x[0]*100), h//2 + int(rotated_x[1]*100)), (0,0,255), 3)
        # draw rotated y-axis in center of image in green
        cv2.line(cv_image, (w//2, h//2), (w//2 + int(rotated_y[0]*100), h//2 + int(rotated_y[1]*100)), (0,255,0), 3)
        
        #draw blue circle in center of image
        cv2.circle(cv_image, (w//2, h//2), 10, (255,0,0), -1)
        
        if self.id4_se2 is not None:
            x,y,yaw = self.id4_se2
            # draw red circle at tag position
            cv2.circle(cv_image, (int(w//2 + x*100), int(h//2 + y*100)), 10, (0,0,255), -1)
            # draw red line in direction of tag yaw
            cv2.line(cv_image, (w//2 + int(x*100), h//2 + int(y*100)), (w//2 + int(x*100 + np.cos(yaw)*100), h//2 + int(y*100 + np.sin(yaw)*100)), (0,0,255), 3)
        if self.id5_se2 is not None:
            x,y,yaw = self.id5_se2
            cv2.circle(cv_image, (int(w//2 + x*100), int(h//2 + y*100)), 10, (0,255,0), -1)
            cv2.line(cv_image, (w//2 + int(x*100), h//2 + int(y*100)), (w//2 + int(x*100 + np.cos(yaw)*100), h//2 + int(y*100 + np.sin(yaw)*100)), (0,255,0), 3)
        if self.id6_se2 is not None:
            x,y,yaw = self.id6_se2
            cv2.circle(cv_image, (int(w//2 + x*100), int(h//2 + y*100)), 10, (255,0,0), -1)
            cv2.line(cv_image, (w//2 + int(x*100), h//2 + int(y*100)), (w//2 + int(x*100 + np.cos(yaw)*100), h//2 + int(y*100 + np.sin(yaw)*100)), (255,0,0), 3)
        if self.id7_se2 is not None:
            x,y,yaw = self.id7_se2
            cv2.circle(cv_image, (int(w//2 + x*100), int(h//2 + y*100)), 10, (0,128,128), -1)
            cv2.line(cv_image, (w//2 + int(x*100), h//2 + int(y*100)), (w//2 + int(x*100 + np.cos(yaw)*100), h//2 + int(y*100 + np.sin(yaw)*100)), (0,128,128), 3)
        cv2.imshow("Medusa", cv_image)
        cv2.waitKey(1)
    def medusa_tag_callback(self, data: AprilTagDetectionArray):
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
            if id == 4:
                self.id4_se2 = (x/z,y/z,yaw)
            elif id == 5:
                self.id5_se2 = (x/z,y/z,yaw)
            elif id == 6:
                self.id6_se2 = (x/z,y/z,yaw)
            elif id == 7:
                self.id7_se2 = (x/z,y/z,yaw)
            if id == 7:
                print(f"Medusa detected tag {id} at ({x},{y}, {z}) with yaw {yaw * 180 / np.pi}")
            print(id)
if __name__ == '__main__':
    rospy.init_node('tag_visualization')
    tv = TagVisualization()
    rospy.spin()