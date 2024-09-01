import rospy
import cv2
from sensor_msgs.msg import Image, CameraInfo
from apriltag_ros.msg import AprilTagDetectionArray, AprilTagDetection
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import tf
import threading
class TagVisualization:
    def __init__(self):
        self.medusa_cam_sub = rospy.Subscriber("/panda_1_gelslim_left/tag_detections_image", Image, self.medusa_callback)
        self.thanos_cam_sub = rospy.Subscriber("/panda_1_gelslim_right/tag_detections_image", Image, self.thanos_callback)
        
        self.medusa_tag_sub = rospy.Subscriber("/panda_1_gelslim_left/tag_detections", AprilTagDetectionArray, self.medusa_tag_callback)
        self.thanos_cam_sub = rospy.Subscriber("/panda_1_gelslim_right/tag_detections", AprilTagDetectionArray, self.thanos_tag_callback)
        self.bridge = CvBridge()
        
        self.medusa_se2 = None
        self.thanos_se2 = None
        self.lock = threading.Lock()
    def medusa_callback(self, data: Image):
        cv_image = self.image_callback_fn(data, self.medusa_se2)
        with self.lock:
            cv2.imshow("Medusa", cv_image)
            cv2.waitKey(1)
    def medusa_tag_callback(self, data: AprilTagDetectionArray):
        se2 = self.tag_callback_fn(data)
        self.medusa_se2 = se2 if se2 is not None else self.medusa_se2
    def thanos_callback(self, data: Image):
        cv_image = self.image_callback_fn(data, self.thanos_se2)
        with self.lock:
            cv2.imshow("Thanos", cv_image)
    def thanos_tag_callback(self, data: AprilTagDetectionArray):
        se2 = self.tag_callback_fn(data)
        self.thanos_se2 = se2 if se2 is not None else self.thanos_se2
    
    def image_callback_fn(self, data: Image, se2: tuple):
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
        
        if se2 is not None:
            x,y,yaw = se2
            # draw red circle at tag position
            cv2.circle(cv_image, (int(w//2 + x*100), int(h//2 + y*100)), 10, (0,0,255), -1)
            # draw red line in direction of tag yaw
            cv2.line(cv_image, (w//2 + int(x*100), h//2 + int(y*100)), (w//2 + int(x*100 + np.cos(yaw)*100), h//2 + int(y*100 + np.sin(yaw)*100)), (0,0,255), 3)
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
            return (x/z,y/z,yaw)
        return None
            
if __name__ == '__main__':
    rospy.init_node('tag_visualization')
    tv = TagVisualization()
    rospy.spin()