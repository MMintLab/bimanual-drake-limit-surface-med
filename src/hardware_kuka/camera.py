import rospy
from geometry_msgs.msg import Vector3
import numpy as np

class CameraManager:
    def __init__(self):
        self.thanos_se2_sub = rospy.Subscriber("/thanos_se2_pose", Vector3, self.thanos_se2_callback, queue_size=1)
        self.medusa_se2_sub = rospy.Subscriber("/medusa_se2_pose", Vector3, self.medusa_se2_callback, queue_size=1)
        self.obj2thanos = None
        self.obj2thanos = None
    def thanos_se2_callback(self, data: Vector3):
        self.obj2thanos = np.array([data.x, data.y, data.z])
    def medusa_se2_callback(self, data: Vector3):
        self.obj2medusa = np.array([data.x, data.y, data.z])
    def get_thanos_se2(self):
        return self.obj2thanos
    def get_medusa_se2(self):
        return self.obj2medusa