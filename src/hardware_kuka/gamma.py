
import rospy
from geometry_msgs.msg import WrenchStamped
from netft_rdt_driver.srv import Zero
from pydrake.all import (
    StartMeshcat,
    RotationMatrix
)
from scipy.linalg import block_diag
import numpy as np

def wrenchstamped2numpy(msg: WrenchStamped):
    return np.array([msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z, 
                     msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])

class GammaManager:
    def __init__(self):
        self.thanos_ati_sub = rospy.Subscriber('/netft_thanos/netft_data', WrenchStamped, self.thanos_cb)
        self.medusa_ati_sub = rospy.Subscriber('/netft_medusa/netft_data', WrenchStamped, self.medusa_cb)
        self.thanos_wrench = None
        self.medusa_wrench = None
        rotationMat_medusa = RotationMatrix.MakeZRotation(135 * np.pi / 180.0).matrix()
        rotationMat_thanos = RotationMatrix.MakeZRotation(-90 * np.pi / 180.0).matrix()
        #blkdiag
        self.R_medusa = block_diag(rotationMat_medusa, rotationMat_medusa)
        self.R_thanos = block_diag(rotationMat_thanos, rotationMat_thanos)
    def thanos_cb(self, msg: WrenchStamped):
        self.thanos_wrench = self.R_thanos @ wrenchstamped2numpy(msg)
    def medusa_cb(self, msg: WrenchStamped):
        self.medusa_wrench = self.R_medusa @ wrenchstamped2numpy(msg)
    def get_thanos_wrench(self):
        return self.thanos_wrench
    def get_medusa_wrench(self):
        return self.medusa_wrench
    
    @staticmethod
    def zero_sensor(self):
        rospy.wait_for_service('/netft_thanos/zero')
        rospy.wait_for_service('/netft_medusa/zero')
        zero_thanos = rospy.ServiceProxy('/netft_thanos/zero', Zero)
        zero_medusa = rospy.ServiceProxy('/netft_medusa/zero', Zero)
        zero_thanos()
        zero_medusa()