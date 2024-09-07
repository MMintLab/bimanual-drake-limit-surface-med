
import rospy
from geometry_msgs.msg import WrenchStamped
from netft_rdt_driver.srv import Zero
from pydrake.all import (
    StartMeshcat,
    RotationMatrix
)
from scipy.linalg import block_diag
import numpy as np
from mmint_utils.config import load_cfg

def wrenchstamped2numpy(msg: WrenchStamped):
    return np.array([msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z, 
                     msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])

class GammaManager:
    def __init__(self, use_compensation = False):
        self.thanos_ati_sub = rospy.Subscriber('/netft_thanos/netft_data', WrenchStamped, self.thanos_cb)
        self.medusa_ati_sub = rospy.Subscriber('/netft_medusa/netft_data', WrenchStamped, self.medusa_cb)
        self.thanos_wrench = None
        self.medusa_wrench = None
        rotationMat_medusa = RotationMatrix.MakeZRotation(135 * np.pi / 180.0).matrix()
        rotationMat_thanos = RotationMatrix.MakeZRotation(-90 * np.pi / 180.0).matrix()
        #blkdiag
        self.R_medusa = block_diag(rotationMat_medusa, rotationMat_medusa)
        self.R_thanos = block_diag(rotationMat_thanos, rotationMat_thanos)
        
        self.medusa_wrench_bias = np.zeros(6)
        self.medusa_pos_com = np.zeros(3)
        self.medusa_force_ext = np.zeros(3)
        
        self.thanos_wrench_bias = np.zeros(6)
        self.thanos_pos_com = np.zeros(3)
        self.thanos_force_ext = np.zeros(3)
        if use_compensation:
            medusa_cfg = load_cfg("../config/medusa_gravity_params.yaml")
            thanos_cfg = load_cfg("../config/thanos_gravity_params.yaml")
            self.medusa_wrench_bias = np.array(medusa_cfg['wrench_bias'])
            self.medusa_pos_com = np.array(medusa_cfg['pos_fext'])
            self.medusa_force_ext = np.array(medusa_cfg['force_ext'])
            
            self.thanos_wrench_bias = np.array(thanos_cfg['wrench_bias'])
            self.thanos_pos_com = np.array(thanos_cfg['pos_fext'])
            self.thanos_force_ext = np.array(thanos_cfg['force_ext'])
        
    def thanos_cb(self, msg: WrenchStamped):
        self.thanos_wrench = self.R_thanos @ wrenchstamped2numpy(msg)
    def medusa_cb(self, msg: WrenchStamped):
        self.medusa_wrench = self.R_medusa @ wrenchstamped2numpy(msg)
    def get_thanos_wrench(self, thanos2world_rot = None):
        if not (thanos2world_rot is None): 
            force_ext_thanos_frame = thanos2world_rot.T @ self.thanos_force_ext
            torque_ext_thanos_frame = np.cross(self.thanos_pos_com, force_ext_thanos_frame)
            wrench_ext_thanos_frame = np.concatenate([torque_ext_thanos_frame, force_ext_thanos_frame])
        else:
            wrench_ext_thanos_frame = np.zeros(6)
        
        return self.thanos_wrench - self.thanos_wrench_bias - wrench_ext_thanos_frame
    def get_medusa_wrench(self, medusa2world_rot = None):
        if not (medusa2world_rot is None): 
            force_ext_medusa_frame = medusa2world_rot.T @ self.medusa_force_ext
            torque_ext_medusa_frame = np.cross(self.medusa_pos_com, force_ext_medusa_frame)
            wrench_ext_medusa_frame = np.concatenate([torque_ext_medusa_frame, force_ext_medusa_frame])
        else:
            wrench_ext_medusa_frame = np.zeros(6)
        
        return self.medusa_wrench - self.medusa_wrench_bias - wrench_ext_medusa_frame
    
    @staticmethod
    def zero_sensor():
        rospy.wait_for_service('/netft_thanos/zero')
        rospy.wait_for_service('/netft_medusa/zero')
        zero_thanos = rospy.ServiceProxy('/netft_thanos/zero', Zero)
        zero_medusa = rospy.ServiceProxy('/netft_medusa/zero', Zero)
        zero_thanos()
        zero_medusa()