#! /usr/bin/env python3
import rospy
import numpy as np
import argparse

from geometry_msgs.msg import WrenchStamped, PoseStamped
from mmint_utils.config import dump_cfg, load_cfg
from tf import TransformListener
import tf.transformations as tr
import copy
import os

def skew(vector: np.ndarray):
    return np.array([[0, -vector[2], vector[1]],
                     [vector[2], 0, -vector[0]],
                     [-vector[1], vector[0], 0]])


if __name__ == '__main__':
    package_path = os.path.dirname(os.path.realpath(__file__)).split('mmint_wrench_utils')[0]
    config_dir = os.path.join(package_path, 'mmint_wrench_utils/config', 'gravity_params.yaml')
    
    rospy.init_node("netft_compensated")
    gravity_cfg = load_cfg(config_dir)
    force_ext = np.array(gravity_cfg['force_ext'])
    pos_fext  = np.array(gravity_cfg['pos_fext'])
    wrench_bias = np.array(gravity_cfg['wrench_bias'])
    
    tf_listener = TransformListener()
    world_frame_name = 'world'
    gamma_frame_name = 'gamma_on_hand_link_ft'
    
    compensated_wrench_pub = rospy.Publisher('/netft/compensated_netft_data', WrenchStamped)
    
    pos_fexthat = skew(pos_fext)
    
    def wrench_callback(data: WrenchStamped):
        tf_listener.waitForTransform(world_frame_name, gamma_frame_name, rospy.Time(), rospy.Duration(1.0))
        t = tf_listener.getLatestCommonTime(world_frame_name, gamma_frame_name)
        world_to_gamma_frame_transform = tf_listener.lookupTransform(gamma_frame_name, world_frame_name, t)
        world_to_gamma_quaternion = np.array(world_to_gamma_frame_transform[1])
        world_to_gamma_rotation_matrix = tr.quaternion_matrix(world_to_gamma_quaternion)
        
        force_in_gamma_frame = world_to_gamma_rotation_matrix[0:3,0:3] @ force_ext
        torque_in_gamma_frame = pos_fexthat @ force_in_gamma_frame
        
        
        new_wrenchstamped = copy.deepcopy(data)
        new_wrenchstamped.wrench.force.x = data.wrench.force.x - wrench_bias[0] - force_in_gamma_frame[0]
        new_wrenchstamped.wrench.force.y = data.wrench.force.y - wrench_bias[1] - force_in_gamma_frame[1]
        new_wrenchstamped.wrench.force.z = data.wrench.force.z - wrench_bias[2] - force_in_gamma_frame[2]
        new_wrenchstamped.wrench.torque.x = data.wrench.torque.x - wrench_bias[3] - torque_in_gamma_frame[0]
        new_wrenchstamped.wrench.torque.y = data.wrench.torque.y - wrench_bias[4] - torque_in_gamma_frame[1]
        new_wrenchstamped.wrench.torque.z = data.wrench.torque.z - wrench_bias[5] - torque_in_gamma_frame[2]
        
        compensated_wrench_pub.publish(new_wrenchstamped)
    
    netft_sub = rospy.Subscriber('/netft/netft_data', WrenchStamped, wrench_callback)
    rospy.spin()