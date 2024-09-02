#!/usr/bin/env python3


import rospy
import numpy as np
import tf.transformations as tr
import time
import os

from geometry_msgs.msg import WrenchStamped, PoseStamped
from bubble_utils.bubble_med.bubble_med import BubbleMed
from mmint_utils.gamma_helpers import zero_ati_gamma

from mmint_tools import tr
from mmint_utils.config import dump_cfg, load_cfg
from geometry_msgs.msg import WrenchStamped

import argparse

from tqdm import tqdm

def get_wrench():
    return rospy.wait_for_message('/netft/netft_data', WrenchStamped)
    

def wrenchstamped_to_numpy(wrenchstamped):
    return np.array([wrenchstamped.wrench.force.x, wrenchstamped.wrench.force.y, wrenchstamped.wrench.force.z, 
                    wrenchstamped.wrench.torque.x, wrenchstamped.wrench.torque.y, wrenchstamped.wrench.torque.z])  
def numpy_to_wrenchstamped(numpyarray: np.ndarray, frame_id = 'gamma_on_hand_link_ft'):
    wrenchstamped = WrenchStamped()
    wrenchstamped.header.frame_id = frame_id
    wrenchstamped.header.stamp = rospy.Time(0)
    wrenchstamped.wrench.force.x = numpyarray[0]
    wrenchstamped.wrench.force.y = numpyarray[1]
    wrenchstamped.wrench.force.z = numpyarray[2]
    
    wrenchstamped.wrench.torque.x = numpyarray[3]
    wrenchstamped.wrench.torque.y = numpyarray[4]
    wrenchstamped.wrench.torque.z = numpyarray[5]
    
    return wrenchstamped


def get_ft_avg():
    wrench_sum = np.zeros(6)
    for i in tqdm(range(100)):
        wrench = get_wrench()
        wrench = wrenchstamped_to_numpy(wrench)
        wrench_sum += wrench
        rospy.sleep(0.05)
    wrench_bias = wrench_sum / 100.0
    return wrench_bias
    
def store_params(force_ext, pos_fext, wrench_bias, dir_conf='config/gravity_params.yaml'):
    gravity_cfg = {
        "force_ext": force_ext,
        "pos_fext": pos_fext,
        "wrench_bias": wrench_bias
    }
    dump_cfg(dir_conf, gravity_cfg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Gamma_on_hand_gravity_parameters_acquisition')
    parser.add_argument('--file_name', default='gravity_params.yaml', type=str)
    args = parser.parse_args()
    file_name = args.file_name
    
    package_path = os.path.dirname(os.path.realpath(__file__)).split('mmint_wrench_utils')[0]
    config_dir = os.path.join(package_path, 'mmint_wrench_utils/config', file_name)
    rospy.init_node('gamma_on_hand_gravity_parameters_acquisition')
    med = BubbleMed()
    
    
    # facedown hand (fx,fy = 0, tauz = 0)
    med.plan_to_joint_config(med.arm_group, [0, 0, 0, -np.pi/2, 0.0, np.pi/2, 0])
    rospy.sleep(1.0)
    wrench_bias_upright = get_ft_avg()
    fxy_upright = wrench_bias_upright[:2].copy() # bias
    tauz_upright = np.array([wrench_bias_upright[5]]) # bias
    
    # horizontal hand x-axis up (parallel to ground) (fy, fz = 0, taux = 0)
    med.plan_to_joint_config(med.arm_group, [0, 0, 0, -np.pi/2, 0.0, 0, 0])
    rospy.sleep(1.0)
    wrench_bias_horizontal_xup = get_ft_avg()
    fz_horizontal_xup = np.array([wrench_bias_horizontal_xup[2]])
    taux_horizontal_xup = np.array([wrench_bias_horizontal_xup[3]])

    # horizontal hand y-axis down (fx, fz = 0, tauy = 0)
    med.plan_to_joint_config(med.arm_group, [0, 0, 0, -np.pi/2, 0.0, 0.0, np.pi/2])
    rospy.sleep(1.0)
    wrench_bias_horizontal_ydown = get_ft_avg()
    tauy_horizontal_ydown = np.array([wrench_bias_horizontal_ydown[5]])
    
    # upright hand
    med.plan_to_joint_config(med.arm_group, [0, 0, 0, -np.pi/2, 0.0, -np.pi/2, 0])
    rospy.sleep(1.0)
    
    wrench_bias = np.concatenate([fxy_upright, fz_horizontal_xup, 
                                  taux_horizontal_xup, tauy_horizontal_ydown, tauz_upright])
    
    
    wrench = wrenchstamped_to_numpy(get_wrench())
    wrench_zeroed = wrench - wrench_bias

    
    print("raw wrench:", wrench)
    print("zeroed wrench:", wrench_zeroed)
    print("bias wrench:", wrench_bias)
    
    # now get the CoM of the applied gravity force.

    
    # facedown hand (fx,fy = 0, tauz = 0)
    med.plan_to_joint_config(med.arm_group, [0, 0, 0, -np.pi/2, 0.0, np.pi/2, 0])
    rospy.sleep(1.0)
    wrench_upright_zeroed = get_ft_avg() - wrench_bias
    fz_upright = wrench_upright_zeroed[2]
    taux_upright = wrench_upright_zeroed[3]
    tauy_upright = wrench_upright_zeroed[4]
    
    rx = -tauy_upright / fz_upright
    ry =  taux_upright / fz_upright

    # wrench_ext (force ext due to gravity)
    wrench_ext_gamma_frame = numpy_to_wrenchstamped(wrench_upright_zeroed)
    wrench_ext_medbase_frame = med.tf2_listener.transform_to_frame(wrench_ext_gamma_frame, target_frame='med_base',
                                                                   timeout=rospy.Duration(nsecs=int(5e8)))
    print(wrench_ext_medbase_frame)
    wrench_ext_medbase_frame = wrenchstamped_to_numpy(wrench_ext_medbase_frame)
    
    # horizontal hand x-axis up (parallel to ground) (fy, fz = 0, taux = 0)
    med.plan_to_joint_config(med.arm_group, [0, 0, 0, -np.pi/2, 0.0, 0, 0])
    wrench_horizontal_xup = get_ft_avg() - wrench_bias
    fx_horizontal_xup = wrench_horizontal_xup[0]
    tauy_horizontal_xup = wrench_horizontal_xup[4]
    
    rz = tauy_horizontal_xup / fx_horizontal_xup
    
    radius = np.array([rx, ry, rz]).tolist()
    fext = wrench_ext_medbase_frame[:3].tolist()
    wrench_bias = wrench_bias.tolist()
    
    store_params(fext, radius, wrench_bias, dir_conf=config_dir)