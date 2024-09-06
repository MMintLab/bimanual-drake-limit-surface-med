#!/usr/bin/env python
import numpy as np

import rospy
from geometry_msgs.msg import WrenchStamped

import sys
sys.path.append('..')
from hardware_kuka.gamma import wrenchstamped2numpy, GammaManager
from hardware_kuka.movement_lib import goto_joints
from mmint_utils.config import dump_cfg
from hardware_kuka.bimanual_kuka import BimanualKuka

def get_wrench_thanos():
    return rospy.wait_for_message('/netft_thanos/netft_data', WrenchStamped)
def get_wrench_medusa():
    return rospy.wait_for_message('/netft_medusa/netft_data', WrenchStamped)

def get_ft_avg(get_wrench_fn, samples = 100):
    wrench_sum = np.zeros(6)
    for i in range(samples):
        wrench = get_wrench_fn()
        wrench_sum += wrench
        rospy.sleep(0.05)
    wrench_bias = wrench_sum / float(samples)
    return wrench_bias

def store_params(force_ext, pos_fext, wrench_bias, dir_conf='config/gravity_params.yaml'):
    gravity_cfg = {
        "force_ext": force_ext,
        "pos_fext": pos_fext,
        "wrench_bias": wrench_bias
    }
    dump_cfg(dir_conf, gravity_cfg)

if __name__ == '__main__':
    rospy.init_node("gravity_param_calib")
    
    gamma_manager = GammaManager()
    bimanual_kuka = BimanualKuka()
    
    print("Home the robot")
    #zero joints first
    bimanual_kuka.goto_joints(np.zeros(7), np.zeros(7), endtime = 10.0)
    rospy.sleep(1.0)
    
    print("Face down the hands")
    # facedown hand (fx, fy = 0, tauz = 0)
    q_facedown = np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, 0])
    bimanual_kuka.goto_joints(q_facedown, q_facedown, endtime = 10.0)
    rospy.sleep(1.0)
    medusa_wrench_avg  = get_ft_avg(gamma_manager.get_medusa_wrench, samples=100)
    thanos_wrench_avg = get_ft_avg(gamma_manager.get_thanos_wrench, samples=100)
    
    medusa_fxy = medusa_wrench_avg[3:5].copy() # bias
    medusa_tauz = medusa_wrench_avg[2] # bias
    
    thanos_fxy = thanos_wrench_avg[3:5].copy() # bias
    thanos_tauz = thanos_wrench_avg[2] # bias
    
    
    print("Horizontal hand x-axis up")
    # horizontal hand x-axis up (parallel to ground) (fy, fz = 0, taux = 0)
    q_horizontal = np.array([0, 0, 0, -np.pi/2, 0, 0, 0])
    bimanual_kuka.goto_joints(q_horizontal, q_horizontal, endtime = 10.0)
    rospy.sleep(1.0)
    medusa_wrench_avg  = get_ft_avg(gamma_manager.get_medusa_wrench, samples=100)
    thanos_wrench_avg = get_ft_avg(gamma_manager.get_thanos_wrench, samples=100)
    
    medusa_fz = medusa_wrench_avg[5] # bias
    medusa_taux = medusa_wrench_avg[0] # bias
    
    thanos_fz = thanos_wrench_avg[5] # bias
    thanos_taux = thanos_wrench_avg[0] # bias
    
    print("Horizontal hand y-axis down")
    # horizontal hand y-axis down (fx, fz = 0, tauy = 0)
    q_up = np.array([0, 0, 0, -np.pi/2, 0, 0, np.pi/2])
    bimanual_kuka.goto_joints(q_up, q_up, endtime = 10.0)
    rospy.sleep(1.0)
    medusa_wrench_avg = get_ft_avg(gamma_manager.get_medusa_wrench, samples=100)
    thanos_wrench_avg = get_ft_avg(gamma_manager.get_thanos_wrench, samples=100)
    
    medusa_tauy = medusa_wrench_avg[1] # bias
    
    thanos_tauy = thanos_wrench_avg[1] # bias
    
    print("Calculated biases")
    medusa_bias = np.zeros(6)
    medusa_bias[0] = medusa_taux
    medusa_bias[1] = medusa_tauy
    medusa_bias[2] = medusa_tauz
    medusa_bias[3:5] = medusa_fxy
    medusa_bias[5] = medusa_fz
    
    thanos_bias = np.zeros(6)
    thanos_bias[0] = thanos_taux
    thanos_bias[1] = thanos_tauy
    thanos_bias[2] = thanos_tauz
    thanos_bias[3:5] = thanos_fxy
    thanos_bias[5] = thanos_fz
    
    
    # get CoM of applied gravity force
    
    print("Getting gravity wrench")
    # facedown
    bimanual_kuka.goto_joints(q_facedown, q_facedown, endtime = 10.0)
    rospy.sleep(1.0)
    medusa_wrench_avg = get_ft_avg(gamma_manager.get_medusa_wrench, samples=100)
    thanos_wrench_avg = get_ft_avg(gamma_manager.get_thanos_wrench, samples=100)
    
    medusa_wrench_zeroed = medusa_wrench_avg - medusa_bias
    thanos_wrench_zeroed = thanos_wrench_avg - thanos_bias
    
    medusa_taux, medusa_tauy, medusa_fz = medusa_wrench_zeroed[0], medusa_wrench_zeroed[1], medusa_wrench_zeroed[5]
    thanos_taux, thanos_tauy, thanos_fz = thanos_wrench_zeroed[0], thanos_wrench_zeroed[1], thanos_wrench_zeroed[5]
    
    medusa_rx = -medusa_tauy / medusa_fz
    medusa_ry = medusa_taux / medusa_fz
    
    thanos_rx = -thanos_tauy / thanos_fz
    thanos_ry = thanos_taux / thanos_fz
    
    # get wrench external (due to gravity)
    medusa_wrench_gravity = -medusa_wrench_zeroed[3:]
    thanos_wrench_gravity = -thanos_wrench_zeroed[3:]
    
    print("Getting CoM of applied gravity force")
    # horizontal x-axis up
    bimanual_kuka.goto_joints(q_horizontal, q_horizontal, endtime= 10.0)
    medusa_wrench_avg = get_ft_avg(gamma_manager.get_medusa_wrench, samples=100)
    thanos_wrench_avg = get_ft_avg(gamma_manager.get_thanos_wrench, samples=100)
    
    medusa_wrench_zeroed = medusa_wrench_avg - medusa_bias
    thanos_wrench_zeroed = thanos_wrench_avg - thanos_bias
    
    medusa_fx, medusa_tauy = medusa_wrench_zeroed[3], medusa_wrench_zeroed[1]
    thanos_fx, thanos_tauy = thanos_wrench_zeroed[3], thanos_wrench_zeroed[1]
    
    medusa_rz = medusa_tauy / medusa_fx
    thanos_rz = thanos_tauy / thanos_fx
    
    medusa_CoM = np.array([medusa_rx, medusa_ry, medusa_rz])
    thanos_CoM = np.array([thanos_rx, thanos_ry, thanos_rz])
    
    store_params(medusa_wrench_gravity.tolist(), medusa_CoM.tolist(), medusa_bias.tolist(), dir_conf='../config/medusa_gravity_params.yaml')
    store_params(thanos_wrench_gravity.tolist(), thanos_CoM.tolist(), thanos_bias.tolist(), dir_conf='../config/thanos_gravity_params.yaml')
