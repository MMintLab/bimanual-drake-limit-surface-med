import numpy as np

import sys
sys.path.append('..')
from run_plan_main import goto_joints, curr_joints
from test_impedance import goto_and_torque

JOINT_CONFIG0 = [0.2629627380321955, 0.6651758641535246, 0.7157398241465858, -1.9204422808541055, 2.1896118717866164, 0.8246912707270445, -1.3312565995665175, -1.4523627309516682, 0.7165811053720673, 0.805679937637571, -1.876561512010483, 0.6976893656839942, -1.3458728960727322, 0.7420561347553449]
JOINT0_THANOS = np.array([JOINT_CONFIG0[:7]]).flatten()
JOINT0_MEDUSA = np.array([JOINT_CONFIG0[7:14]]).flatten()

if __name__ == '__main__':
    curr_q = curr_joints()
    des_q = JOINT_CONFIG0
    
    curr_q_thanos = curr_q[:7]
    curr_q_medusa = curr_q[7:14]
    
    joint_speed = 5.0 * np.pi / 180.0 # 1 degree per second
    thanos_displacement = np.max(np.abs(des_q[:7] - curr_q[:7]))
    thanos_endtime = thanos_displacement / joint_speed
    
    medusa_displacement = np.max(np.abs(des_q[7:14] - curr_q[7:14]))
    medusa_endtime = medusa_displacement / joint_speed
    
    print("medusa_endtime: ", medusa_endtime)
    print("thanos_endtime: ", thanos_endtime)
    des_q_thanos = JOINT0_THANOS.copy()
    des_q_medusa = JOINT0_MEDUSA.copy()
    
    input("Press Enter to reset medusa arm.")
    goto_joints(curr_q_thanos, des_q_medusa, endtime = medusa_endtime)
    input("Press Enter to reset thanos arm.")
    goto_joints(des_q_thanos, des_q_medusa, endtime = thanos_endtime)
    
    input("Press Enter to press fingers together")
    wrench_medusa = np.array([0, 0, 0, 0, -30.0, 0.0])
    wrench_thanos = np.array([0, 0, 0, 0, 30.0, 0.0])
    
    goto_and_torque(des_q[:7], des_q[7:], wrench_thanos, wrench_medusa)