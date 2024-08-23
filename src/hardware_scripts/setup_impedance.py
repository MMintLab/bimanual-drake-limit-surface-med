from pydrake.all import (
    StartMeshcat,
    Simulator,
    DiagramBuilder,
    TrajectorySource,
    PiecewisePolynomial,
    RollPitchYaw,
    ConstantVectorSource,
    RotationMatrix,
    Multiplexer
)
from pydrake.multibody.plant import MultibodyPlant, MultibodyPlantConfig, AddMultibodyPlant
from pydrake.systems.framework import DiagramBuilder
from pydrake.math import RigidTransform

from manipulation.scenarios import AddMultibodyTriad
from pydrake.all import Quaternion
import numpy as np

import numpy as np

import sys
sys.path.append('..')
from planning.ik_util import solve_ik_inhand, piecewise_joints, run_full_inhand_og, piecewise_traj, solveDualIK
from diagrams import create_hardware_diagram_plant_bimanual, create_visual_diagram
from load.sim_setup import load_iiwa_setup
from data_record import BenchmarkController
from run_plan_main import goto_joints, curr_joints
from test_impedance import Wrench2Torque

JOINT_CONFIG0 = [0.2712481521271895, 0.6698470022075304, 0.7149546560362198, -1.9117409107694736, 2.1937025889464716, 0.8194818446316442, -1.3333036678254957, -1.4537717803723942, 0.7208004236134968, 0.8031074680671298, -1.8675310720442153, 0.6936791316731125, -1.3391527674342723, 0.7394731700254528]
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