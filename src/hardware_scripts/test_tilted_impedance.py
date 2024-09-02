import numpy as np

from pydrake.all import (
    StartMeshcat,
    Simulator,
    DiagramBuilder,
    TrajectorySource,
    Multiplexer,
    Demultiplexer,
    RigidTransform,
    MultibodyPlant,
    DiagramBuilder,
    PiecewisePolynomial,
    JacobianWrtVariable
)
import numpy as np

import sys
sys.path.append('..')
from diagrams import create_hardware_diagram_plant_bimanual, create_visual_diagram
from planning.ik_util import solve_ik_inhand, run_full_inhand_og, piecewise_traj
from data_record import BenchmarkController
from test_impedance import goto_and_torque
from planning.object_compensation import ApplyForce
from load.sim_setup import load_iiwa_setup
from run_plan_main import curr_joints, goto_joints
from planning.ik_util import generate_push_configuration, inhand_test
from planning.hardware_util import follow_trajectory_and_torque
from enum import Enum

class ProgramState(Enum):
    HORIZONTAL = 0
    TILT_30 = 1
    MEGA_TILTED = 2

program_state = ProgramState.HORIZONTAL
if program_state == ProgramState.HORIZONTAL:
    JOINT_CONFIG0 = [1.1429203100700507, 0.7961505418611234, -0.2334648267973291, -0.4733848043491367, -2.967059728388293, -1.8813130837959888, -2.109669307516976, -1.5677529803998187, 1.9624159075892007, -1.4980034238420221, 0.9229619082335976, -1.9288236774214953, 1.7484457277131111, -2.5145907690545455]
elif program_state == ProgramState.TILT_30:
    #NOTE: tilt is 30 degrees
    raise NotImplementedError("Tilt 30 not implemented")
elif program_state == ProgramState.MEGA_TILTED:
    raise NotImplementedError("Mega tilted not implemented")
else:
    raise ValueError("Invalid program state")

JOINT0_THANOS = np.array([JOINT_CONFIG0[:7]]).flatten()
JOINT0_MEDUSA = np.array([JOINT_CONFIG0[7:14]]).flatten()

def generate_se2_motion(q0, desired_obj2left_se2 = np.array([0.00, 0.03, 0.0]), desired_obj2right_se2 = None, gap = 0.1):
    plant_arms = MultibodyPlant(1e-3) # time step
    load_iiwa_setup(plant_arms, package_file='../../package.xml', directive_path="../../config/bimanual_med_gamma.yaml")
    plant_arms.Finalize()
    
    plant_context = plant_arms.CreateDefaultContext()
    plant_arms.SetPositions(plant_context, plant_arms.GetModelInstanceByName("iiwa_thanos"), q0[:7])
    plant_arms.SetPositions(plant_context, plant_arms.GetModelInstanceByName("iiwa_medusa"), q0[7:14])
    
    thanos_pose = plant_arms.GetFrameByName("thanos_finger").CalcPoseInWorld(plant_context)
    medusa_pose = plant_arms.GetFrameByName("medusa_finger").CalcPoseInWorld(plant_context)
    
    object_pose0 = RigidTransform(thanos_pose.rotation().ToQuaternion(), thanos_pose.translation() + thanos_pose.rotation().matrix() @ np.array([0,0,gap/2.0]))
    
    ts, left_poses, right_poses, obj_poses = inhand_test(desired_obj2left_se2, thanos_pose, medusa_pose, object_pose0, desired_obj2right_se2=desired_obj2right_se2)
    left_piecewise, right_piecewise, _ = piecewise_traj(ts, left_poses, right_poses, obj_poses)
    T = ts[-1]
    
    
    ts = np.linspace(0, T, 100)
    qs = solve_ik_inhand(plant_arms, ts, left_piecewise, right_piecewise, "thanos_finger", "medusa_finger", q0)
    thanos_piecewise = PiecewisePolynomial.FirstOrderHold(ts, qs[:,:7].T)
    medusa_piecewise = PiecewisePolynomial.FirstOrderHold(ts, qs[:,7:].T)
    
    return thanos_piecewise, medusa_piecewise, T

if __name__ == '__main__':
    scenario_file = "../../config/bimanual_med_hardware_gamma.yaml"
    directives_file = "../../config/bimanual_med_gamma.yaml"
    
    plant_arms = MultibodyPlant(1e-3) # time step
    load_iiwa_setup(plant_arms, package_file='../../package.xml', directive_path=directives_file)
    plant_arms.Finalize()
    
    
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
    
    input("Press Enter to setup medusa arm.")
    goto_joints(curr_q_thanos, des_q_medusa, endtime = medusa_endtime, scenario_file=scenario_file, directives_file=directives_file)
    input("Press Enter to setup thanos arm.")
    goto_joints(des_q_thanos, des_q_medusa, endtime = thanos_endtime, scenario_file=scenario_file, directives_file=directives_file)
    
    input("Press Enter to get fingers closer")
    gap = 0.001
    des_q = generate_push_configuration(plant_arms, JOINT_CONFIG0, gap=gap)
    des_q_medusa = des_q[7:]
    des_q_thanos = des_q[:7]
    goto_joints(des_q_thanos, des_q_medusa, endtime = 1.0, scenario_file=scenario_file, directives_file=directives_file)
    
    curr_q = curr_joints()
    plant_context = plant_arms.CreateDefaultContext()
    plant_arms.SetPositions(plant_context, plant_arms.GetModelInstanceByName("iiwa_thanos"), des_q[:7])
    plant_arms.SetPositions(plant_context, plant_arms.GetModelInstanceByName("iiwa_medusa"), des_q[7:14])
    thanos_pose = plant_arms.GetFrameByName("thanos_finger").CalcPoseInWorld(plant_context)
    medusa_pose = plant_arms.GetFrameByName("medusa_finger").CalcPoseInWorld(plant_context)
    
    input("Press Enter to start pushing")
    object_kg = 0.5
    grav = 9.81
    force = 10.0
    wrench_thanos = thanos_pose.rotation().matrix() @ np.array([0, 0.0, force])
    wrench_medusa = medusa_pose.rotation().matrix() @ np.array([0, 0.0, force + object_kg * grav])
    wrench_thanos = np.concatenate([np.zeros(3), wrench_thanos])
    wrench_medusa = np.concatenate([np.zeros(3), wrench_medusa])
    goto_and_torque(des_q[:7], des_q[7:], endtime = 5.0, wrench_thanos=wrench_thanos, wrench_medusa=wrench_medusa, scenario_file=scenario_file, directives_file=directives_file)
    
    
    #make medusa go in se(2) motion
    traj_thanos, traj_medusa, T = generate_se2_motion(des_q, desired_obj2left_se2 = np.array([0.0, -0.04, 0.0]), gap = gap)
    input("Press Enter to start se2 motion")
    follow_trajectory_and_torque(traj_thanos, traj_medusa, force = force, endtime = T)