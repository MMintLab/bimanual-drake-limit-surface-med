from pydrake.all import (
    RigidTransform,
    MultibodyPlant,
    PiecewisePolynomial
)
import numpy as np

import sys
sys.path.append('..')
from planning.ik_util import solve_ik_inhand, inhand_test, piecewise_traj
from load.sim_setup import load_iiwa_setup
from run_plan_main import curr_joints, goto_joints, follow_trajectory

JOINT_CONFIG0   = [0.8057998785688361, 0.843834930898186, 0.8817375343174186, -0.6412127704101249, 2.5038478567698452, -1.8218295137005034, -1.838969954216997,
                   -1.4647284816944872, 1.7587974000473647, 1.0286391175807714, -0.829148254439125, -1.3322830063140652, -2.0943951023869016, 0.64738334805492]

def generate_traj(q0, desired_obj2left_se2 = np.array([0.00, 0.03, 0.0]), gap = 0.475):
    plant_arms = MultibodyPlant(1e-3) # time step
    load_iiwa_setup(plant_arms, package_file='../../package.xml', directive_path="../../config/bimanual_med.yaml")
    plant_arms.Finalize()
    
    plant_context = plant_arms.CreateDefaultContext()
    plant_arms.SetPositions(plant_context, plant_arms.GetModelInstanceByName("iiwa_thanos"), q0[:7])
    plant_arms.SetPositions(plant_context, plant_arms.GetModelInstanceByName("iiwa_medusa"), q0[7:14])
    
    left_pose0 = plant_arms.EvalBodyPoseInWorld(plant_context, plant_arms.GetBodyByName("iiwa_link_7", plant_arms.GetModelInstanceByName("iiwa_thanos")))
    right_pose0 = plant_arms.EvalBodyPoseInWorld(plant_context, plant_arms.GetBodyByName("iiwa_link_7", plant_arms.GetModelInstanceByName("iiwa_medusa")))
    object_pose0 = RigidTransform(left_pose0.rotation().ToQuaternion(), left_pose0.translation() + left_pose0.rotation().matrix() @ np.array([0,0,gap/2.0]))
    
    ts, left_poses, right_poses, obj_poses = inhand_test(desired_obj2left_se2, left_pose0, right_pose0, object_pose0)
    left_piecewise, right_piecewise, _ = piecewise_traj(ts, left_poses, right_poses, obj_poses)
    
    T = ts[-1]
    
    
    ts = np.linspace(0, T, 100)
    qs = solve_ik_inhand(plant_arms, ts, left_piecewise, right_piecewise, "thanos_finger", "medusa_finger", q0)
    thanos_piecewise = PiecewisePolynomial.FirstOrderHold(ts, qs[:,:7].T)
    medusa_piecewise = PiecewisePolynomial.FirstOrderHold(ts, qs[:,7:].T)
    
    return ts, thanos_piecewise, medusa_piecewise

if __name__ == '__main__':
    GAP = 0.475    
    
    curr_q = curr_joints()
    des_q = JOINT_CONFIG0
    curr_q_thanos = curr_q[:7]
    curr_q_medusa = curr_q[7:14]
    des_q_thanos = des_q[:7]
    des_q_medusa = des_q[7:14]
    
    input("Press Enter to reset medusa arm.")
    goto_joints(curr_q_thanos, des_q_medusa, endtime = 60.0)
    input("Press Enter to reset thanos arm.")
    goto_joints(des_q_thanos, des_q_medusa, endtime = 60.0)
    
    curr_q = curr_joints()
    curr_thanos = curr_q[:7]
    curr_medusa = curr_q[7:14]
    if np.max(np.abs(curr_thanos - des_q_thanos)) > 1e-3:
        raise ValueError(f"Error: {np.linalg.norm(curr_thanos - des_q_thanos)}, Initial joint configuration for Thanos is not correct. WRONG SENSING")
    if np.max(np.abs(curr_medusa - des_q_medusa)) > 1e-3:
        raise ValueError(f"Error: {np.linalg.norm(curr_medusa - des_q_medusa)}, Initial joint configuration for Medusa is not correct. WRONG SENSING")
    
    ts, thanos_traj, medusa_traj = generate_traj(des_q)
    
    # check if current joint configuration is close to the desired joint traj at time = 0
    desired_start_thanos = thanos_traj.value(0)
    desired_start_medusa = medusa_traj.value(0)
    if np.linalg.norm(curr_thanos - desired_start_thanos) > 1e-3:
        raise ValueError(f"Error: {np.linalg.norm(curr_thanos - desired_start_thanos)}, Initial joint configuration for Thanos is not correct. BAD TRAJECTORY")
    if np.linalg.norm(curr_medusa - desired_start_medusa) > 1e-3:
        raise ValueError(f"Error: {np.linalg.norm(curr_medusa - desired_start_medusa)}, Initial joint configuration for Medusa is not correct. BAD TRAJECTORY")
    
    input("Press Enter to start in-hand manipulation.")
    follow_trajectory(thanos_traj, medusa_traj, ts[-1])
    
    pass