import sys
sys.path.append("..")
from planning.ik_util import solveDualIK
from pydrake.multibody.plant import MultibodyPlant
from pydrake.math import RotationMatrix, RollPitchYaw, RigidTransform
from pydrake.all import PiecewisePose, PiecewiseTrajectory, Quaternion, PiecewisePolynomial
import numpy as np
from typing import List

'''
    Ik Utils 2.0
    =============
    We will build a set of functions to calculate all of the desired 3d poses for bimanual robot.
    
    Then run ik to follow the 3d poses.
'''
def CalculateRotation(pose: RigidTransform, object_pose: RigidTransform, desired_rotation: float = 0.0) -> RigidTransform:
    '''
    NOTE: hardcode inhand rotation to be in y-axis
    '''
    
    Hobj2world = object_pose.GetAsMatrix4()
    Hee2world = pose.GetAsMatrix4()
    
    # get object->ee
    Hobj2ee = np.linalg.inv(Hee2world) @ Hobj2world
    
    # rotate object to get new object-> world
    rot = RotationMatrix.MakeYRotation(desired_rotation).matrix()
    Hobj2world[:3, :3] = Hobj2world[:3, :3] @ rot # rotate around its y-axis from its current frame
    
    # calculate new ee -> world based on rotated object-> world
    Hee2world = Hobj2world @ np.linalg.inv(Hobj2ee)
    
    return RigidTransform(Hee2world)

def get_se2_vector(pose: RigidTransform):
    # use only yaw angle
    yaw_angle = pose.rotation().ToRollPitchYaw().yaw_angle()
    # get only xy translation, drop z
    xy = pose.translation()[:2]
    return np.array([xy[0], xy[1], yaw_angle])

def get_se2_pose(se2_vector: np.array):
    rotz = RotationMatrix.MakeZRotation(se2_vector[2])
    return RigidTransform(rotz, [se2_vector[0], se2_vector[1], 0.0])

def inhand_rotate(left_pose: RigidTransform, right_pose: RigidTransform, object_pose: RigidTransform, rotation: float = 0.0, steps: int = 30):
    angles = np.linspace(0, rotation, steps)
    left_poses = [CalculateRotation(left_pose, object_pose, angle) for angle in angles]
    right_poses = [CalculateRotation(right_pose, object_pose, angle) for angle in angles]
    object_poses = [object_pose @ RigidTransform(RollPitchYaw(0.0, angle, 0.0), np.zeros(3)) for angle in angles]
    return left_poses, right_poses, object_poses

def inhand_se2(desired_se2: np.ndarray, left_pose: RigidTransform, right_pose: RigidTransform, object_pose: RigidTransform, left=True):
    # left => move the left hand
    # right => move the right hand
    
    # desired_se2 is the desired pose of the object in the other hand
    # left => desired_se2 is obj2right pose
    # right => desired_se2 is obj2left pose
    
    # get the current se2 of the object
    obj2world = object_pose
    left2world = left_pose
    right2world = right_pose
    
    still2world = right2world if left else left2world
    moving2world = left2world if left else right2world
    
    moving2obj = obj2world.inverse() @ moving2world
    
    # obj2moving stays constant
    # obj2still changes to desired_se2
    obj2still = still2world.inverse() @ obj2world
    obj2still_se2 = get_se2_vector(obj2still)
    obj2still_se2_pose = get_se2_pose(obj2still_se2)
    
    obj2still_new_se2_pose = get_se2_pose(desired_se2)
    obj2still_new = obj2still_new_se2_pose @ obj2still_se2_pose.inverse() @ obj2still
    
    obj2world_new = still2world @ obj2still_new
    
    moving2world_new = obj2world_new @ moving2obj
    
    return obj2world_new, moving2world_new, still2world

def run_full_inhand(desired_obj2left_se2s: List[np.ndarray], desired_obj2right_se2s: List[np.ndarray], left_pose0: RigidTransform, right_pose0: RigidTransform, object_pose0: RigidTransform, rotation: float = np.pi/3, fix_right=False, rotate_steps=10, rotate_time=10.0, se2_time = 10.0, back_time=10.0):
    left_poses = [left_pose0]
    right_poses = [right_pose0]
    obj_poses = [object_pose0]
    ts = [0.0]
    
    
    for desired_obj2left_se2, desired_obj2right_se2 in zip(desired_obj2left_se2s, desired_obj2right_se2s):
        ts, left_poses, right_poses, obj_poses = pause_for(1.0, ts, left_poses, right_poses, obj_poses)
        
        # rotate left
        ts, left_poses, right_poses, obj_poses = inhand_rotate_poses(rotation, object_pose0, ts, left_poses, right_poses, obj_poses, steps=rotate_steps, rotate_time=rotate_time)
        ts, left_poses, right_poses, obj_poses = pause_for(2.00, ts, left_poses, right_poses, obj_poses)
        
        # move left
        ts, left_poses, right_poses, obj_poses = inhand_se2_poses(desired_obj2left_se2, ts, left_poses, right_poses, obj_poses, left=False, se2_time=se2_time)
        ts, left_poses, right_poses, obj_poses = pause_for(2.0, ts, left_poses, right_poses, obj_poses)
        
        #rotate back
        ts, left_poses, right_poses, obj_poses = inhand_rotate_poses(-rotation, object_pose0, ts, left_poses, right_poses, obj_poses, steps=rotate_steps, rotate_time=rotate_time)
        ts, left_poses, right_poses, obj_poses = pause_for(2.0, ts, left_poses, right_poses, obj_poses)
        
        # move back (original object pose is the reference)
        ts, left_poses, right_poses, obj_poses = inhand_back_poses(ts, left_poses, right_poses, obj_poses, object_pose0, back_time=back_time)
        ts, left_poses, right_poses, obj_poses = pause_for(2.0, ts, left_poses, right_poses, obj_poses)
        
        # rotate right
        ts, left_poses, right_poses, obj_poses = inhand_rotate_poses(-rotation, object_pose0, ts, left_poses, right_poses, obj_poses, steps=rotate_steps, rotate_time=rotate_time)
        ts, left_poses, right_poses, obj_poses = pause_for(2.00, ts, left_poses, right_poses, obj_poses)
        
        # move right
        ts, left_poses, right_poses, obj_poses = inhand_se2_poses(desired_obj2right_se2, ts, left_poses, right_poses, obj_poses, left=True, se2_time=se2_time)
        ts, left_poses, right_poses, obj_poses = pause_for(2.0, ts, left_poses, right_poses, obj_poses)
        
        #rotate back
        ts, left_poses, right_poses, obj_poses = inhand_rotate_poses(rotation, object_pose0, ts, left_poses, right_poses, obj_poses, steps=rotate_steps, rotate_time=rotate_time)
        ts, left_poses, right_poses, obj_poses = pause_for(2.0, ts, left_poses, right_poses, obj_poses)
        
        # move back (original object pose is the reference)
        ts, left_poses, right_poses, obj_poses = inhand_back_poses(ts, left_poses, right_poses, obj_poses, object_pose0, back_time=back_time)
        ts, left_poses, right_poses, obj_poses = pause_for(2.0, ts, left_poses, right_poses, obj_poses)
    if fix_right:    
        # fix the right pose to be rotated by pi/2 on y-axis
        right_poses = [right_pose @ RigidTransform(RollPitchYaw(0.0,np.pi,0.0), np.zeros(3)) for right_pose in right_poses]
        
    return ts, left_poses, right_poses, obj_poses

def inhand_move_back(left2world: RigidTransform, right2world: RigidTransform, obj2world: RigidTransform, reference_pose: RigidTransform):
    #get left2obj and right2obj
    left2obj = obj2world.inverse() @ left2world
    right2obj = obj2world.inverse() @ right2world
    
    # get obj2reference
    obj2reference = reference_pose.inverse() @ obj2world
    
    # set z to 0
    obj2reference_new = RigidTransform(
        obj2reference.rotation(), 
        np.array([obj2reference.translation()[0], obj2reference.translation()[1], 0])
    )
    
    obj2world_new = reference_pose @ obj2reference_new
    left2world_new = obj2world_new @ left2obj
    right2world_new = obj2world_new @ right2obj
    
    return left2world_new, right2world_new, obj2world_new

def pause_for(duration: float, ts: List[float], left_poses: List[RigidTransform], right_poses: List[RigidTransform], obj_poses: List[RigidTransform]):
    left_poses.append(left_poses[-1])
    right_poses.append(right_poses[-1])
    obj_poses.append(obj_poses[-1])
    ts.append(ts[-1] + duration)
    return ts, left_poses, right_poses, obj_poses

def inhand_rotate_poses(rotation: float, reference_pose: RigidTransform, ts: List[float], left_poses: List[RigidTransform], right_poses: List[RigidTransform], obj_poses: List[RigidTransform], steps: int = 30, rotate_time = 0.05):
    tmp2world = RigidTransform(reference_pose.rotation(), obj_poses[-1].translation())
    obj2tmp = tmp2world.inverse() @ obj_poses[-1]
    
    rotated_left, rotated_right, rotated_object = inhand_rotate(left_poses[-1], right_poses[-1], tmp2world, rotation, steps=steps)
    
    # hacky way to rotate the object
    rotated_object = [RigidTransform(tmp2world.rotation() @ RotationMatrix.MakeYRotation(angle), tmp2world.translation()) @ obj2tmp for angle in np.linspace(0, rotation, steps)]
    
    rotated_ts = np.linspace(ts[-1] + 1e-4, ts[-1] + rotate_time, steps)
    
    left_poses = left_poses + rotated_left
    right_poses = right_poses + rotated_right
    obj_poses = obj_poses + rotated_object
    ts = ts + rotated_ts.tolist()
    
    
    return ts, left_poses, right_poses, obj_poses

def inhand_se2_poses(desired_se2, ts: List[float], left_poses: List[RigidTransform], right_poses: List[RigidTransform], obj_poses: List[RigidTransform], left=True, se2_time=0.1):
    obj2world_moved, moving2world_moved, still2world_moved = inhand_se2(desired_se2, left_poses[-1], right_poses[-1], obj_poses[-1], left=left)
    left2world_moved = moving2world_moved if left else still2world_moved
    right2world_moved = still2world_moved if left else moving2world_moved
    
    left_poses.append(left2world_moved)
    right_poses.append(right2world_moved)
    obj_poses.append(obj2world_moved)
    ts.append(ts[-1] + se2_time)
    return ts, left_poses, right_poses, obj_poses

def inhand_se2_dls_poses(desired_se2, ts: List[float], left_poses: List[RigidTransform], right_poses: List[RigidTransform], obj_poses: List[RigidTransform], left=True, se2_time=0.1, kv=0.1, inverse=False, radius=0.05):
    obj2world_moveds, moving2world_moveds, still2world_moveds = inhand_se2_dls(desired_se2, left_poses[-1], right_poses[-1], obj_poses[-1], left=left, kv=kv, inverse=inverse, radius=radius)
    left2world_moveds = moving2world_moveds if left else still2world_moveds
    right2world_moveds = still2world_moveds if left else moving2world_moveds
    
    left_poses = left_poses + left2world_moveds
    right_poses = right_poses + right2world_moveds
    obj_poses = obj_poses + obj2world_moveds
    ts = ts + [ts[-1]+1e-3 + se2_time*i for i in range(len(obj2world_moveds))]
    
    return ts, left_poses, right_poses, obj_poses

def inhand_back_poses(ts: List[float], left_poses: List[RigidTransform], right_poses: List[RigidTransform], obj_poses: List[RigidTransform], reference_pose: RigidTransform, back_time=0.01):
    left2world_back, right2world_back, obj2world_back = inhand_move_back(left_poses[-1], right_poses[-1], obj_poses[-1], reference_pose)
    
    left_poses = left_poses + [left2world_back]
    right_poses = right_poses + [right2world_back]
    obj_poses = obj_poses + [obj2world_back]
    ts.append(ts[-1] + back_time)
    return ts, left_poses, right_poses, obj_poses

def run_full_inhand_og(desired_obj2left_se2: np.ndarray, desired_obj2right_se2: np.ndarray, left_pose0: RigidTransform, right_pose0: RigidTransform, object_pose0: RigidTransform, rotation: float = np.pi/3):
    left_poses = [left_pose0]
    right_poses = [right_pose0]
    obj_poses = [object_pose0]
    ts = [0.0]
    ts, left_poses, right_poses, obj_poses = pause_for(0.05, ts, left_poses, right_poses, obj_poses)
    
    rotate_steps = 30
    rotate_time = 1.0
    se2_time = 1.0
    back_time = 1.0
    
    # rotate left
    ts, left_poses, right_poses, obj_poses = inhand_rotate_poses(-rotation, object_pose0, ts, left_poses, right_poses, obj_poses, steps=rotate_steps, rotate_time=rotate_time)
    ts, left_poses, right_poses, obj_poses = pause_for(0.05, ts, left_poses, right_poses, obj_poses)
    
    # move left
    ts, left_poses, right_poses, obj_poses = inhand_se2_poses(desired_obj2left_se2, ts, left_poses, right_poses, obj_poses, left=False, se2_time=se2_time)
    ts, left_poses, right_poses, obj_poses = pause_for(0.05, ts, left_poses, right_poses, obj_poses)
    
    #rotate back
    ts, left_poses, right_poses, obj_poses = inhand_rotate_poses(rotation, object_pose0, ts, left_poses, right_poses, obj_poses, steps=rotate_steps, rotate_time=rotate_time)
    ts, left_poses, right_poses, obj_poses = pause_for(0.05, ts, left_poses, right_poses, obj_poses)
    
    # move back (original object pose is the reference)
    ts, left_poses, right_poses, obj_poses = inhand_back_poses(ts, left_poses, right_poses, obj_poses, object_pose0, back_time=back_time)
    ts, left_poses, right_poses, obj_poses = pause_for(0.05, ts, left_poses, right_poses, obj_poses)
    
    # rotate right
    ts, left_poses, right_poses, obj_poses = inhand_rotate_poses(rotation, object_pose0, ts, left_poses, right_poses, obj_poses, steps=rotate_steps, rotate_time=rotate_time)
    ts, left_poses, right_poses, obj_poses = pause_for(0.05, ts, left_poses, right_poses, obj_poses)
    
    # move right
    ts, left_poses, right_poses, obj_poses = inhand_se2_poses(desired_obj2right_se2, ts, left_poses, right_poses, obj_poses, left=True, se2_time=se2_time)
    ts, left_poses, right_poses, obj_poses = pause_for(0.05, ts, left_poses, right_poses, obj_poses)
    
    #rotate back
    ts, left_poses, right_poses, obj_poses = inhand_rotate_poses(-rotation, object_pose0, ts, left_poses, right_poses, obj_poses, steps=rotate_steps, rotate_time=rotate_time)
    ts, left_poses, right_poses, obj_poses = pause_for(0.02, ts, left_poses, right_poses, obj_poses)
    
    # move back (original object pose is the reference)
    ts, left_poses, right_poses, obj_poses = inhand_back_poses(ts, left_poses, right_poses, obj_poses, object_pose0, back_time=back_time)
    ts, left_poses, right_poses, obj_poses = pause_for(0.02, ts, left_poses, right_poses, obj_poses)
    
    # fix the right pose to be rotated by pi/2 on y-axis
    right_poses = [right_pose @ RigidTransform(RollPitchYaw(0.0,np.pi,0.0), np.zeros(3)) for right_pose in right_poses]
    
    return ts, left_poses, right_poses, obj_poses

def run_full_inhand_simple_dls(desired_obj2left_se2: np.ndarray, desired_obj2right_se2: np.ndarray, left_pose0: RigidTransform, right_pose0: RigidTransform, object_pose0: RigidTransform, rotation: float = np.pi/3, kv=0.3, inverse=False, radius=0.05):
    left_poses = [left_pose0]
    right_poses = [right_pose0]
    obj_poses = [object_pose0]
    ts = [0.0]
    ts, left_poses, right_poses, obj_poses = pause_for(0.05, ts, left_poses, right_poses, obj_poses)
    
    rotate_steps = 30
    rotate_time = 1.0
    se2_time = 1.0
    back_time = 1.0
    
    # rotate left
    ts, left_poses, right_poses, obj_poses = inhand_rotate_poses(-rotation, object_pose0, ts, left_poses, right_poses, obj_poses, steps=rotate_steps, rotate_time=rotate_time)
    ts, left_poses, right_poses, obj_poses = pause_for(0.05, ts, left_poses, right_poses, obj_poses)
    
    # move left
    ts, left_poses, right_poses, obj_poses = inhand_se2_dls_poses(desired_obj2left_se2, ts, left_poses, right_poses, obj_poses, left=False, se2_time=se2_time, kv=kv, inverse=inverse, radius=radius)
    ts, left_poses, right_poses, obj_poses = pause_for(0.05, ts, left_poses, right_poses, obj_poses)
    
    #rotate back
    ts, left_poses, right_poses, obj_poses = inhand_rotate_poses(rotation, object_pose0, ts, left_poses, right_poses, obj_poses, steps=rotate_steps, rotate_time=rotate_time)
    ts, left_poses, right_poses, obj_poses = pause_for(0.05, ts, left_poses, right_poses, obj_poses)
    
    # move back (original object pose is the reference)
    ts, left_poses, right_poses, obj_poses = inhand_back_poses(ts, left_poses, right_poses, obj_poses, object_pose0, back_time=back_time)
    ts, left_poses, right_poses, obj_poses = pause_for(0.05, ts, left_poses, right_poses, obj_poses)
    
    # rotate right
    ts, left_poses, right_poses, obj_poses = inhand_rotate_poses(rotation, object_pose0, ts, left_poses, right_poses, obj_poses, steps=rotate_steps, rotate_time=rotate_time)
    ts, left_poses, right_poses, obj_poses = pause_for(0.05, ts, left_poses, right_poses, obj_poses)
    
    # move right
    ts, left_poses, right_poses, obj_poses = inhand_se2_poses(desired_obj2right_se2, ts, left_poses, right_poses, obj_poses, left=True, se2_time=se2_time)
    ts, left_poses, right_poses, obj_poses = pause_for(0.05, ts, left_poses, right_poses, obj_poses)
    
    #rotate back
    ts, left_poses, right_poses, obj_poses = inhand_rotate_poses(-rotation, object_pose0, ts, left_poses, right_poses, obj_poses, steps=rotate_steps, rotate_time=rotate_time)
    ts, left_poses, right_poses, obj_poses = pause_for(0.02, ts, left_poses, right_poses, obj_poses)
    
    # move back (original object pose is the reference)
    ts, left_poses, right_poses, obj_poses = inhand_back_poses(ts, left_poses, right_poses, obj_poses, object_pose0, back_time=back_time)
    ts, left_poses, right_poses, obj_poses = pause_for(0.02, ts, left_poses, right_poses, obj_poses)
    
    # fix the right pose to be rotated by pi/2 on y-axis
    right_poses = [right_pose @ RigidTransform(RollPitchYaw(0.0,np.pi,0.0), np.zeros(3)) for right_pose in right_poses]
    
    return ts, left_poses, right_poses, obj_poses

def run_full_inhand_asymmetric_dls_notilt(desired_obj2left_se2: np.ndarray, desired_obj2right_se2: np.ndarray, left_pose0: RigidTransform, right_pose0: RigidTransform, object_pose0: RigidTransform, kv=0.3, kv_inv=5.0, radius=0.05):
    left_poses = [left_pose0]
    right_poses = [right_pose0]
    obj_poses = [object_pose0]
    ts = [0.0]
    ts, left_poses, right_poses, obj_poses = pause_for(0.05, ts, left_poses, right_poses, obj_poses)
    
    se2_time = 1.0
    # move left
    ts, left_poses, right_poses, obj_poses = inhand_se2_dls_poses(desired_obj2left_se2, ts, left_poses, right_poses, obj_poses, left=False, se2_time=se2_time, kv=kv, inverse=False, radius=radius)
    ts, left_poses, right_poses, obj_poses = pause_for(0.05, ts, left_poses, right_poses, obj_poses)
    
    # move right
    ts, left_poses, right_poses, obj_poses = inhand_se2_dls_poses(desired_obj2right_se2, ts, left_poses, right_poses, obj_poses, left=True, se2_time=se2_time, kv=kv_inv, inverse=True, radius=radius)
    ts, left_poses, right_poses, obj_poses = pause_for(0.05, ts, left_poses, right_poses, obj_poses)
    
    right_poses = [right_pose @ RigidTransform(RollPitchYaw(0.0,np.pi,0.0), np.zeros(3)) for right_pose in right_poses]
    return ts, left_poses, right_poses, obj_poses

def piecewise_traj(ts: List[float], left_poses: List[RigidTransform], right_poses: List[RigidTransform], object_poses: List[RigidTransform]):
    #first order hold piecewise
    left_piecewise =  PiecewisePose.MakeLinear(ts, left_poses)
    right_piecewise = PiecewisePose.MakeLinear(ts, right_poses)
    object_piecewise = PiecewisePose.MakeLinear(ts, object_poses)
    return left_piecewise, right_piecewise, object_piecewise

# solve IK
def solve_ik_inhand(plant: MultibodyPlant, ts: np.ndarray, left_piecewise: PiecewisePose, right_piecewise: PiecewisePose, left_frame_name: str, right_frame_name: str, q0 = 1e-10*np.ones(14)):
    #NOTE: make sure q0 is seeded well
    
    left_p_G = left_piecewise.get_position_trajectory()
    left_R_G = left_piecewise.get_orientation_trajectory()
    
    right_p_G = right_piecewise.get_position_trajectory()
    right_R_G = right_piecewise.get_orientation_trajectory()
    
    qs = []
    
    curr_q = q0.copy()
    for t in ts:
        left_pose = RigidTransform(Quaternion(left_R_G.value(t)), left_p_G.value(t))
        right_pose = RigidTransform(Quaternion(right_R_G.value(t)), right_p_G.value(t))
        q, success = solveDualIK(plant, left_pose, right_pose, left_frame_name=left_frame_name, right_frame_name=right_frame_name, q0=curr_q)
        # if not success:
        #     raise ValueError("IK failed")
        qs.append(q)
        curr_q = q
    return np.array(qs)

def piecewise_joints(ts: List[float], qs: List[np.ndarray]) -> PiecewiseTrajectory:
    #first order hold piecewise
    return PiecewisePolynomial.FirstOrderHold(ts, qs.T)