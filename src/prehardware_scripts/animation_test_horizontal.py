import sys
sys.path.append("..")
from load.sim_setup import load_iiwa_setup
from pydrake.geometry import StartMeshcat
from pydrake.multibody.plant import MultibodyPlant, MultibodyPlantConfig, AddMultibodyPlant
from pydrake.systems.framework import DiagramBuilder
from pydrake.math import RigidTransform
from pydrake.visualization import AddDefaultVisualization
from manipulation.scenarios import AddMultibodyTriad
from pydrake.all import Quaternion
import numpy as np
from planning.ik_util import solve_ik_inhand, pause_for, inhand_se2_poses, piecewise_joints, run_full_inhand_og, piecewise_traj, inhand_rotate_poses

JOINT0   = [1.0702422097407691, 0.79111135304063, 0.039522481390182704, -0.47337899137126993, -0.029476186840982563, 1.8773559661476429, 1.0891375237383238,
            -0.6243724965777308, 1.8539706319471008, -1.419344148470764, -0.9229579763233258, 1.7124576303632164, -1.8588769537333005, 1.5895425219089256]
GAP = 0.04
def inhand_test(desired_obj2left_se2: np.ndarray, left_pose0: RigidTransform, right_pose0: RigidTransform, object_pose0: RigidTransform, se2_time = 10.0):
    left_poses = [left_pose0]
    right_poses = [right_pose0]
    obj_poses = [object_pose0]
    ts = [0.0]
    
    rotation = -np.pi/4
    rotate_steps = 30
    rotate_time  = 30.0
    ts, left_poses, right_poses, obj_poses = inhand_rotate_poses(rotation, object_pose0, ts, left_poses, right_poses, obj_poses, steps=rotate_steps, rotate_time=rotate_time)
    ts, left_poses, right_poses, obj_poses = pause_for(1.0, ts, left_poses, right_poses, obj_poses)
    ts, left_poses, right_poses, obj_poses = inhand_se2_poses(np.array([0,0,np.pi + np.pi/2]), ts, left_poses, right_poses, obj_poses, left=True, se2_time=se2_time)
    ts, left_poses, right_poses, obj_poses = inhand_se2_poses(np.array([0.03,0,0]), ts, left_poses, right_poses, obj_poses, left=False, se2_time=se2_time)
    ts, left_poses, right_poses, obj_poses = inhand_se2_poses(np.array([0,0,np.pi + np.pi/2]), ts, left_poses, right_poses, obj_poses, left=True, se2_time=se2_time)
    ts, left_poses, right_poses, obj_poses = pause_for(2.0, ts, left_poses, right_poses, obj_poses)
    
    return ts, left_poses, right_poses, obj_poses
    
def generate_traj():
    plant_arms = MultibodyPlant(1e-3) # time step
    load_iiwa_setup(plant_arms, package_file='../../package.xml', directive_path="../../config/bimanual_med_gamma.yaml")
    plant_arms.Finalize()
    
    plant_context = plant_arms.CreateDefaultContext()
    plant_arms.SetPositions(plant_context, plant_arms.GetModelInstanceByName("iiwa_thanos"), JOINT0[:7])
    plant_arms.SetPositions(plant_context, plant_arms.GetModelInstanceByName("iiwa_medusa"), JOINT0[7:14])


    left_pose0 = plant_arms.GetFrameByName("thanos_finger").CalcPoseInWorld(plant_context)
    right_pose0 = plant_arms.GetFrameByName("medusa_finger").CalcPoseInWorld(plant_context)
    gap = GAP
    object_pose0 = RigidTransform(left_pose0.rotation().ToQuaternion(), left_pose0.translation() + left_pose0.rotation().matrix() @ np.array([0,0,gap/2.0]))
    
    desired_obj2left_se2 = np.array([0.00, 0.03, 0.0])
    ts, left_poses, right_poses, obj_poses = inhand_test(desired_obj2left_se2, left_pose0, right_pose0, object_pose0)
    left_piecewise, right_piecewise, _ = piecewise_traj(ts, left_poses, right_poses, obj_poses)
    
    T = ts[-1]
    
    
    ts = np.linspace(0, T, 100)
    seed_q0 = JOINT0
    qs = solve_ik_inhand(plant_arms, ts, left_piecewise, right_piecewise, "thanos_finger", "medusa_finger", seed_q0)
    q_piecewise = piecewise_joints(ts, qs)
    
    return ts, q_piecewise
    

if __name__ == '__main__':
    meshcat = StartMeshcat()
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlant(MultibodyPlantConfig(), builder)
    load_iiwa_setup(plant, scene_graph, package_file='../../package.xml', directive_path="../../config/bimanual_med.yaml")
    plant.Finalize()
    
    AddDefaultVisualization(builder, meshcat)
    AddMultibodyTriad(plant.GetFrameByName("iiwa_link_7", plant.GetModelInstanceByName("iiwa_thanos")), scene_graph, length=0.1, radius=0.003)
    AddMultibodyTriad(plant.GetFrameByName("iiwa_link_7", plant.GetModelInstanceByName("iiwa_medusa")), scene_graph, length=0.1, radius=0.003)
    
    AddMultibodyTriad(plant.GetFrameByName("thanos_finger"), scene_graph, length=0.05, radius=0.003)
    AddMultibodyTriad(plant.GetFrameByName("medusa_finger"), scene_graph, length=0.05, radius=0.003)
    
    diagram = builder.Build()
    
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    
    ts, q_piecewise = generate_traj()
    
    meshcat.StartRecording()
    diagram.ForcedPublish(context)
    for t in ts:
        context.SetTime(t)
        
        q = q_piecewise.value(t)
        
        plant.SetPositions(plant_context, plant.GetModelInstanceByName("iiwa_thanos"), q[:7])
        plant.SetPositions(plant_context, plant.GetModelInstanceByName("iiwa_medusa"), q[7:14])
        
        diagram.ForcedPublish(context)
    meshcat.StopRecording()
    meshcat.PublishRecording()
    print("Done")
    input()