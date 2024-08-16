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
from planning.ik_util import solve_ik_inhand, pause_for, inhand_se2_poses, piecewise_joints, run_full_inhand_og, piecewise_traj

JOINT0   = [1.091627175380399, 0.7504935659013866, -0.03066493117212439, -0.641208649724199, 0.02125370004032286, 1.7500789182727032, 1.0729869938631913,
          -1.464729579371752, 1.758801739349052, 1.028649487365105, -0.8291452599791955, -1.3322962015581077, -2.0943951023905, 0.6473815288375726]
GAP = 0.475

def inhand_test(desired_obj2left_se2: np.ndarray, left_pose0: RigidTransform, right_pose0: RigidTransform, object_pose0: RigidTransform, se2_time = 10.0):
    left_poses = [left_pose0]
    right_poses = [right_pose0]
    obj_poses = [object_pose0]
    ts = [0.0]
    ts, left_poses, right_poses, obj_poses = pause_for(1.0, ts, left_poses, right_poses, obj_poses)
    ts, left_poses, right_poses, obj_poses = inhand_se2_poses(desired_obj2left_se2, ts, left_poses, right_poses, obj_poses, left=False, se2_time=se2_time)
    ts, left_poses, right_poses, obj_poses = pause_for(2.0, ts, left_poses, right_poses, obj_poses)
    
    return ts, left_poses, right_poses, obj_poses
    
def generate_traj():
    plant_arms = MultibodyPlant(1e-3) # time step
    load_iiwa_setup(plant_arms, package_file='../../package.xml', directive_path="../../config/bimanual_med.yaml")
    plant_arms.Finalize()
    
    plant_context = plant_arms.CreateDefaultContext()
    plant_arms.SetPositions(plant_context, plant_arms.GetModelInstanceByName("iiwa_thanos"), JOINT0[:7])
    plant_arms.SetPositions(plant_context, plant_arms.GetModelInstanceByName("iiwa_medusa"), JOINT0[7:14])
    
    left_pose0 = plant_arms.EvalBodyPoseInWorld(plant_context, plant_arms.GetBodyByName("iiwa_link_7", plant_arms.GetModelInstanceByName("iiwa_thanos")))
    right_pose0 = plant_arms.EvalBodyPoseInWorld(plant_context, plant_arms.GetBodyByName("iiwa_link_7", plant_arms.GetModelInstanceByName("iiwa_medusa")))
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
    AddMultibodyTriad(plant.GetFrameByName("thanos_finger"), scene_graph, length=0.3, radius=0.003)
    AddMultibodyTriad(plant.GetFrameByName("medusa_finger"), scene_graph, length=0.3, radius=0.003)
    
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