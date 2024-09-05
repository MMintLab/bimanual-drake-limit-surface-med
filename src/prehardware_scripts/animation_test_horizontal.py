import sys
sys.path.append("..")
from load.sim_setup import load_iiwa_setup
from pydrake.geometry import StartMeshcat
from pydrake.multibody.plant import MultibodyPlant, MultibodyPlantConfig, AddMultibodyPlant
from pydrake.systems.framework import DiagramBuilder
from pydrake.math import RigidTransform
from pydrake.visualization import AddDefaultVisualization
from manipulation.scenarios import AddMultibodyTriad
from pydrake.all import Quaternion, PiecewisePose
import numpy as np
from planning.ik_util import solve_ik_inhand, pause_for, inhand_se2_poses, piecewise_joints, inhand_rotate_arms, inhand_se2_arms

JOINT0   = [1.0702422097407691, 0.79111135304063, 0.039522481390182704, -0.47337899137126993, -0.029476186840982563, 1.8773559661476429, 1.0891375237383238,
            -0.6243724965777308, 1.8539706319471008, -1.419344148470764, -0.9229579763233258, 1.7124576303632164, -1.8588769537333005, 1.5895425219089256]

def inhand_test(left_pose0: RigidTransform, right_pose0: RigidTransform, current_obj2medusa_se2 = np.array([0.01,0.01,np.pi/4])):
    ts, left_poses, right_poses = inhand_rotate_arms(left_pose0, right_pose0, current_obj2medusa_se2, rotation=np.pi/4, rotate_time = 30.0)
    
    return ts, left_poses, right_poses
def inhand_test_se2(left_pose0: RigidTransform, right_pose0: RigidTransform, current_obj2arm_se2 = np.array([0.00,0.0,0]), desired_obj2arm_se2 = np.array([0.00,0.0,0]),medusa = True, se2_time = 10.0):
    left_pose, right_pose = inhand_se2_arms(left_pose0, right_pose0, current_obj2arm_se2, desired_obj2arm_se2, medusa=medusa)
    
    left_poses = [left_pose0, left_pose]
    right_poses = [right_pose0, right_pose]
    ts = [0, se2_time]
    
    return ts, left_poses, right_poses

def generate_traj():
    plant_arms = MultibodyPlant(1e-3) # time step
    load_iiwa_setup(plant_arms, package_file='../../package.xml', directive_path="../../config/bimanual_med_gamma.yaml")
    plant_arms.Finalize()
    
    plant_context = plant_arms.CreateDefaultContext()
    plant_arms.SetPositions(plant_context, plant_arms.GetModelInstanceByName("iiwa_thanos"), JOINT0[:7])
    plant_arms.SetPositions(plant_context, plant_arms.GetModelInstanceByName("iiwa_medusa"), JOINT0[7:14])


    left_pose0 = plant_arms.GetFrameByName("thanos_finger").CalcPoseInWorld(plant_context)
    right_pose0 = plant_arms.GetFrameByName("medusa_finger").CalcPoseInWorld(plant_context)
    
    # ts, left_poses, right_poses = inhand_test(left_pose0, right_pose0)
    ts, left_poses, right_poses = inhand_test_se2(left_pose0, right_pose0, current_obj2arm_se2 = np.array([0.00,0.00,0.0]), desired_obj2arm_se2 = np.array([0.00,0.03,np.pi]), medusa=False)
    left_piecewise = PiecewisePose.MakeLinear(ts, left_poses)
    right_piecewise = PiecewisePose.MakeLinear(ts, right_poses)
    
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