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
from planning.ik_util import solve_ik_inhand, piecewise_joints, run_full_inhand_og, piecewise_traj

JOINT_CONFIG0 = [-0.32823683178594826, 0.9467527057457398, 1.5375963846252783, -2.055496608537348, -0.8220809597822779, -0.31526250680171636, 1.3872151028590527,
                 -1.7901817338098867, 1.2653964889934661, 1.740960078785441, -2.014334314596287, 0.35305405885912783, -1.8242723561461582, -0.01502208888994321]

GAP = 0.475

if __name__ == '__main__':
    config = MultibodyPlantConfig()
    config.time_step = 1e-3
    config.penetration_allowance = 1e-3
    config.contact_model = "hydroelastic_with_fallback"
    config.contact_surface_representation = "polygon"
    config.discrete_contact_approximation = "tamsi"
    
    meshcat = StartMeshcat()
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlant(config, builder)
    
    load_iiwa_setup(plant, scene_graph, package_file='../package.xml', directive_path="../config/bimanual_med.yaml")
    plant.Finalize()
    
    plant_arms = MultibodyPlant(1e-3) # time step
    load_iiwa_setup(plant_arms, package_file='../package.xml', directive_path="../config/bimanual_med.yaml")
    plant_arms.Finalize()
    
    
    AddDefaultVisualization(builder, meshcat)
    AddMultibodyTriad(plant.GetFrameByName("thanos_finger"), scene_graph, length=0.3, radius=0.003)
    AddMultibodyTriad(plant.GetFrameByName("medusa_finger"), scene_graph, length=0.3, radius=0.003)
    
    diagram = builder.Build()
    
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    
    plant.SetPositions(plant_context, plant.GetModelInstanceByName("iiwa_thanos"), JOINT_CONFIG0[:7])
    plant.SetPositions(plant_context, plant.GetModelInstanceByName("iiwa_medusa"), JOINT_CONFIG0[7:14])
    
    # get pose
    left_pose0 = plant.EvalBodyPoseInWorld(plant_context, plant.GetBodyByName("iiwa_link_7", plant.GetModelInstanceByName("iiwa_thanos")))
    right_pose0 = plant.EvalBodyPoseInWorld(plant_context, plant.GetBodyByName("iiwa_link_7", plant.GetModelInstanceByName("iiwa_medusa")))
    gap = GAP

    # make object gap/2 away from left_pose0 in the z direction of left end-effector
    object_pose0 = RigidTransform(left_pose0.rotation().ToQuaternion(), left_pose0.translation() + left_pose0.rotation().matrix() @ np.array([0,0,gap/2.0]))
    

    # get panda config 
    
    desired_obj2left_se2 = np.array([0.0, -0.03, 0])
    desired_obj2right_se2 = np.array([0.0, -0.03, np.pi])
    
    ts, left_poses, right_poses, obj_poses = run_full_inhand_og(desired_obj2left_se2, desired_obj2right_se2, left_pose0, right_pose0, object_pose0, rotation=np.pi/3, rotate_steps=10, rotate_time=10.0, se2_time=10.0, back_time=10.0, fix_right=False)
    left_piecewise, right_piecewise, _ = piecewise_traj(ts, left_poses, right_poses, obj_poses)
    T = ts[-1]
    
    ts = np.linspace(0, T, 1_000)
    seed_q0 = JOINT_CONFIG0


    qs = solve_ik_inhand(plant_arms, ts, left_piecewise, right_piecewise, "thanos_finger", "medusa_finger", seed_q0)
    q_piecewise = piecewise_joints(ts, qs)
    
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