import numpy as np

from pydrake.all import (
    Quaternion
)
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
from planning.ik_util import solve_ik_inhand, piecewise_joints, run_full_inhand, piecewise_traj, run_full_inhand_og
from planning.drake_inhand_planner2 import DualLimitSurfaceParams, inhand_planner
from load.finger_lib import AddSingleFinger
from load.shape_lib import AddBox

JOINT_CONFIG0 = [ 0.29795828,  0.45783705,  0.54191084, -2.00647729,  2.12890974,  0.93841575, -1.12574845,  0.19033173,  1.14214006,  1.63679035,  1.95986071, -2.56029611, -0.2526229,   2.1651646 ]
GAP = 0.01

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
    
    load_iiwa_setup(plant, scene_graph, package_file='../../package.xml', directive_path="../../config/bimanual_med_gamma.yaml")
    length = 0.01
    thanos_plate = AddSingleFinger(plant, radius=0.19/2.0, length=length, name="thanos_plate", mass=1.0, mu=1.0, color=[1,0,0,0.1])
    medusa_plate = AddSingleFinger(plant, radius=0.19/2.0, length=length, name="medusa_plate", mass=1.0, mu=1.0, color=[1,0,0,0.1])
        
    plant.WeldFrames(plant.GetFrameByName("thanos_finger"), plant.GetFrameByName("thanos_plate"), RigidTransform(np.array([0,0,-length/2])))
    plant.WeldFrames(plant.GetFrameByName("medusa_finger"), plant.GetFrameByName("medusa_plate"), RigidTransform(np.array([0,0,-length/2])))
    box = AddBox(plant, "object", (0.04,0.04,0.01), mass=1.0, mu = 1.0, color=[1,0,0,1])
    plant.Finalize()
    
    plant_arms = MultibodyPlant(1e-3) # time step
    load_iiwa_setup(plant_arms, package_file='../../package.xml', directive_path="../../config/bimanual_med_gamma.yaml")
    plant_arms.Finalize()
    
    
    AddDefaultVisualization(builder, meshcat)
    AddMultibodyTriad(plant.GetFrameByName("thanos_finger"), scene_graph, length=0.05, radius=0.003)
    AddMultibodyTriad(plant.GetFrameByName("medusa_finger"), scene_graph, length=0.05, radius=0.003)
    AddMultibodyTriad(plant.GetFrameByName("object_body", box), scene_graph, length=0.05, radius=0.003)
    
    diagram = builder.Build()
    
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    
    plant.SetPositions(plant_context, plant.GetModelInstanceByName("iiwa_thanos"), JOINT_CONFIG0[:7])
    plant.SetPositions(plant_context, plant.GetModelInstanceByName("iiwa_medusa"), JOINT_CONFIG0[7:14])
    
    # get pose
    left_pose0 = plant.GetFrameByName("thanos_finger").CalcPoseInWorld(plant_context)
    right_pose0 = plant.GetFrameByName("medusa_finger").CalcPoseInWorld(plant_context)
    gap = GAP

    # make object gap/2 away from left_pose0 in the z direction of left end-effector
    object_pose0 = RigidTransform(left_pose0.rotation().ToQuaternion(), left_pose0.translation() + left_pose0.rotation().matrix() @ np.array([0,0,gap/2.0]))
    

    current_obj2left_se2 = np.array([0.0, 0.0, 0.0])
    current_obj2right_se2 = np.array([0.0, 0.0, np.pi])
    
    desired_obj2left_se2 = np.array([0.00, 0.02, 0.0])
    desired_obj2right_se2 = np.array([0.00, -0.02, np.pi])
    
    
    dls_params = DualLimitSurfaceParams(mu_A = 0.75, r_A = 0.04, N_A = 20.0, mu_B = 0.75, r_B = 0.04, N_B = 20.0)
    horizon = 7
    obj2left, obj2right, vs = inhand_planner(current_obj2left_se2, current_obj2right_se2, desired_obj2left_se2, desired_obj2right_se2, dls_params, steps = horizon, angle = 60, palm_radius=0.04, kv = 20.0)
    
    print(np.round(desired_obj2left_se2 - obj2left[:,-1],4))
    print(np.round(desired_obj2right_se2 - obj2right[:,-1],4))
    
    print(np.round(obj2left,4))
    print(np.round(obj2right,4))
    
    desired_obj2left_se2s = []
    desired_obj2right_se2s = []
    for i in range(1,horizon):
        if i % 2 == 1:
            desired_obj2left_se2s.append(obj2left[:,i])
        else:
            desired_obj2right_se2s.append(obj2right[:,i] * np.array([-1,1,-1]))
        
    print(np.round(desired_obj2left_se2s,4))
    print(np.round(desired_obj2right_se2s,4))
    # print(np.round(obj2left,4))
    # print(np.round(obj2right,4))
    # input("Stop")
    
    # desired_obj2left_se2s = [np.array([0.0, 0.03, 0.0]), np.array([0.0, 0.00, 0.0])]
    # desired_obj2right_se2 = [np.array([0.00, -0.03, np.pi]), np.array([0.00, 0.00, np.pi])]
    
    # ts, left_poses, right_poses, obj_poses = run_full_inhand_og(desired_obj2left_se2, desired_obj2right_se2, left_pose0, right_pose0, object_pose0, rotation=70 * np.pi/180, rotate_steps=40, rotate_time=10.0, se2_time=10.0, back_time=10.0, fix_right=False)
    ts, left_poses, right_poses, obj_poses = run_full_inhand(desired_obj2left_se2s, desired_obj2right_se2s, left_pose0, right_pose0, object_pose0, rotation= 60 * np.pi/180, rotate_steps=40, rotate_time=10.0, se2_time=10.0, back_time=10.0, fix_right=False)
    left_piecewise, right_piecewise, object_piecewise = piecewise_traj(ts, left_poses, right_poses, obj_poses)
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
        obj_pos = object_piecewise.get_position_trajectory().value(t)
        obj_ori = object_piecewise.get_orientation_trajectory().value(t)
        
        plant.SetPositions(plant_context, plant.GetModelInstanceByName("iiwa_thanos"), q[:7])
        plant.SetPositions(plant_context, plant.GetModelInstanceByName("iiwa_medusa"), q[7:14])
        plant.SetFreeBodyPose(plant_context, plant.GetBodyByName("object_body", box), RigidTransform(Quaternion(obj_ori), obj_pos))
        diagram.ForcedPublish(context)
    meshcat.StopRecording()
    meshcat.PublishRecording()
    print("Done")
    input()