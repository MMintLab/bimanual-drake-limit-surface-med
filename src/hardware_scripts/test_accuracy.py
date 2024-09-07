from pydrake.all import (
    StartMeshcat,
    DiagramBuilder,
    RollPitchYaw,
    AddDefaultVisualization,
    RotationMatrix,
    RigidTransform,
    MultibodyPlant
)
from pydrake.multibody.plant import MultibodyPlantConfig, AddMultibodyPlant
from pydrake.systems.framework import DiagramBuilder
from pydrake.visualization import AddDefaultVisualization
from manipulation.scenarios import AddMultibodyTriad
import numpy as np


import sys
sys.path.append('..')
from load.sim_setup import load_iiwa_setup
from load.finger_lib import AddSingleFinger
from run_plan_main import curr_joints, curr_des_joints
from pydrake.visualization import AddDefaultVisualization


if __name__ == '__main__':
    curr_q = curr_joints()
    JOINT_CONFIG0 = curr_des_joints()
    
    print("Starting meshcat server...")
    meshcat = StartMeshcat()
    config = MultibodyPlantConfig()
    builder = DiagramBuilder()
    plant_arms, scene_graph = AddMultibodyPlant(config, builder)
    plant_arms: MultibodyPlant = plant_arms
    load_iiwa_setup(plant_arms, package_file='../../package.xml', directive_path="../../config/bimanual_med_gamma.yaml")
    
    length = 0.01
    thanos_plate = AddSingleFinger(plant_arms, radius=0.19/2.0, length=length, name="thanos_plate", mass=1.0, mu=1.0, color=[1,0,0,1])
    medusa_plate = AddSingleFinger(plant_arms, radius=0.19/2.0, length=length, name="medusa_plate", mass=1.0, mu=1.0, color=[1,0,0,1])
    
    plant_arms.Finalize()
    
    AddDefaultVisualization(builder, meshcat)
    AddMultibodyTriad(plant_arms.GetFrameByName("thanos_finger"), scene_graph, length=0.3, radius=0.003)
    AddMultibodyTriad(plant_arms.GetFrameByName("medusa_finger"), scene_graph, length=0.3, radius=0.003)
    
    diagram = builder.Build()
    
    context = diagram.CreateDefaultContext()
    plant_context = plant_arms.GetMyContextFromRoot(context)
    
    plant_arms.SetPositions(plant_context, plant_arms.GetModelInstanceByName("iiwa_thanos"), curr_q[:7])
    plant_arms.SetPositions(plant_context, plant_arms.GetModelInstanceByName("iiwa_medusa"), curr_q[7:14])
    
    diagram.ForcedPublish(context)
    
    left_pose = plant_arms.GetFrameByName("thanos_finger").CalcPoseInWorld(plant_context)
    right_pose = plant_arms.GetFrameByName("medusa_finger").CalcPoseInWorld(plant_context)
    
    #get desired pose
    plant_arms.SetPositions(plant_context, plant_arms.GetModelInstanceByName("iiwa_thanos"), JOINT_CONFIG0[:7])
    plant_arms.SetPositions(plant_context, plant_arms.GetModelInstanceByName("iiwa_medusa"), JOINT_CONFIG0[7:14])
    
    desired_left_pose = plant_arms.GetFrameByName("thanos_finger").CalcPoseInWorld(plant_context)
    desired_right_pose = plant_arms.GetFrameByName("medusa_finger").CalcPoseInWorld(plant_context)
    
    #NOTE: kukas are rated at sub-mm repeatability
    #NOTE: end-effector distance is 0.4757
    #NOTE: total joint error is about 0.229 vs. 0.1077 (0.229 is because of end-effector)
    # DESIRED is 0.475, so it has sub-mm error.
    print("End-Effector Distance:", np.linalg.norm(left_pose.translation() - right_pose.translation()))
    print("Max Thanos Joint Error:", np.max(np.abs(curr_q[:7] - JOINT_CONFIG0[:7])) * 180 / np.pi)
    print("Max Medusa Joint Error:", np.max(np.abs(curr_q[7:14] - JOINT_CONFIG0[7:14])) * 180 / np.pi)
    print("Sum Thanos Joint Error:", np.sum(np.abs(curr_q[:7] - JOINT_CONFIG0[:7])) * 180 / np.pi)
    print("Sum Medusa Joint Error:", np.sum(np.abs(curr_q[7:14] - JOINT_CONFIG0[7:14])) * 180 / np.pi )
    print("Avg Thanos Joint Error:", np.mean(np.abs(curr_q[:7] - JOINT_CONFIG0[:7])) * 180 / np.pi)
    print("Avg Medusa Joint Error:", np.mean(np.abs(curr_q[7:14] - JOINT_CONFIG0[7:14])) * 180 / np.pi)
    print("Left Pose Error:", np.linalg.norm(left_pose.translation() - desired_left_pose.translation()))
    print("Right Pose Error:", np.linalg.norm(right_pose.translation() - desired_right_pose.translation()))
    
    print("Left Pose Error Vector:", left_pose.translation() - desired_left_pose.translation())
    print("Right Pose Error Vector:", right_pose.translation() - desired_right_pose.translation())
    
    print("Medusa RPY:", right_pose.rotation().ToRollPitchYaw().vector() * 180 / np.pi)
    print("Thanos RPY:", left_pose.rotation().ToRollPitchYaw().vector() * 180 / np.pi)
    input()