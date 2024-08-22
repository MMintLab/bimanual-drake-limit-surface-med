from pydrake.all import (
    StartMeshcat,
    DiagramBuilder,
    RollPitchYaw,
    AddDefaultVisualization,
    RotationMatrix,
    RigidTransform
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
from run_plan_main import curr_joints
from pydrake.visualization import AddDefaultVisualization

JOINT_CONFIG0 = [0.08232356364776336, 0.49329539590471605, 0.7554412443584381, -2.0426179181360524, 2.0754790345007996, 0.8874891667572512, -1.1673120760704268,
                 -1.4536369838514789, 0.5612986824682098, 0.8971038307962235, -2.003297518161298, 0.8415437358419539, -1.392097329426083, 0.7279235421513163]

if __name__ == '__main__':
    curr_q = curr_joints()
    
    print("Starting meshcat server...")
    meshcat = StartMeshcat()
    config = MultibodyPlantConfig()
    builder = DiagramBuilder()
    plant_arms, scene_graph = AddMultibodyPlant(config, builder)
    load_iiwa_setup(plant_arms, package_file='../../package.xml', directive_path="../../config/bimanual_med.yaml")
    
    length = 0.01
    thanos_plate = AddSingleFinger(plant_arms, radius=0.19/2.0, length=length, name="thanos_plate", mass=1.0, mu=1.0, color=[1,0,0,1])
    medusa_plate = AddSingleFinger(plant_arms, radius=0.19/2.0, length=length, name="medusa_plate", mass=1.0, mu=1.0, color=[1,0,0,1])
    
    
    plant_arms.WeldFrames(plant_arms.GetFrameByName("thanos_finger"), plant_arms.GetFrameByName("thanos_plate"), RigidTransform(np.array([0,0,0.2 - length/2 + 0.035])))
    plant_arms.WeldFrames(plant_arms.GetFrameByName("medusa_finger"), plant_arms.GetFrameByName("medusa_plate"), RigidTransform(np.array([0,0,0.2 - length/2 + 0.035])))
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
    
    left_pose: RigidTransform = plant_arms.EvalBodyPoseInWorld(plant_context, plant_arms.GetBodyByName("iiwa_link_7", plant_arms.GetModelInstanceByName("iiwa_thanos")))
    right_pose: RigidTransform = plant_arms.EvalBodyPoseInWorld(plant_context, plant_arms.GetBodyByName("iiwa_link_7", plant_arms.GetModelInstanceByName("iiwa_medusa")))
    
    #NOTE: kukas are rated at sub-mm repeatability
    #NOTE: end-effector distance is 0.4757
    #NOTE: total joint error is about 0.229 vs. 0.1077 (0.229 is because of end-effector)
    # DESIRED is 0.475, so it has sub-mm error.
    print(left_pose.translation())
    print("End-Effector Distance:", np.linalg.norm(left_pose.translation() - right_pose.translation()))
    print("End-Effector Distance:", np.linalg.norm(left_pose.translation()[1] - right_pose.translation()[1]))
    print("Max Thanos Joint Error:", np.max(np.abs(curr_q[:7] - JOINT_CONFIG0[:7])) * 180 / np.pi)
    print("Max Medusa Joint Error:", np.max(np.abs(curr_q[7:14] - JOINT_CONFIG0[7:14])) * 180 / np.pi)
    print("Sum Thanos Joint Error:", np.sum(np.abs(curr_q[:7] - JOINT_CONFIG0[:7])) * 180 / np.pi)
    print("Sum Medusa Joint Error:", np.sum(np.abs(curr_q[7:14] - JOINT_CONFIG0[7:14])) * 180 / np.pi )
    print("Avg Thanos Joint Error:", np.mean(np.abs(curr_q[:7] - JOINT_CONFIG0[:7])) * 180 / np.pi)
    print("Avg Medusa Joint Error:", np.mean(np.abs(curr_q[7:14] - JOINT_CONFIG0[7:14])) * 180 / np.pi)
    
    rotdiff = right_pose.rotation().matrix().T @ left_pose.rotation().matrix()
    print(rotdiff)
    print(RotationMatrix.MakeYRotation(np.pi).matrix())
    input()