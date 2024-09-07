from pydrake.all import (
    StartMeshcat,
    Simulator,
    DiagramBuilder,
    TrajectorySource,
    PiecewisePolynomial,
    RollPitchYaw,
    ConstantVectorSource,
    RotationMatrix,
    Multiplexer
)
from pydrake.multibody.plant import MultibodyPlant, MultibodyPlantConfig, AddMultibodyPlant
from pydrake.systems.framework import DiagramBuilder
from pydrake.math import RigidTransform

from manipulation.scenarios import AddMultibodyTriad
from pydrake.all import Quaternion
import numpy as np

import numpy as np

import sys
sys.path.append('..')
from planning.ik_util import solve_ik_inhand, piecewise_joints, run_full_inhand_og, piecewise_traj, solveDualIK
from diagrams import create_hardware_diagram_plant_bimanual, create_visual_diagram
from load.sim_setup import load_iiwa_setup
from data_record import BenchmarkController

JOINT_CONFIG0 = [0.11467720408573778, 0.5848234311893432, 0.7214010101336089, -2.0606035482880722, 2.1018508757506353, 0.9204573806512731, -1.285284657896468, -1.4020361847132508, 0.6555956355828122, 0.8483553662375546, -2.03859736848514, 0.7688058141268922, -1.498929485466734, 0.8033779305302288]
JOINT0_THANOS = np.array([JOINT_CONFIG0[:7]]).flatten()
JOINT0_MEDUSA = np.array([JOINT_CONFIG0[7:14]]).flatten()

def curr_des_joints():
    scenario_file = "../../config/bimanual_med_hardware_gamma.yaml"
    hardware_diagram, hardware_plant = create_hardware_diagram_plant_bimanual(scenario_filepath=scenario_file, meshcat=None, position_only=True)
    context = hardware_diagram.CreateDefaultContext()
    hardware_diagram.ExecuteInitializationEvents(context)
    curr_q_medusa = hardware_diagram.GetOutputPort("iiwa_medusa.position_commanded").Eval(context)
    curr_q_thanos = hardware_diagram.GetOutputPort("iiwa_thanos.position_commanded").Eval(context)
    curr_q = np.concatenate([curr_q_thanos, curr_q_medusa])
    return curr_q

def curr_joints():
    scenario_file = "../../config/bimanual_med_hardware_gamma.yaml"
    hardware_diagram, hardware_plant = create_hardware_diagram_plant_bimanual(scenario_filepath=scenario_file, meshcat=None, position_only=True)
    context = hardware_diagram.CreateDefaultContext()
    hardware_diagram.ExecuteInitializationEvents(context)
    curr_q_medusa = hardware_diagram.GetOutputPort("iiwa_medusa.position_measured").Eval(context)
    curr_q_thanos = hardware_diagram.GetOutputPort("iiwa_thanos.position_measured").Eval(context)
    curr_q = np.concatenate([curr_q_thanos, curr_q_medusa])
    return curr_q

#NOTE: medusa is off by 1 degree, until we fix the calibration, we need to correct the medusa arm
def correct_medusa():
    q = curr_joints()
    
    #######################################################################################################
    plant_arms = MultibodyPlant(1e-3) # time step
    load_iiwa_setup(plant_arms, package_file='../../package.xml', directive_path="../../config/bimanual_med.yaml")
    plant_arms.Finalize()

    plant_context = plant_arms.CreateDefaultContext()
    plant_arms.SetPositions(plant_context, plant_arms.GetModelInstanceByName("iiwa_thanos"), q[:7])
    plant_arms.SetPositions(plant_context, plant_arms.GetModelInstanceByName("iiwa_medusa"), q[7:14])
    #######################################################################################################
    
    thanos_pose = plant_arms.EvalBodyPoseInWorld(plant_context, plant_arms.GetBodyByName("iiwa_link_7", plant_arms.GetModelInstanceByName("iiwa_thanos")))
    medusa_pose = plant_arms.EvalBodyPoseInWorld(plant_context, plant_arms.GetBodyByName("iiwa_link_7", plant_arms.GetModelInstanceByName("iiwa_medusa")))
    
    x_radians = -1.0 * np.pi / 180.0
    new_medusa_pose = RigidTransform((medusa_pose.rotation() @ RotationMatrix.MakeXRotation(x_radians)).ToQuaternion(), medusa_pose.translation() + np.array([0,0,-0.003]))
    
    new_q, _ = solveDualIK(plant_arms, thanos_pose, new_medusa_pose, "thanos_finger", "medusa_finger", q0=q)
    new_q_thanos = new_q[:7]
    new_q_medusa = new_q[7:14]
    
    goto_joints(new_q_thanos, new_q_medusa, endtime = 15.0)
    

def goto_joints(joint_thanos, joint_medusa, endtime = 30.0, scenario_file = "../../config/bimanual_med_hardware.yaml", directives_file = "../../config/bimanual_med.yaml"):
    meshcat = StartMeshcat()
    
    root_builder = DiagramBuilder()
    
    hardware_diagram, hardware_plant = create_hardware_diagram_plant_bimanual(scenario_filepath=scenario_file, meshcat=meshcat, position_only=True)
    vis_diagram = create_visual_diagram(directives_filepath=directives_file, meshcat=meshcat, package_file="../../package.xml")
    
    hardware_block = root_builder.AddSystem(hardware_diagram)
    
    ## make a plan from current position to desired position
    context = hardware_diagram.CreateDefaultContext()
    hardware_diagram.ExecuteInitializationEvents(context)
    curr_q_medusa = hardware_diagram.GetOutputPort("iiwa_medusa.position_commanded").Eval(context)
    curr_q_thanos = hardware_diagram.GetOutputPort("iiwa_thanos.position_commanded").Eval(context)
    ts = np.array([0.0, endtime])
    
    qs_medusa = np.array([curr_q_medusa, joint_medusa])
    traj_medusa = PiecewisePolynomial.FirstOrderHold(ts, qs_medusa.T)
    
    qs_thanos = np.array([curr_q_thanos, joint_thanos])
    traj_thanos = PiecewisePolynomial.FirstOrderHold(ts, qs_thanos.T)
    
    traj_medusa_block = root_builder.AddSystem(TrajectorySource(traj_medusa))
    root_builder.Connect(traj_medusa_block.get_output_port(), hardware_block.GetInputPort("iiwa_medusa.position"))
    root_builder.Connect(traj_medusa_block.get_output_port(), hardware_block.GetInputPort("iiwa_medusa_fake.position"))
    
    traj_thanos_block = root_builder.AddSystem(TrajectorySource(traj_thanos))
    root_builder.Connect(traj_thanos_block.get_output_port(), hardware_block.GetInputPort("iiwa_thanos.position"))
    root_builder.Connect(traj_thanos_block.get_output_port(), hardware_block.GetInputPort("iiwa_thanos_fake.position"))
    
    root_diagram = root_builder.Build()
    
    # run simulation
    simulator = Simulator(root_diagram)
    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(endtime + 2.0)
    
def goto_joints_straight(joint_thanos, joint_medusa, endtime = 30.0):
    meshcat = StartMeshcat()
    scenario_file = "../../config/bimanual_med_hardware.yaml"
    directives_file = "../../config/bimanual_med.yaml"
    
    root_builder = DiagramBuilder()
    
    hardware_diagram, hardware_plant = create_hardware_diagram_plant_bimanual(scenario_filepath=scenario_file, meshcat=meshcat, position_only=True)
    vis_diagram = create_visual_diagram(directives_filepath=directives_file, meshcat=meshcat, package_file="../../package.xml")
    
    hardware_block = root_builder.AddSystem(hardware_diagram)
    
    traj_medusa_block = root_builder.AddSystem(ConstantVectorSource(joint_medusa))
    root_builder.Connect(traj_medusa_block.get_output_port(), hardware_block.GetInputPort("iiwa_medusa.position"))
    root_builder.Connect(traj_medusa_block.get_output_port(), hardware_block.GetInputPort("iiwa_medusa_fake.position"))
    
    traj_thanos_block = root_builder.AddSystem(ConstantVectorSource(joint_thanos))
    root_builder.Connect(traj_thanos_block.get_output_port(), hardware_block.GetInputPort("iiwa_thanos.position"))
    root_builder.Connect(traj_thanos_block.get_output_port(), hardware_block.GetInputPort("iiwa_thanos_fake.position"))
    
    root_diagram = root_builder.Build()
    
    # run simulation
    simulator = Simulator(root_diagram)
    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(endtime + 2.0)

def generate_trajectory(seed_q0, rotation=np.pi/3):
    plant_arms = MultibodyPlant(1e-3) # time step
    load_iiwa_setup(plant_arms, package_file='../../package.xml', directive_path="../../config/bimanual_med.yaml")
    plant_arms.Finalize()

    plant_context = plant_arms.CreateDefaultContext()
    plant_arms.SetPositions(plant_context, plant_arms.GetModelInstanceByName("iiwa_thanos"), seed_q0[:7])
    plant_arms.SetPositions(plant_context, plant_arms.GetModelInstanceByName("iiwa_medusa"), seed_q0[7:14])
    
    left_pose0 = plant_arms.EvalBodyPoseInWorld(plant_context, plant_arms.GetBodyByName("iiwa_link_7", plant_arms.GetModelInstanceByName("iiwa_thanos")))
    right_pose0 = plant_arms.EvalBodyPoseInWorld(plant_context, plant_arms.GetBodyByName("iiwa_link_7", plant_arms.GetModelInstanceByName("iiwa_medusa")))
    
    gap = GAP - PUSH_DISTANCE
    object_pose0 = RigidTransform(left_pose0.rotation().ToQuaternion(), left_pose0.translation() + left_pose0.rotation().matrix() @ np.array([0,0,(gap)/2.0]))

    desired_obj2left_se2 = np.array([0.00, 0.0, 0.0])
    desired_obj2right_se2 = np.array([0.00, 0.0, np.pi])

    ts, left_poses, right_poses, obj_poses = run_full_inhand_og(desired_obj2left_se2, desired_obj2right_se2, left_pose0, right_pose0, object_pose0, rotation=rotation, rotate_steps=1000, rotate_time=30.0, se2_time=20.0, back_time=5.0, fix_right=False)
    left_piecewise, right_piecewise, _ = piecewise_traj(ts, left_poses, right_poses, obj_poses)
    
    T = ts[-1]    
    ts = np.linspace(0, T, 1_000)
    qs = solve_ik_inhand(plant_arms, ts, left_piecewise, right_piecewise, "thanos_finger", "medusa_finger", seed_q0)
    qs_thanos = qs[:, :7]
    qs_medusa = qs[:, 7:14]
    
    
    traj_medusa = PiecewisePolynomial.FirstOrderHold(ts, qs_medusa.T)
    traj_thanos = PiecewisePolynomial.FirstOrderHold(ts, qs_thanos.T)
    return traj_medusa, traj_thanos, T

def generate_push_configuration(seed_q0, gap = 0.475, push_distance = 0.025):
    plant_arms = MultibodyPlant(1e-3) # time step
    load_iiwa_setup(plant_arms, package_file='../../package.xml', directive_path="../../config/bimanual_med.yaml")
    plant_arms.Finalize()
    
    plant_context = plant_arms.CreateDefaultContext()
    plant_arms.SetPositions(plant_context, plant_arms.GetModelInstanceByName("iiwa_thanos"), seed_q0[:7])
    plant_arms.SetPositions(plant_context, plant_arms.GetModelInstanceByName("iiwa_medusa"), seed_q0[7:14])
    
    thanos_pose = plant_arms.EvalBodyPoseInWorld(plant_context, plant_arms.GetBodyByName("iiwa_link_7", plant_arms.GetModelInstanceByName("iiwa_thanos")))
    medusa_pose = plant_arms.EvalBodyPoseInWorld(plant_context, plant_arms.GetBodyByName("iiwa_link_7", plant_arms.GetModelInstanceByName("iiwa_medusa")))
    
    new_thanos_pose = RigidTransform(thanos_pose.rotation().ToQuaternion(), thanos_pose.translation() + thanos_pose.rotation().matrix() @ np.array([0,0,push_distance/2]))
    
    new_medusa_rotation = thanos_pose.rotation() @ RollPitchYaw(0.0, np.pi, 0.0).ToRotationMatrix()
    new_medusa_pose = RigidTransform(new_medusa_rotation, thanos_pose.translation() - new_medusa_rotation.matrix() @ np.array([0,0, gap - push_distance]))

    new_joints,_ = solveDualIK(plant_arms, new_thanos_pose, new_medusa_pose, "thanos_finger", "medusa_finger", q0=seed_q0)
    
    return new_joints

def follow_trajectory(traj_thanos, traj_medusa, endtime = 1e12):
    meshcat = StartMeshcat()
    scenario_file = "../../config/bimanual_med_hardware.yaml"
    directives_file = "../../config/bimanual_med.yaml"
        
    root_builder = DiagramBuilder()
    
    hardware_diagram, hardware_plant = create_hardware_diagram_plant_bimanual(scenario_filepath=scenario_file, meshcat=meshcat, position_only=True)
    vis_diagram = create_visual_diagram(directives_filepath=directives_file, meshcat=meshcat, package_file="../../package.xml")
    
    hardware_block = root_builder.AddSystem(hardware_diagram)
    recorder = BenchmarkController(hardware_plant)
    recorder_block = root_builder.AddSystem(recorder)
    
    traj_medusa_block = root_builder.AddSystem(TrajectorySource(traj_medusa))
    root_builder.Connect(traj_medusa_block.get_output_port(), hardware_block.GetInputPort("iiwa_medusa.position"))
    root_builder.Connect(traj_medusa_block.get_output_port(), hardware_block.GetInputPort("iiwa_medusa_fake.position"))
    
    traj_thanos_block = root_builder.AddSystem(TrajectorySource(traj_thanos))
    root_builder.Connect(traj_thanos_block.get_output_port(), hardware_block.GetInputPort("iiwa_thanos.position"))
    root_builder.Connect(traj_thanos_block.get_output_port(), hardware_block.GetInputPort("iiwa_thanos_fake.position"))
    
    multiplexer_block_target = root_builder.AddSystem(Multiplexer([7,7]))
    root_builder.Connect(traj_thanos_block.get_output_port(), multiplexer_block_target.get_input_port(0))
    root_builder.Connect(traj_medusa_block.get_output_port(), multiplexer_block_target.get_input_port(1))
    root_builder.Connect(multiplexer_block_target.get_output_port(), recorder_block.GetInputPort("target"))
    
    multiplexer_block_measured = root_builder.AddSystem(Multiplexer([7,7]))
    root_builder.Connect(hardware_block.GetOutputPort("iiwa_thanos.position_measured"), multiplexer_block_measured.get_input_port(0))
    root_builder.Connect(hardware_block.GetOutputPort("iiwa_medusa.position_measured"), multiplexer_block_measured.get_input_port(1))
    root_builder.Connect(multiplexer_block_measured.get_output_port(), recorder_block.GetInputPort("measure"))
    
    root_diagram = root_builder.Build()

    simulator = Simulator(root_diagram)
    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(endtime + 5.0)
    
    return recorder

if __name__ == '__main__':
    GAP = 0.475
    PUSH_DISTANCE = 0.020
    
    curr_q = curr_joints()
    des_q = JOINT_CONFIG0
    
    curr_q_thanos = curr_q[:7]
    curr_q_medusa = curr_q[7:14]
    
    joint_speed = 3.0 * np.pi / 180.0 # 1 degree per second
    thanos_displacement = np.max(np.abs(des_q[:7] - curr_q[:7]))
    thanos_endtime = thanos_displacement / joint_speed
    
    medusa_displacement = np.max(np.abs(des_q[7:14] - curr_q[7:14]))
    medusa_endtime = medusa_displacement / joint_speed
    
    print("medusa_endtime: ", medusa_endtime)
    print("thanos_endtime: ", thanos_endtime)
    des_q_thanos = JOINT0_THANOS.copy()
    des_q_medusa = JOINT0_MEDUSA.copy()
    
    input("Press Enter to reset medusa arm.")
    goto_joints(curr_q_thanos, des_q_medusa, endtime = medusa_endtime)
    input("Press Enter to reset thanos arm.")
    goto_joints(des_q_thanos, des_q_medusa, endtime = thanos_endtime)
    input("Press Enter to correct arms.")
    goto_joints_straight(des_q_thanos, des_q_medusa, endtime = 10.0)
    # correct_medusa()
    
    # input("Press Enter to move end-effectors to a \"pushed\" position.")
    # # curr_q = curr_joints()
    # # des_q = generate_push_configuration(JOINT_CONFIG0, gap = GAP, push_distance = PUSH_DISTANCE)
    
    # des_q_thanos = des_q[:7]
    # des_q_medusa = des_q[7:14]
    # goto_joints(des_q_thanos, des_q_medusa, endtime = 15.0)
    # goto_joints_straight(des_q_thanos, des_q_medusa, endtime = 15.0)
    
    
    curr_q = curr_joints()
    curr_thanos = curr_q[:7]
    curr_medusa = curr_q[7:14]
    if np.max(np.abs(curr_thanos - des_q_thanos)) > 5e-3:
        raise ValueError(f"Error: {np.linalg.norm(curr_thanos - JOINT0_THANOS)}, Initial joint configuration for Thanos is not correct. WRONG SENSING")
    if np.max(np.abs(curr_medusa - des_q_medusa)) > 5e-3:
        raise ValueError(f"Error: {np.linalg.norm(curr_medusa - JOINT0_MEDUSA)}, Initial joint configuration for Medusa is not correct. WRONG SENSING")
    
    seed_q0 = des_q
    traj_medusa, traj_thanos, T = generate_trajectory(seed_q0)
    
    input("Press Enter to start the simulation.")
    recorder = follow_trajectory(traj_thanos, traj_medusa, T)
    
    import matplotlib.pyplot as plt
    #plot joint error over time
    ts = recorder.ts
    q_targets = np.array(recorder.q_targets)
    q_measures = np.array(recorder.q_measures)
    plt.figure()
    for i in range(14):
        plt.plot(ts, q_targets[:, i] - q_measures[:, i], label=f"q{i}")
    plt.legend()
    plt.show()
    