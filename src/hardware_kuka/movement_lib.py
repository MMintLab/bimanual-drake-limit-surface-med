import numpy as np

from pydrake.all import (
    StartMeshcat,
    Simulator,
    DiagramBuilder,
    TrajectorySource,
    DiagramBuilder,
    PiecewisePolynomial,
    MultibodyPlant,
    RigidTransform,
    RollPitchYaw,
    Demultiplexer,
    ConstantVectorSource,
    Multiplexer
)
import numpy as np

import sys
sys.path.append('..')
from diagrams import create_hardware_diagram_plant_bimanual, create_visual_diagram
from planning.ik_util import solveDualIK, inhand_rotate_poses, pause_for, piecewise_traj, solve_ik_inhand
from bimanual_systems import Wrench2Torque
from planning.object_compensation import ApplyForce

def curr_joints(scenario_file = "../../config/bimanual_med_hardware.yaml"):
    hardware_diagram, hardware_plant = create_hardware_diagram_plant_bimanual(scenario_filepath=scenario_file, meshcat=None, position_only=True)
    context = hardware_diagram.CreateDefaultContext()
    hardware_diagram.ExecuteInitializationEvents(context)
    curr_q_medusa = hardware_diagram.GetOutputPort("iiwa_medusa.position_measured").Eval(context)
    curr_q_thanos = hardware_diagram.GetOutputPort("iiwa_thanos.position_measured").Eval(context)
    curr_q = np.concatenate([curr_q_thanos, curr_q_medusa])
    return curr_q

def curr_desired_joints(scenario_file = "../../config/bimanual_med_hardware.yaml"):
    hardware_diagram, hardware_plant = create_hardware_diagram_plant_bimanual(scenario_filepath=scenario_file, meshcat=None, position_only=True)
    context = hardware_diagram.CreateDefaultContext()
    hardware_diagram.ExecuteInitializationEvents(context)
    curr_q_medusa = hardware_diagram.GetOutputPort("iiwa_medusa.position_commanded").Eval(context)
    curr_q_thanos = hardware_diagram.GetOutputPort("iiwa_thanos.position_commanded").Eval(context)
    curr_q = np.concatenate([curr_q_thanos, curr_q_medusa])
    return curr_q


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
    
def direct_joint_torque(joint_thanos, joint_medusa, wrench_thanos, wrench_medusa, endtime = 30.0, scenario_file = "../../config/bimanual_med_hardware_impedance.yaml", directives_file = "../../config/bimanual_med.yaml"):
    meshcat = StartMeshcat()
    
    root_builder = DiagramBuilder()
    hardware_diagram, hardware_plant = create_hardware_diagram_plant_bimanual(scenario_filepath=scenario_file, position_only=False, meshcat = meshcat)
    vis_diagram = create_visual_diagram(directives_filepath=directives_file, package_file="../../package.xml", meshcat = meshcat)
    
    hardware_block = root_builder.AddSystem(hardware_diagram)
    traj_medusa_block = root_builder.AddSystem(ConstantVectorSource(joint_medusa))
    traj_thanos_block = root_builder.AddSystem(ConstantVectorSource(joint_thanos))
    
    thanos_wrench_block = root_builder.AddSystem(ConstantVectorSource(wrench_thanos))
    medusa_wrench_block = root_builder.AddSystem(ConstantVectorSource(wrench_medusa))
    
    torque_block = root_builder.AddSystem(Wrench2Torque(hardware_plant))
    
    torque_demultiplexer_block = root_builder.AddSystem(Demultiplexer(14, 7))

    root_builder.Connect(traj_thanos_block.get_output_port(), hardware_block.GetInputPort("iiwa_thanos.position"))
    root_builder.Connect(traj_thanos_block.get_output_port(), hardware_block.GetInputPort("iiwa_thanos_fake.position"))
    
    root_builder.Connect(traj_medusa_block.get_output_port(), hardware_block.GetInputPort("iiwa_medusa.position"))
    root_builder.Connect(traj_medusa_block.get_output_port(), hardware_block.GetInputPort("iiwa_medusa_fake.position"))

    root_builder.Connect(hardware_block.GetOutputPort("iiwa_thanos.position_measured"), torque_block.GetInputPort("thanos_position"))
    root_builder.Connect(hardware_block.GetOutputPort("iiwa_medusa.position_measured"), torque_block.GetInputPort("medusa_position"))

    root_builder.Connect(thanos_wrench_block.get_output_port(), torque_block.GetInputPort("wrench_thanos"))
    root_builder.Connect(medusa_wrench_block.get_output_port(), torque_block.GetInputPort("wrench_medusa"))
    
    root_builder.Connect(torque_block.GetOutputPort("torque"), torque_demultiplexer_block.get_input_port())
    
    root_builder.Connect(torque_demultiplexer_block.get_output_port(0), hardware_block.GetInputPort("iiwa_thanos.feedforward_torque"))
    root_builder.Connect(torque_demultiplexer_block.get_output_port(1), hardware_block.GetInputPort("iiwa_medusa.feedforward_torque"))
    
    root_diagram = root_builder.Build()
    
    # run simulation
    simulator = Simulator(root_diagram)
    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(endtime + 1.0)    

def follow_trajectory_apply_push(traj_thanos, traj_medusa, force = 30.0, endtime = 1e12, scenario_file = "../../config/bimanual_med_hardware_gamma.yaml", directives_file = "../../config/bimanual_med_gamma.yaml"):
    meshcat = StartMeshcat()
        
    root_builder = DiagramBuilder()
    hardware_diagram, hardware_plant = create_hardware_diagram_plant_bimanual(scenario_filepath=scenario_file, position_only=False, meshcat = meshcat)
    vis_diagram = create_visual_diagram(directives_filepath=directives_file, meshcat=meshcat, package_file="../../package.xml")
    
    hardware_block = root_builder.AddSystem(hardware_diagram)
    apply_torque_block = root_builder.AddSystem(ApplyForce(hardware_plant, object_kg = 1.0, force=force))
    torque_demultiplexer_block = root_builder.AddSystem(Demultiplexer(14, 7))
    
    traj_medusa_block = root_builder.AddSystem(TrajectorySource(traj_medusa))
    root_builder.Connect(traj_medusa_block.get_output_port(), hardware_block.GetInputPort("iiwa_medusa.position"))
    root_builder.Connect(traj_medusa_block.get_output_port(), hardware_block.GetInputPort("iiwa_medusa_fake.position"))
    
    traj_thanos_block = root_builder.AddSystem(TrajectorySource(traj_thanos))
    root_builder.Connect(traj_thanos_block.get_output_port(), hardware_block.GetInputPort("iiwa_thanos.position"))
    root_builder.Connect(traj_thanos_block.get_output_port(), hardware_block.GetInputPort("iiwa_thanos_fake.position"))
    
    root_builder.Connect(hardware_block.GetOutputPort("iiwa_thanos.position_measured"), apply_torque_block.GetInputPort("thanos_position"))
    root_builder.Connect(hardware_block.GetOutputPort("iiwa_medusa.position_measured"), apply_torque_block.GetInputPort("medusa_position"))
    root_builder.Connect(apply_torque_block.GetOutputPort("torque"), torque_demultiplexer_block.get_input_port())
    root_builder.Connect(torque_demultiplexer_block.get_output_port(0), hardware_block.GetInputPort("iiwa_thanos.feedforward_torque"))
    root_builder.Connect(torque_demultiplexer_block.get_output_port(1), hardware_block.GetInputPort("iiwa_medusa.feedforward_torque"))
    
    root_diagram = root_builder.Build()

    simulator = Simulator(root_diagram)
    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(endtime + 5.0)

def close_arms(plant_arms: MultibodyPlant, plant_context, q0, gap = 0.0001):
    plant_arms.SetPositions(plant_context, plant_arms.GetModelInstanceByName("iiwa_thanos"), q0[:7])
    plant_arms.SetPositions(plant_context, plant_arms.GetModelInstanceByName("iiwa_medusa"), q0[7:14])
    
    thanos_pose = plant_arms.GetFrameByName("thanos_finger").CalcPoseInWorld(plant_context)
    medusa_pose = plant_arms.GetFrameByName("medusa_finger").CalcPoseInWorld(plant_context)
    
    midpoint_pose = RigidTransform(thanos_pose.rotation().ToQuaternion(), (thanos_pose.translation() + medusa_pose.translation())/2)
    new_thanos_pose = RigidTransform(thanos_pose.rotation().ToQuaternion(), midpoint_pose.translation() + midpoint_pose.rotation().matrix() @ np.array([0, 0, -gap/2.0]))
    
    new_medusa_rotation = thanos_pose.rotation() @ RollPitchYaw(0.0, np.pi, 0.0).ToRotationMatrix()
    new_medusa_pose = RigidTransform(new_medusa_rotation, midpoint_pose.translation() + midpoint_pose.rotation().matrix() @ np.array([0,0, gap/2.0]))

    new_joints,_ = solveDualIK(plant_arms, new_thanos_pose, new_medusa_pose, "thanos_finger", "medusa_finger", q0=q0)
    
    goto_joints(new_joints[:7], new_joints[7:], endtime=5.0)
    
    return new_joints

def inhand_rotate_traj(rotation, rotate_steps, rotate_time, left_pose0, right_pose0, object_pose0):
    left_poses = [left_pose0]
    right_poses = [right_pose0]
    obj_poses = [object_pose0]
    ts = [0.0]
    
    ts, left_poses, right_poses, obj_poses = pause_for(1.0, ts, left_poses, right_poses, obj_poses)
    ts, left_poses, right_poses, obj_poses = inhand_rotate_poses(rotation, object_pose0, ts, left_poses, right_poses, obj_poses, steps=rotate_steps, rotate_time=rotate_time)
    ts, left_poses, right_poses, obj_poses = pause_for(1.0, ts, left_poses, right_poses, obj_poses)
    left_piecewise, right_piecewise, _ = piecewise_traj(ts, left_poses, right_poses, obj_poses)
    return left_piecewise, right_piecewise, ts[-1]
def generate_trajectory(plant_arms: MultibodyPlant, q0, left_piecewise, right_piecewise, T, tsteps = 100):
    ts = np.linspace(0, T, tsteps)
    qs = solve_ik_inhand(plant_arms, ts, left_piecewise, right_piecewise, "thanos_finger", "medusa_finger", q0)
    thanos_piecewise = PiecewisePolynomial.FirstOrderHold(ts, qs[:,:7].T)
    medusa_piecewise = PiecewisePolynomial.FirstOrderHold(ts, qs[:,7:].T)
    return thanos_piecewise, medusa_piecewise, T