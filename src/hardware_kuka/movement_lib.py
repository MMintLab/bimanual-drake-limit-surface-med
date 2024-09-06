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
    Multiplexer,
    PiecewisePose,
    Adder,
    LeafSystem
)
import numpy as np

import sys
sys.path.append('..')
from diagrams import create_hardware_diagram_plant_bimanual, create_visual_diagram
from planning.ik_util import solveDualIK, inhand_rotate_poses, pause_for, piecewise_traj, solve_ik_inhand, inhand_rotate_arms, inhand_se2_arms
from bimanual_systems import Wrench2Torque, ApplyForceCompensateGravity
from planning.object_compensation import ApplyForce
from gamma import GammaManager
from scipy.linalg import block_diag
from camera import CameraManager

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

def follow_trajectory_apply_push(traj_thanos, traj_medusa, camera_manager: CameraManager, force = 30.0, object_kg = 0.5, feedforward_z_force = 0.0, endtime = 1e12, scenario_file = "../../config/bimanual_med_hardware_gamma.yaml", directives_file = "../../config/bimanual_med_gamma.yaml"):
    meshcat = StartMeshcat()
        
    root_builder = DiagramBuilder()
    hardware_diagram, hardware_plant = create_hardware_diagram_plant_bimanual(scenario_filepath=scenario_file, position_only=False, meshcat = meshcat)
    vis_diagram = create_visual_diagram(directives_filepath=directives_file, meshcat=meshcat, package_file="../../package.xml")
    
    hardware_block = root_builder.AddSystem(hardware_diagram)
    # apply_torque_block = root_builder.AddSystem(ApplyForce(hardware_plant, object_kg = 1.0, force=force))
    apply_torque_block = root_builder.AddSystem(ApplyForceCompensateGravity(hardware_plant, camera_manager, object_kg = object_kg, applied_force=force, feedforward_z_force=feedforward_z_force))
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

class ReactiveGamma(LeafSystem):
    def __init__(self, plant: MultibodyPlant, gamma_manager:  GammaManager, filter_vector_medusa = np.ones(6), filter_vector_thanos = np.ones(6)):
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self.gamma_manager = gamma_manager
        self.filter_mat_medusa = np.diag(filter_vector_medusa)
        self.filter_mat_thanos = np.diag(filter_vector_thanos)
        self.medusa_iiwa = self.DeclareVectorInputPort("medusa_iiwa", 7)
        self.thanos_iiwa = self.DeclareVectorInputPort("thanos_iiwa", 7)
        self.wrench_thanos_port = self.DeclareVectorOutputPort("wrench_thanos", 6, self.CalcWrenchThanos)
        self.wrench_medusa_port = self.DeclareVectorOutputPort("wrench_medusa", 6, self.CalcWrenchMedusa)
    def CalcWrenchThanos(self, context, output):
        thanos_q = self.thanos_iiwa.Eval(context)
        self._plant.SetPositions(self._plant_context, self._plant.GetModelInstanceByName("iiwa_thanos"), thanos_q)
        
        thanos_pose = self._plant.GetFrameByName("thanos_finger").CalcPoseInWorld(self._plant_context)
        thanos_rot = thanos_pose.rotation().matrix()
        
        thanos_wrench = self.gamma_manager.thanos_wrench if self.gamma_manager.thanos_wrench is not None else np.zeros(6)
        thanos_wrench = -1 * (block_diag(thanos_rot, thanos_rot) @ self.filter_mat_thanos @ thanos_wrench)
        output.SetFromVector(thanos_wrench)
    def CalcWrenchMedusa(self, context, output):
        medusa_q = self.medusa_iiwa.Eval(context)
        self._plant.SetPositions(self._plant_context, self._plant.GetModelInstanceByName("iiwa_medusa"), medusa_q)
        feedforward_z_force = 0.0,
        medusa_pose = self._plant.GetFrameByName("medusa_finger").CalcPoseInWorld(self._plant_context)
        medusa_rot = medusa_pose.rotation().matrix()
        
        medusa_wrench = self.gamma_manager.medusa_wrench if self.gamma_manager.medusa_wrench is not None else np.zeros(6)
        medusa_wrench = -1 * (block_diag(medusa_rot, medusa_rot) @ self.filter_mat_medusa @ medusa_wrench)
        output.SetFromVector(medusa_wrench)

def follow_traj_and_torque_gamma(traj_thanos, traj_medusa, camera_manager: CameraManager, gamma_manager: GammaManager, force = 30.0, object_kg = 0.5, endtime = 1e12, scenario_file = "../../config/bimanual_med_hardware_gamma.yaml", directives_file = "../../config/bimanual_med_gamma.yaml", feedforward_z_force = 0.0, filter_vector_medusa = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 0.0]), filter_vector_thanos = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 0.0])):
    meshcat = StartMeshcat()
    
    root_builder = DiagramBuilder()
    hardware_diagram, hardware_plant = create_hardware_diagram_plant_bimanual(scenario_filepath=scenario_file, position_only=False, meshcat = meshcat)
    vis_diagram = create_visual_diagram(directives_filepath=directives_file, meshcat=meshcat, package_file="../../package.xml")
    
    hardware_block = root_builder.AddSystem(hardware_diagram)
    
    apply_torque_block = root_builder.AddSystem(ApplyForce(hardware_plant, object_kg = object_kg, force=force))
    # apply_torque_block = root_builder.AddSystem(ApplyForceCompensateGravity(hardware_plant, camera_manager, object_kg = object_kg, applied_force=force, feedforward_z_force=feedforward_z_force))
    torque_demultiplexer_block = root_builder.AddSystem(Demultiplexer(14, 7))
    
    traj_medusa_block = root_builder.AddSystem(TrajectorySource(traj_medusa))
    traj_thanos_block = root_builder.AddSystem(TrajectorySource(traj_thanos))
    
    reactive_wrench_block = root_builder.AddSystem(ReactiveGamma(hardware_plant, gamma_manager, filter_vector_medusa = filter_vector_medusa, filter_vector_thanos=filter_vector_thanos))
    wrench2torque_block = root_builder.AddSystem(Wrench2Torque(hardware_plant))
    reactive_torque_demultiplexer_block = root_builder.AddSystem(Demultiplexer(14, 7))
    
    adder_torque_thanos_block = root_builder.AddSystem(Adder(2, 7))
    adder_torque_medusa_block = root_builder.AddSystem(Adder(2, 7))
    
    root_builder.Connect(traj_thanos_block.get_output_port(), hardware_block.GetInputPort("iiwa_thanos.position"))
    root_builder.Connect(traj_thanos_block.get_output_port(), hardware_block.GetInputPort("iiwa_thanos_fake.position"))
    
    root_builder.Connect(traj_medusa_block.get_output_port(), hardware_block.GetInputPort("iiwa_medusa.position"))
    root_builder.Connect(traj_medusa_block.get_output_port(), hardware_block.GetInputPort("iiwa_medusa_fake.position"))
    
    
    # connections for reactive torque
    root_builder.Connect(hardware_block.GetOutputPort("iiwa_thanos.position_measured"), reactive_wrench_block.GetInputPort("thanos_iiwa"))
    root_builder.Connect(hardware_block.GetOutputPort("iiwa_medusa.position_measured"), reactive_wrench_block.GetInputPort("medusa_iiwa"))
    root_builder.Connect(hardware_block.GetOutputPort("iiwa_thanos.position_measured"), wrench2torque_block.GetInputPort("thanos_position"))
    root_builder.Connect(hardware_block.GetOutputPort("iiwa_medusa.position_measured"), wrench2torque_block.GetInputPort("medusa_position"))
    root_builder.Connect(reactive_wrench_block.GetOutputPort("wrench_thanos"), wrench2torque_block.GetInputPort("wrench_thanos"))
    root_builder.Connect(reactive_wrench_block.GetOutputPort("wrench_medusa"), wrench2torque_block.GetInputPort("wrench_medusa"))
    root_builder.Connect(wrench2torque_block.GetOutputPort("torque"), reactive_torque_demultiplexer_block.get_input_port())
    
    # connections for apply torque
    root_builder.Connect(hardware_block.GetOutputPort("iiwa_thanos.position_measured"), apply_torque_block.GetInputPort("thanos_position"))
    root_builder.Connect(hardware_block.GetOutputPort("iiwa_medusa.position_measured"), apply_torque_block.GetInputPort("medusa_position"))
    root_builder.Connect(apply_torque_block.GetOutputPort("torque"), torque_demultiplexer_block.get_input_port())
    
    # sum torques
    root_builder.Connect(torque_demultiplexer_block.get_output_port(0), adder_torque_thanos_block.get_input_port(0))
    root_builder.Connect(reactive_torque_demultiplexer_block.get_output_port(0), adder_torque_thanos_block.get_input_port(1))
    
    root_builder.Connect(torque_demultiplexer_block.get_output_port(1), adder_torque_medusa_block.get_input_port(0))
    root_builder.Connect(reactive_torque_demultiplexer_block.get_output_port(1), adder_torque_medusa_block.get_input_port(1))
    
    # connect to hardware feedforward torque
    root_builder.Connect(adder_torque_thanos_block.get_output_port(), hardware_block.GetInputPort("iiwa_thanos.feedforward_torque"))
    root_builder.Connect(adder_torque_medusa_block.get_output_port(), hardware_block.GetInputPort("iiwa_medusa.feedforward_torque"))
    
    root_diagram = root_builder.Build()
    simulator = Simulator(root_diagram)
    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(endtime + 2.0)

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

def generate_trajectory(plant_arms: MultibodyPlant, q0, left_piecewise, right_piecewise, T, tsteps = 100):
    ts = np.linspace(0, T, tsteps)
    qs = solve_ik_inhand(plant_arms, ts, left_piecewise, right_piecewise, "thanos_finger", "medusa_finger", q0)
    thanos_piecewise = PiecewisePolynomial.FirstOrderHold(ts, qs[:,:7].T)
    medusa_piecewise = PiecewisePolynomial.FirstOrderHold(ts, qs[:,7:].T)
    return thanos_piecewise, medusa_piecewise, T

def inhand_rotate_traj(rotation, rotate_steps, rotate_time, left_pose0: RigidTransform, right_pose0: RigidTransform, current_obj2medusa_se2 = np.array([0.00,0.0,0])):
    ts, left_poses, right_poses = inhand_rotate_arms(left_pose0, right_pose0, current_obj2medusa_se2, rotation=rotation, steps=rotate_steps, rotate_time=rotate_time)
    
    left_piecewise = PiecewisePose.MakeLinear(ts, left_poses)
    right_piecewise = PiecewisePose.MakeLinear(ts, right_poses)
    return left_piecewise, right_piecewise, ts[-1]

def inhand_se2_traj(left_pose0: RigidTransform, right_pose0: RigidTransform, current_obj2arm_se2 = np.array([0.00,0.0,0]), desired_obj2arm_se2 = np.array([0.00,0.0,0]),medusa = True, se2_time = 10.0):
    left_pose, right_pose = inhand_se2_arms(left_pose0, right_pose0, current_obj2arm_se2, desired_obj2arm_se2, medusa=medusa)
    
    left_poses = [left_pose0, left_pose]
    right_poses = [right_pose0, right_pose]
    ts = [0, se2_time]
    
    left_piecewise = PiecewisePose.MakeLinear(ts, left_poses)
    right_piecewise = PiecewisePose.MakeLinear(ts, right_poses)
    
    return left_piecewise, right_piecewise, ts[-1]