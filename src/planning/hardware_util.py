import numpy as np

from pydrake.all import (
    StartMeshcat,
    Simulator,
    DiagramBuilder,
    TrajectorySource,
    Demultiplexer,
    RigidTransform,
    MultibodyPlant,
    DiagramBuilder,
    PiecewisePolynomial,
    ConstantVectorSource,
    LeafSystem,
    JacobianWrtVariable
)
import numpy as np

import sys
sys.path.append('..')
from diagrams import create_hardware_diagram_plant_bimanual, create_visual_diagram
from planning.ik_util import solve_ik_inhand, run_full_inhand_og, piecewise_traj
from planning.object_compensation import ApplyForce
from load.sim_setup import load_iiwa_setup

class Wrench2Torque(LeafSystem):
    def __init__(self, plant: MultibodyPlant):
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._wrench_thanos = self.DeclareVectorInputPort("wrench_thanos", 6)
        self._wrench_medusa = self.DeclareVectorInputPort("wrench_medusa", 6)
        self._medusa_position = self.DeclareVectorInputPort("medusa_position", 7)
        self._thanos_position = self.DeclareVectorInputPort("thanos_position", 7)
        
        self._torque_port = self.DeclareVectorOutputPort("torque", 14, self.DoCalcOutput)
    def DoCalcOutput(self, context, output):
        wrench_thanos = self._wrench_thanos.Eval(context)
        wrench_medusa = self._wrench_medusa.Eval(context)
        medusa_pos = self._medusa_position.Eval(context)
        thanos_pos = self._thanos_position.Eval(context)
        
        self._thanos_instance = self._plant.GetModelInstanceByName("iiwa_thanos")
        self._medusa_instance = self._plant.GetModelInstanceByName("iiwa_medusa")
        
        #calc jacobian
        self._plant.SetPositions(self._plant_context, self._thanos_instance, thanos_pos)
        self._plant.SetPositions(self._plant_context, self._medusa_instance, medusa_pos)
        J_thanos = self._plant.CalcJacobianSpatialVelocity(self._plant_context, 
                                                           JacobianWrtVariable.kQDot,
                                                           self._plant.GetBodyByName("iiwa_link_7", self._thanos_instance).body_frame(),
                                                           [0,0,0],
                                                           self._plant.world_frame(),
                                                           self._plant.world_frame())[:, 7:]
        J_medusa = self._plant.CalcJacobianSpatialVelocity(self._plant_context,
                                                           JacobianWrtVariable.kQDot,
                                                           self._plant.GetBodyByName("iiwa_link_7", self._medusa_instance).body_frame(),
                                                           [0,0,0],
                                                           self._plant.world_frame(),
                                                           self._plant.world_frame())[:, :7]
        
        thanos_torque = J_thanos.T @ wrench_thanos
        medusa_torque = J_medusa.T @ wrench_medusa
        output.SetFromVector(np.concatenate([thanos_torque, medusa_torque]))

def curr_joints():
    scenario_file = "../../config/bimanual_med_hardware.yaml"
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

def goto_and_torque(joint_thanos, joint_medusa, wrench_thanos, wrench_medusa, endtime = 30.0, scenario_file = "../../config/bimanual_med_hardware_impedance.yaml", directives_file = "../../config/bimanual_med.yaml"):
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
    simulator.AdvanceTo(endtime + 2.0)

def generate_trajectory(seed_q0, rotation=np.pi/3, desired_obj2left_se2 = np.array([0.00, 0.0, 0.0]), desired_obj2right_se2 = np.array([0.00, 0.0, np.pi])):
    plant_arms = MultibodyPlant(1e-3) # time step
    load_iiwa_setup(plant_arms, package_file='../../package.xml', directive_path="../../config/bimanual_med.yaml")
    plant_arms.Finalize()

    plant_context = plant_arms.CreateDefaultContext()
    plant_arms.SetPositions(plant_context, plant_arms.GetModelInstanceByName("iiwa_thanos"), seed_q0[:7])
    plant_arms.SetPositions(plant_context, plant_arms.GetModelInstanceByName("iiwa_medusa"), seed_q0[7:14])
    
    left_pose0 = plant_arms.GetFrameByName("thanos_finger").CalcPoseInWorld(plant_context)
    right_pose0 = plant_arms.GetFrameByName("medusa_finger").CalcPoseInWorld(plant_context)
    
    gap = np.linalg.norm(left_pose0.translation() - right_pose0.translation()) # ASSUME left and right fingers are xy-aligned
    print(gap)
    object_pose0 = RigidTransform(left_pose0.rotation().ToQuaternion(), left_pose0.translation() + left_pose0.rotation().matrix() @ np.array([0,0,(gap)/2.0]))

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

def follow_trajectory_and_torque(traj_thanos, traj_medusa, force = 30.0, object_kg = 0.5, endtime = 1e12):
    meshcat = StartMeshcat()
    scenario_file = "../../config/bimanual_med_hardware_impedance.yaml"
    directives_file = "../../config/bimanual_med.yaml"
        
    root_builder = DiagramBuilder()
    
    hardware_diagram, hardware_plant = create_hardware_diagram_plant_bimanual(scenario_filepath=scenario_file, position_only=False, meshcat = meshcat)
    vis_diagram = create_visual_diagram(directives_filepath=directives_file, meshcat=meshcat, package_file="../../package.xml")
    
    hardware_block = root_builder.AddSystem(hardware_diagram)
    apply_torque_block = root_builder.AddSystem(ApplyForce(hardware_plant, object_kg = object_kg, force=force))
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