#!/usr/bin/env python
import rospy
from netft_rdt_driver.srv import Zero
from geometry_msgs.msg import WrenchStamped


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
    JacobianWrtVariable,
    Adder
)
import numpy as np

import sys
sys.path.append('..')
from diagrams import create_hardware_diagram_plant_bimanual, create_visual_diagram
from run_plan_main import curr_joints, goto_joints
from load.sim_setup import load_iiwa_setup
from planning.ik_util import generate_push_configuration, inhand_test
from test_impedance import goto_and_torque
from test_gamma_stuff import GammaManager, ReactiveGamma
from planning.object_compensation import ApplyForce
from test_impedance import Wrench2Torque
from test_tilted_impedance import generate_se2_motion


def zero_ati_gamma():
    rospy.wait_for_service('/netft_thanos/zero')
    rospy.wait_for_service('/netft_medusa/zero')
    zero_thanos = rospy.ServiceProxy('/netft_thanos/zero', Zero)
    zero_medusa = rospy.ServiceProxy('/netft_medusa/zero', Zero)
    zero_thanos()
    zero_medusa()

def setup_joints(des_q, joint_speed = 5.0 * np.pi / 180.0):
    curr_q = curr_joints()
    thanos_endtime = np.max(np.abs(des_q[:7] - curr_q[:7])) / joint_speed
    medusa_endtime = np.max(np.abs(des_q[7:14] - curr_q[7:14])) / joint_speed
    
    print("medusa_endtime: ", medusa_endtime)
    print("thanos_endtime: ", thanos_endtime)
    input("Press Enter to setup medusa arm.")
    goto_joints(curr_q[:7], des_q[7:], endtime = medusa_endtime)
    input("Press Enter to setup thanos arm.")
    goto_joints(des_q[:7], des_q[7:], endtime = thanos_endtime)

def setup_push(desired_gap, start_q):
    scenario_file = "../../config/bimanual_med_hardware_gamma.yaml"
    directives_file = "../../config/bimanual_med_gamma.yaml"
    
    plant_arms = MultibodyPlant(1e-3) # time step
    load_iiwa_setup(plant_arms, package_file='../../package.xml', directive_path=directives_file)
    plant_arms.Finalize()
    
    des_q = generate_push_configuration(plant_arms, start_q, gap=desired_gap)
    input("Press Enter to get fingers closer")
    
    goto_joints(des_q[:7], des_q[7:], endtime = 1.0, scenario_file=scenario_file, directives_file=directives_file)
    return des_q
def start_pushing(des_q, force = 10.0, object_kg = 0.5):
    scenario_file = "../../config/bimanual_med_hardware_gamma.yaml"
    directives_file = "../../config/bimanual_med_gamma.yaml"
    
    plant_arms = MultibodyPlant(1e-3) # time step
    load_iiwa_setup(plant_arms, package_file='../../package.xml', directive_path=directives_file)
    plant_arms.Finalize()
    
    curr_q = curr_joints()
    plant_context = plant_arms.CreateDefaultContext()
    plant_arms.SetPositions(plant_context, plant_arms.GetModelInstanceByName("iiwa_thanos"), des_q[:7])
    plant_arms.SetPositions(plant_context, plant_arms.GetModelInstanceByName("iiwa_medusa"), des_q[7:14])
    thanos_pose = plant_arms.GetFrameByName("thanos_finger").CalcPoseInWorld(plant_context)
    medusa_pose = plant_arms.GetFrameByName("medusa_finger").CalcPoseInWorld(plant_context)
    
    input("Press Enter to start pushing")
    object_kg = 0.5
    grav = 9.81
    force = 10.0
    wrench_thanos = thanos_pose.rotation().matrix() @ np.array([0, 0.0, force])
    wrench_medusa = medusa_pose.rotation().matrix() @ np.array([0, 0.0, force + object_kg * grav])
    wrench_thanos = np.concatenate([np.zeros(3), wrench_thanos])
    wrench_medusa = np.concatenate([np.zeros(3), wrench_medusa])
    goto_and_torque(des_q[:7], des_q[7:], endtime = 5.0, wrench_thanos=wrench_thanos, wrench_medusa=wrench_medusa, scenario_file=scenario_file, directives_file=directives_file)
    
def wrenchstamped2wrench(wrenchstamped):
    return np.array([wrenchstamped.wrench.torque.x, wrenchstamped.wrench.torque.y, wrenchstamped.wrench.torque.z, 
                    wrenchstamped.wrench.force.x, wrenchstamped.wrench.force.y, wrenchstamped.wrench.force.z])


def follow_traj_and_torque_gamma(traj_thanos, traj_medusa, gamma_manager: GammaManager, force = 30.0, object_kg = 0.5, endtime = 1e12):
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
    traj_thanos_block = root_builder.AddSystem(TrajectorySource(traj_thanos))
    
    reactive_wrench_block = root_builder.AddSystem(ReactiveGamma(hardware_plant, gamma_manager, filter_vector_medusa = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 0.0]), filter_vector_thanos=np.zeros(6)))
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

if __name__ == '__main__':
    joint_horiz = np.array([1.1429203100700507, 0.7961505418611234, -0.2334648267973291, -0.4733848043491367, -2.967059728388293, -1.8813130837959888, -2.109669307516976,
                   -1.5677529803998187, 1.9624159075892007, -1.4980034238420221, 0.9229619082335976, -1.9288236774214953, 1.7484457277131111, -2.5145907690545455])
    setup_joints(joint_horiz)
    desired_gap = 0.001
    desired_q = setup_push(desired_gap, joint_horiz)
    start_pushing(desired_q, force=10.0, object_kg=0.5)
    zero_ati_gamma()
    
    gamma_manager = GammaManager()
    traj_thanos, traj_medusa, T = generate_se2_motion(desired_q, desired_obj2left_se2 = np.array([0.0, -0.04, 0.0]), gap = desired_gap)
    input("Press Enter to start pushing with gamma")
    follow_traj_and_torque_gamma(traj_thanos, traj_medusa, gamma_manager, force = 10.0, object_kg = 0.5, endtime = T)
    pass