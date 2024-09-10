#!/usr/bin/env python
import rospy

from gamma import GammaManager

from pydrake.all import (
    ConstantVectorSource,
    DiagramBuilder,
    StartMeshcat,
    Demultiplexer,
    Simulator
)

import sys
sys.path.append('..')
from diagrams import create_hardware_diagram_plant_bimanual, create_visual_diagram
from bimanual_systems import Wrench2Torque
from movement_lib import curr_joints, ReactiveGamma


def reactive_arm_force(joint_thanos, joint_medusa, gamma_manager: GammaManager, endtime = 30.0, scenario_file = "../../config/bimanual_med_hardware_impedance.yaml", directives_file = "../../config/bimanual_med.yaml"):
    meshcat = StartMeshcat()
    
    root_builder = DiagramBuilder()
    hardware_diagram, hardware_plant = create_hardware_diagram_plant_bimanual(scenario_filepath=scenario_file, position_only=False, meshcat = meshcat)
    vis_diagram = create_visual_diagram(directives_filepath=directives_file, package_file="../../package.xml", meshcat = meshcat)
    
    hardware_block = root_builder.AddSystem(hardware_diagram)
    traj_medusa_block = root_builder.AddSystem(ConstantVectorSource(joint_medusa))
    traj_thanos_block = root_builder.AddSystem(ConstantVectorSource(joint_thanos))
    
    
    wrench_block = root_builder.AddSystem(ReactiveGamma(hardware_plant,gamma_manager))
    
    torque_block = root_builder.AddSystem(Wrench2Torque(hardware_plant))
    
    torque_demultiplexer_block = root_builder.AddSystem(Demultiplexer(14, 7))

    root_builder.Connect(traj_thanos_block.get_output_port(), hardware_block.GetInputPort("iiwa_thanos.position"))
    root_builder.Connect(traj_thanos_block.get_output_port(), hardware_block.GetInputPort("iiwa_thanos_fake.position"))
    
    root_builder.Connect(traj_medusa_block.get_output_port(), hardware_block.GetInputPort("iiwa_medusa.position"))
    root_builder.Connect(traj_medusa_block.get_output_port(), hardware_block.GetInputPort("iiwa_medusa_fake.position"))

    root_builder.Connect(hardware_block.GetOutputPort("iiwa_thanos.position_measured"), torque_block.GetInputPort("thanos_position"))
    root_builder.Connect(hardware_block.GetOutputPort("iiwa_medusa.position_measured"), torque_block.GetInputPort("medusa_position"))

    root_builder.Connect(hardware_block.GetOutputPort("iiwa_thanos.position_measured"), wrench_block.GetInputPort("thanos_iiwa"))
    root_builder.Connect(hardware_block.GetOutputPort("iiwa_medusa.position_measured"), wrench_block.GetInputPort("medusa_iiwa"))
    
    root_builder.Connect(wrench_block.GetOutputPort("wrench_thanos"), torque_block.GetInputPort("wrench_thanos"))
    root_builder.Connect(wrench_block.GetOutputPort("wrench_medusa"), torque_block.GetInputPort("wrench_medusa"))
    
    root_builder.Connect(torque_block.GetOutputPort("torque"), torque_demultiplexer_block.get_input_port())
    
    root_builder.Connect(torque_demultiplexer_block.get_output_port(0), hardware_block.GetInputPort("iiwa_thanos.feedforward_torque"))
    root_builder.Connect(torque_demultiplexer_block.get_output_port(1), hardware_block.GetInputPort("iiwa_medusa.feedforward_torque"))
    
    root_diagram = root_builder.Build()
    
    # run simulation
    simulator = Simulator(root_diagram)
    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(endtime + 2.0)
    
    
if __name__ == '__main__':
    rospy.init_node('test_perfect_gamma')
    scenario_file = "../../config/bimanual_med_hardware_gamma.yaml"
    directives_file = "../../config/bimanual_med_gamma.yaml"
    
    curr_q = curr_joints(scenario_file=scenario_file)
    
    # get poses
    
    
    input("Press Enter to activate reactive mode.")
    gamma_manager = GammaManager(use_compensation=True)
    reactive_arm_force(curr_q[:7], curr_q[7:], gamma_manager, endtime = 30.0, scenario_file = scenario_file, directives_file = directives_file)