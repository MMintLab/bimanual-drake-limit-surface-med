#!/usr/bin/env python
import numpy as np
from pydrake.all import (
    StartMeshcat,
    Simulator,
    DiagramBuilder,
    Demultiplexer,
    MultibodyPlant,
    DiagramBuilder,
    RotationMatrix,
    ConstantVectorSource,
    LeafSystem
)


import rospy
from geometry_msgs.msg import WrenchStamped
from netft_rdt_driver.srv import Zero

import sys
sys.path.append('..')
from diagrams import create_hardware_diagram_plant_bimanual, create_visual_diagram
from run_plan_main import curr_joints
from test_impedance import Wrench2Torque

#import blkdiag
from scipy.linalg import block_diag

'''
    GOAL:
    - cancel out the contact wrench's effect on kukas using the ATI Gamma
    - Must Zero gammas, then any sensed force is due to the contact wrench
    STEPS:
    - Get robots in position
    - Use hacky way of pushing lower hand to upper hand
        - the result is both kukas aligning.
    - zero gammas
    - move se2 and feedforward any sensed forces
'''

JOINT_CONFIG0 = [1.1429203100700507, 0.7961505418611234, -0.2334648267973291, -0.4733848043491367, -2.967059728388293, -1.8813130837959888, -2.109669307516976, -1.5677529803998187, 1.9624159075892007, -1.4980034238420221, 0.9229619082335976, -1.9288236774214953, 1.7484457277131111, -2.5145907690545455]

class GammaManager:
    def __init__(self):
        self.thanos_ati_sub = rospy.Subscriber('/netft_thanos/netft_data', WrenchStamped, self.thanos_cb)
        self.medusa_ati_sub = rospy.Subscriber('/netft_medusa/netft_data', WrenchStamped, self.medusa_cb)
        self.thanos_wrench = None
        self.medusa_wrench = None
        rotationMat_medusa = RotationMatrix.MakeZRotation(135 * np.pi / 180.0).matrix()
        rotationMat_thanos = RotationMatrix.MakeZRotation(-90 * np.pi / 180.0).matrix()
        #blkdiag
        self.R_medusa = block_diag(rotationMat_medusa, rotationMat_medusa)
        self.R_thanos = block_diag(rotationMat_thanos, rotationMat_thanos)
    def thanos_cb(self, msg: WrenchStamped):
        self.thanos_wrench = self.R_thanos @ np.array([msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z, 
                    msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])
    def medusa_cb(self, msg: WrenchStamped):
        self.medusa_wrench = self.R_medusa @ np.array([msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z,
                    msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])
# develop code to counteract any force applied to arm
# use the ATI gamma to measure the force applied to the arm
# zero the ATI gamma
# feedforward the force measured by the ATI gamma
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
        thanos_wrench = -1 * (self.filter_mat_thanos @ block_diag(thanos_rot, thanos_rot) @ thanos_wrench)
        output.SetFromVector(thanos_wrench)
    def CalcWrenchMedusa(self, context, output):
        medusa_q = self.medusa_iiwa.Eval(context)
        self._plant.SetPositions(self._plant_context, self._plant.GetModelInstanceByName("iiwa_medusa"), medusa_q)
        
        medusa_pose = self._plant.GetFrameByName("medusa_finger").CalcPoseInWorld(self._plant_context)
        medusa_rot = medusa_pose.rotation().matrix()
        
        medusa_wrench = self.gamma_manager.medusa_wrench if self.gamma_manager.medusa_wrench is not None else np.zeros(6)
        medusa_wrench = -1 * (self.filter_mat_medusa @ block_diag(medusa_rot, medusa_rot) @ medusa_wrench)
        output.SetFromVector(medusa_wrench)
        
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
    rospy.init_node('test_gamma_stuff')
    rospy.wait_for_service('/netft_thanos/zero')
    rospy.wait_for_service('/netft_medusa/zero')
    
    scenario_file = "../../config/bimanual_med_hardware_gamma.yaml"
    directives_file = "../../config/bimanual_med_gamma.yaml"
    
    curr_q = curr_joints()
    
    
    input("Press Enter to zero gammas.")
    # zero gammas
    zero_thanos = rospy.ServiceProxy('/netft_thanos/zero', Zero)
    zero_medusa = rospy.ServiceProxy('/netft_medusa/zero', Zero)
    zero_thanos()
    zero_medusa()
    
    # reactive time
    
    input("Press Enter to activate reactive mode.")
    gamma_manager = GammaManager()
    reactive_arm_force(curr_q[:7], curr_q[7:], gamma_manager, endtime = 30.0, scenario_file = scenario_file, directives_file = directives_file)
    
    
    