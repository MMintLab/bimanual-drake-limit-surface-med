from pydrake.all import (
    StartMeshcat,
    Simulator,
    DiagramBuilder,
    ConstantVectorSource,
    LeafSystem,
    JacobianWrtVariable,
    Demultiplexer
)
from pydrake.multibody.plant import MultibodyPlant
from pydrake.systems.framework import DiagramBuilder

import numpy as np

import numpy as np

import sys
sys.path.append('..')
from diagrams import create_hardware_diagram_plant_bimanual, create_visual_diagram
from run_plan_main import goto_joints, curr_joints

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
                                                           self._plant.world_frame())[:, :7]
        J_medusa = self._plant.CalcJacobianSpatialVelocity(self._plant_context,
                                                           JacobianWrtVariable.kQDot,
                                                           self._plant.GetBodyByName("iiwa_link_7", self._medusa_instance).body_frame(),
                                                           [0,0,0],
                                                           self._plant.world_frame(),
                                                           self._plant.world_frame())[:, 7:]
        print(J_thanos)
        thanos_torque = J_thanos.T @ wrench_thanos
        medusa_torque = J_medusa.T @ wrench_medusa
        output.SetFromVector(np.concatenate([thanos_torque, medusa_torque]))

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

if __name__ == '__main__':
    iiwa_default_q = np.array([0, np.pi/4,  0, -np.pi/4, 0 , np.pi/2 , 0])
    des_q = np.concatenate([iiwa_default_q, iiwa_default_q])
    curr_q = curr_joints()
    input("Press Enter to move medusa to default")
    goto_joints(curr_q[:7], des_q[7:],endtime=15)
    input("Press Enter to move thanos to default")
    goto_joints(des_q[:7], des_q[7:],endtime=15)
    
    input("Press Enter to start impedance test")
    wrench = np.array([0, 0, 0, 0, 0, -0.0])
    
    goto_and_torque(des_q[:7], des_q[7:], wrench, wrench)
    
    
    pass