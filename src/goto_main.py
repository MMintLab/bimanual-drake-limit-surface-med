from diagrams import create_hardware_diagram_plant_bimanual, create_visual_diagram
from pydrake.all import (
    StartMeshcat,
    Simulator,
    DiagramBuilder,
    TrajectorySource,
    PiecewisePolynomial
)
import numpy as np
JOINT0_THANOS = np.array([-1.4950940483482236, 0.45737168347437573, 1.4222491359501224, -1.790658436904843, 1.1211460873482524, -1.6082243127886424, 0.26302021289848687])
JOINT0_MEDUSA = np.array([0.3022651168325996, 0.8943288129955809, 1.1148467035895304, -1.1602488967760345, 1.0310164359946563, 0.30727928231604223, -0.24626643310262913])
# JOINT0 = np.zeros(7)
ENDTIME = 30.0
if __name__ == '__main__':
    meshcat = StartMeshcat()
    scenario_file = "../config/bimanual_med_hardware.yaml"
    directives_file = "../config/bimanual_med.yaml"
    
    root_builder = DiagramBuilder()
    
    hardware_diagram, hardware_plant = create_hardware_diagram_plant_bimanual(scenario_filepath=scenario_file, meshcat=meshcat, position_only=True)
    vis_diagram = create_visual_diagram(directives_filepath=directives_file, meshcat=meshcat, package_file="../package.xml")
    
    hardware_block = root_builder.AddSystem(hardware_diagram)
    
    ## make a plan from current position to desired position
    context = hardware_diagram.CreateDefaultContext()
    hardware_diagram.ExecuteInitializationEvents(context)
    curr_q_medusa = hardware_diagram.GetOutputPort("iiwa_medusa.position_commanded").Eval(context)
    curr_q_thanos = hardware_diagram.GetOutputPort("iiwa_thanos.position_commanded").Eval(context)
    ts = np.array([0.0, ENDTIME])
    
    qs_medusa = np.array([curr_q_medusa, JOINT0_MEDUSA])
    traj_medusa = PiecewisePolynomial.FirstOrderHold(ts, qs_medusa.T)
    
    qs_thanos = np.array([curr_q_thanos, JOINT0_THANOS])
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
    simulator.AdvanceTo(ENDTIME + 2.0)