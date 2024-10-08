from diagrams import create_hardware_diagram_plant_bimanual, create_visual_diagram
from pydrake.all import (
    StartMeshcat,
    Simulator,
    DiagramBuilder,
    TrajectorySource,
    PiecewisePolynomial
)
import numpy as np
JOINT_CONFIG0 = [-0.32823683178594826, 0.9467527057457398, 1.5375963846252783, -2.055496608537348, -0.8220809597822779, -0.31526250680171636, 1.3872151028590527,
                 -1.7901817338098867, 1.2653964889934661, 1.740960078785441, -2.014334314596287, 0.35305405885912783, -1.8242723561461582, -0.01502208888994321]
# JOINT_CONFIG0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
#                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
JOINT0_THANOS = np.array([JOINT_CONFIG0[:7]]).flatten()
JOINT0_MEDUSA = np.array([JOINT_CONFIG0[7:14]]).flatten()

ENDTIME = 60.0
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