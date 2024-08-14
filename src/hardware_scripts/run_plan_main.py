from diagrams import create_hardware_diagram_plant_bimanual, create_visual_diagram
from pydrake.all import (
    StartMeshcat,
    Simulator,
    DiagramBuilder,
    TrajectorySource,
    PiecewisePolynomial
)
from load.sim_setup import load_iiwa_setup
from pydrake.multibody.plant import MultibodyPlant, MultibodyPlantConfig, AddMultibodyPlant
from pydrake.systems.framework import DiagramBuilder
from pydrake.math import RigidTransform
from pydrake.visualization import AddDefaultVisualization
from manipulation.scenarios import AddMultibodyTriad
from pydrake.all import Quaternion
import numpy as np
from planning.ik_util import solve_ik_inhand, piecewise_joints, run_full_inhand_og, piecewise_traj
import numpy as np
# JOINT_CONFIG0 = [-0.32823683178594826, 0.9467527057457398, 1.5375963846252783, -2.055496608537348, -0.8220809597822779, -0.31526250680171636, 1.3872151028590527,
#                  -1.7901817338098867, 1.2653964889934661, 1.740960078785441, -2.014334314596287, 0.35305405885912783, -1.8242723561461582, -0.01502208888994321]
JOINT_PUSHED0 = [-0.2643785411955492, 0.9298086509635983, 1.4995078330737035, -2.0178285851796245, -0.7956487782329831, -0.33073534353598244, 1.3776540552821948, 
                 -1.8013266922213016, 1.2733419322042039, 1.7303881921586168, -1.975218704173754, 0.33999535648913914, -1.772941644931603, -0.0229759501547755]

GAP = 0.475
PUSH_DISTANCE = 0.025

JOINT0_THANOS = np.array([JOINT_PUSHED0[:7]]).flatten()
JOINT0_MEDUSA = np.array([JOINT_PUSHED0[7:14]]).flatten()
# JOINT0 = np.zeros(7)

if __name__ == '__main__':
    meshcat = StartMeshcat()
    scenario_file = "../config/bimanual_med_hardware.yaml"
    directives_file = "../config/bimanual_med.yaml"
    
    root_builder = DiagramBuilder()
    
    hardware_diagram, hardware_plant = create_hardware_diagram_plant_bimanual(scenario_filepath=scenario_file, meshcat=meshcat, position_only=True)
    vis_diagram = create_visual_diagram(directives_filepath=directives_file, meshcat=meshcat, package_file="../package.xml")
    
    context = hardware_diagram.CreateDefaultContext()
    hardware_diagram.ExecuteInitializationEvents(context)
    curr_thanos = hardware_diagram.GetOutputPort("iiwa_thanos.position_commanded").Eval(context)
    curr_medusa = hardware_diagram.GetOutputPort("iiwa_medusa.position_commanded").Eval(context)
    if np.linalg.norm(curr_thanos - JOINT0_THANOS) > 1e-3:
        raise ValueError("Initial joint configuration for Thanos is not correct. WRONG SENSING")
    if np.linalg.norm(curr_medusa - JOINT0_MEDUSA) > 1e-3:
        raise ValueError("Initial joint configuration for Medusa is not correct. WRONG SENSING")
    
    hardware_block = root_builder.AddSystem(hardware_diagram)
    
    plant_arms = MultibodyPlant(1e-3) # time step
    load_iiwa_setup(plant_arms, package_file='../package.xml', directive_path="../config/bimanual_med.yaml")
    plant_arms.Finalize()
    
    plant_context = plant_arms.CreateDefaultContext()
    plant_arms.SetPositions(plant_context, plant_arms.GetModelInstanceByName("iiwa_thanos"), JOINT_PUSHED0[:7])
    plant_arms.SetPositions(plant_context, plant_arms.GetModelInstanceByName("iiwa_medusa"), JOINT_PUSHED0[7:14])
    
    left_pose0 = plant_arms.EvalBodyPoseInWorld(plant_context, plant_arms.GetBodyByName("iiwa_link_7", plant_arms.GetModelInstanceByName("iiwa_thanos")))
    right_pose0 = plant_arms.EvalBodyPoseInWorld(plant_context, plant_arms.GetBodyByName("iiwa_link_7", plant_arms.GetModelInstanceByName("iiwa_medusa")))
    gap = GAP
    
    object_pose0 = RigidTransform(left_pose0.rotation().ToQuaternion(), left_pose0.translation() + left_pose0.rotation().matrix() @ np.array([0,0,(gap)/2.0]))
    
    desired_obj2left_se2 = np.array([0.00, 0.03, 0.0])
    desired_obj2right_se2 = np.array([0.03, 0.00, np.pi])
    
    ts, left_poses, right_poses, obj_poses = run_full_inhand_og(desired_obj2left_se2, desired_obj2right_se2, left_pose0, right_pose0, object_pose0, rotation=np.pi/3, rotate_steps=10, rotate_time=20.0, se2_time=10.0, back_time=15.0, fix_right=False)
    left_piecewise, right_piecewise, _ = piecewise_traj(ts, left_poses, right_poses, obj_poses)
    T = ts[-1]    
    ts = np.linspace(0, T, 1_000)
    seed_q0 = JOINT_PUSHED0
    qs = solve_ik_inhand(plant_arms, ts, left_piecewise, right_piecewise, "thanos_finger", "medusa_finger", seed_q0)
    qs_thanos = qs[:, :7]
    qs_medusa = qs[:, 7:14]
    
    traj_medusa = PiecewisePolynomial.FirstOrderHold(ts, qs_medusa.T)
    traj_thanos = PiecewisePolynomial.FirstOrderHold(ts, qs_thanos.T)
    
    if np.linalg.norm(qs_thanos[0, :] - JOINT0_THANOS) > 1e-3:
        raise ValueError("Initial joint configuration for Thanos is not correct.")
    if np.linalg.norm(qs_medusa[0, :] - JOINT0_MEDUSA) > 1e-3:
        raise ValueError("Initial joint configuration for Medusa is not correct.")
    
    
    traj_medusa_block = root_builder.AddSystem(TrajectorySource(traj_medusa))
    root_builder.Connect(traj_medusa_block.get_output_port(), hardware_block.GetInputPort("iiwa_medusa.position"))
    root_builder.Connect(traj_medusa_block.get_output_port(), hardware_block.GetInputPort("iiwa_medusa_fake.position"))
    
    traj_thanos_block = root_builder.AddSystem(TrajectorySource(traj_thanos))
    root_builder.Connect(traj_thanos_block.get_output_port(), hardware_block.GetInputPort("iiwa_thanos.position"))
    root_builder.Connect(traj_thanos_block.get_output_port(), hardware_block.GetInputPort("iiwa_thanos_fake.position"))
    
    root_diagram = root_builder.Build()
    
    input("Press Enter to start the simulation.")
    # run simulation
    simulator = Simulator(root_diagram)
    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(T + 5.0)