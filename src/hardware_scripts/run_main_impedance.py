import numpy as np

from pydrake.all import (
    StartMeshcat,
    Simulator,
    DiagramBuilder,
    TrajectorySource,
    Multiplexer,
    Demultiplexer,
    RigidTransform,
    MultibodyPlant,
    DiagramBuilder,
    PiecewisePolynomial
)
import numpy as np

import sys
sys.path.append('..')
from diagrams import create_hardware_diagram_plant_bimanual, create_visual_diagram
from planning.ik_util import solve_ik_inhand, run_full_inhand_og, piecewise_traj
from data_record import BenchmarkController
from test_impedance import goto_and_torque
from planning.object_compensation import ApplyForce
from load.sim_setup import load_iiwa_setup
from run_plan_main import curr_joints, goto_joints

# JOINT_CONFIG0 = [0.2629627380321955, 0.6651758641535246, 0.7157398241465858, -1.9204422808541055, 2.1896118717866164, 0.8246912707270445, -1.3312565995665175, -1.4523627309516682, 0.7165811053720673, 0.805679937637571, -1.876561512010483, 0.6976893656839942, -1.3458728960727322, 0.7420561347553449]
JOINT_CONFIG0 = [0.17080075882456985, 0.7238722374425697, 0.8492805114836889, -1.9204438843365204, 2.1829845334586073, 0.7533481390560287, -1.402310073825533, -1.294071627186691, 0.634174042636141, 0.5807367784216141, -1.8765631835542942, 0.7622053135453455, -1.2557337334531637, 0.8464526131666775]
JOINT0_THANOS = np.array([JOINT_CONFIG0[:7]]).flatten()
JOINT0_MEDUSA = np.array([JOINT_CONFIG0[7:14]]).flatten()

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

def follow_trajectory_and_torque(traj_thanos, traj_medusa, force = 30.0, endtime = 1e12):
    meshcat = StartMeshcat()
    scenario_file = "../../config/bimanual_med_hardware_impedance.yaml"
    directives_file = "../../config/bimanual_med.yaml"
        
    root_builder = DiagramBuilder()
    
    hardware_diagram, hardware_plant = create_hardware_diagram_plant_bimanual(scenario_filepath=scenario_file, position_only=False, meshcat = meshcat)
    vis_diagram = create_visual_diagram(directives_filepath=directives_file, meshcat=meshcat, package_file="../../package.xml")
    
    hardware_block = root_builder.AddSystem(hardware_diagram)
    recorder = BenchmarkController(hardware_plant)
    recorder_block = root_builder.AddSystem(recorder)
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
    
    # code specific to connecting to BenchmarkController
    multiplexer_block_target = root_builder.AddSystem(Multiplexer([7,7]))
    root_builder.Connect(traj_thanos_block.get_output_port(), multiplexer_block_target.get_input_port(0))
    root_builder.Connect(traj_medusa_block.get_output_port(), multiplexer_block_target.get_input_port(1))
    root_builder.Connect(multiplexer_block_target.get_output_port(), recorder_block.GetInputPort("target"))
    
    multiplexer_block_measured = root_builder.AddSystem(Multiplexer([7,7]))
    root_builder.Connect(hardware_block.GetOutputPort("iiwa_thanos.position_measured"), multiplexer_block_measured.get_input_port(0))
    root_builder.Connect(hardware_block.GetOutputPort("iiwa_medusa.position_measured"), multiplexer_block_measured.get_input_port(1))
    root_builder.Connect(multiplexer_block_measured.get_output_port(), recorder_block.GetInputPort("measure"))
    
    root_diagram = root_builder.Build()

    simulator = Simulator(root_diagram)
    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(endtime + 5.0)
    
    return recorder

if __name__ == '__main__':
    curr_q = curr_joints()
    des_q = JOINT_CONFIG0
    
    curr_q_thanos = curr_q[:7]
    curr_q_medusa = curr_q[7:14]
    
    joint_speed = 5.0 * np.pi / 180.0 # 1 degree per second
    thanos_displacement = np.max(np.abs(des_q[:7] - curr_q[:7]))
    thanos_endtime = thanos_displacement / joint_speed
    
    medusa_displacement = np.max(np.abs(des_q[7:14] - curr_q[7:14]))
    medusa_endtime = medusa_displacement / joint_speed
    
    print("medusa_endtime: ", medusa_endtime)
    print("thanos_endtime: ", thanos_endtime)
    des_q_thanos = JOINT0_THANOS.copy()
    des_q_medusa = JOINT0_MEDUSA.copy()
    
    input("Press Enter to reset medusa arm.")
    goto_joints(curr_q_thanos, des_q_medusa, endtime = medusa_endtime)
    input("Press Enter to reset thanos arm.")
    goto_joints(des_q_thanos, des_q_medusa, endtime = thanos_endtime)
    
    input("Press Enter to press fingers together")
    force = 30.0
    wrench_medusa = np.array([0, 0, 0, 0, -force, 0.0])
    wrench_thanos = np.array([0, 0, 0, 0, force, 0.0])
    goto_and_torque(des_q[:7], des_q[7:], wrench_thanos, wrench_medusa, endtime = 5.0)
    
    
    traj_medusa, traj_thanos, T = generate_trajectory(des_q, rotation=np.pi/2, desired_obj2left_se2 = np.array([0.00, -0.03, 0.0]), desired_obj2right_se2 = np.array([0.00, 0.03, np.pi]))
    input("Press Enter to follow trajectory")
    recorder = follow_trajectory_and_torque(traj_thanos, traj_medusa, force = force,endtime = T)
    