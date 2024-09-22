import sys
sys.path.append("..")
import numpy as np
from pydrake.systems.framework import DiagramBuilder
from manipulation.scenarios import AddMultibodyTriad
from pydrake.math import RollPitchYaw, RigidTransform
from pydrake.visualization import AddDefaultVisualization

from load.workstation import WorkStation
from pydrake.multibody.plant import MultibodyPlant
from pydrake.systems.controllers import InverseDynamicsController
from pydrake.systems.framework import LeafSystem
import load.shape_lib as shape_lib
from pydrake.systems.primitives import Adder
from pydrake.all import PiecewisePolynomial, AddMultibodyPlant, MultibodyPlantConfig

from planning.ik_util_panda import solve_ik_inhand, piecewise_joints, run_full_inhand, piecewise_traj, solve_ik_inhand_no_collision
from record_lib import RecordPoses, ContactForceReporter
from arm_utils import ApplyForce, CombineArmStates
from load.panda_load import load_bimanual_setup, load_bimanual_custom
from planning.drake_inhand_planner2 import DualLimitSurfaceParams, inhand_planner

class IIwaFollower(LeafSystem):
    def __init__(self, qs_piecewise: PiecewisePolynomial):
        LeafSystem.__init__(self)
        self.qs_piecewise = qs_piecewise
        self.DeclareVectorOutputPort("arm_targets", 28, self.CalcOutput)
    def CalcOutput(self, context, output):
        t = context.get_time()
        q = self.qs_piecewise.value(t) #size 14
        dq = self.qs_piecewise.derivative(1).value(t) #size 14
        qd = np.concatenate((q,dq))
        output.SetFromVector(qd)

class ArmStation(WorkStation):
    def __init__(self, use_custom_object = False, rotation = 70 *np.pi/180, target_se2_left = np.array([0,0,0]), target_se2_right = np.array([0,0,0]), horizon = 7, object="square"):
        WorkStation.__init__(self, "hydroelastic_with_fallback", multibody_dt=1e-3,penetration_allowance=1e-3,visual=True)
        self.object_name = object
        self.seed_q0 = np.array([-0.41819804, -0.3893826, 0.1001545, -2.50638261, 1.81897449, 1.8743886, -2.15126654, 
                                0.0484422, -0.21649935, 0.48641599, -2.30843626, -1.65661519, 1.91091385, -0.99507908])
        
        self.target_se2_left  = target_se2_left
        self.target_se2_right = target_se2_right * np.array([-1,1,1])
        self.object_pose = RigidTransform(RollPitchYaw(3.14, -1.58, 0.86), [0.46, -0.04, 0.48])
        
        self.qstart = self.seed_q0
        self.use_custom_object = use_custom_object
        self.rotation = rotation
        
        self.T = 5.0
        self.horizon = horizon
    def setup_simulate(self, builder: DiagramBuilder, plant: MultibodyPlant, scene_graph):
        plant.mutable_gravity_field().set_gravity_vector([0, 0, -9.83])
        
        #load object
        if self.use_custom_object:
            finger_length = 0.03
            self.box_width = 0.0025
            obj_mass = 0.2
            self.object = shape_lib.AddCustomObject(plant, "object", 13*self.box_width, self.box_width, 0.6, obj_mass)
        else:
            finger_length = 0.03
            self.box_width = 0.005
            obj_mass = 0.5
            
            if self.object_name == "square":
                self.object = shape_lib.AddBox(plant, "object", lwh=(self.box_width*8, self.box_width*8,self.box_width), mass=obj_mass, mu=1.0, color=[0,0,1,1.0])
            else:
                self.object = shape_lib.AddCylinder(plant, self.box_width*8/2, self.box_width, name="object", mass = obj_mass, mu=1.0, color=[0,0,1,0.3])
            
        dls_params = DualLimitSurfaceParams(mu_A = 1.0, r_A = self.box_width, N_A = 20.0, mu_B = 1.0, r_B = self.box_width, N_B = 20.0)
        horizon = self.horizon
        
        builder_dead = DiagramBuilder()
        plant_arms, scene_graph_arms = AddMultibodyPlant(MultibodyPlantConfig(), builder_dead)
        load_bimanual_custom(plant_arms, scene_graph_arms)
        diagram = builder_dead.Build()
        diagram_context = diagram.CreateDefaultContext()
        plant_arm_context = diagram.GetMutableSubsystemContext(plant_arms, diagram_context)
        
        plant_arms_arms = MultibodyPlant(1e-3)
        load_bimanual_setup(plant_arms_arms)
        
        
        current_obj2left_se2 = np.array([0,0,0])
        current_obj2right_se2 = np.array([0,0,0])
        desired_obj2left_se2, desired_obj2right_se2 = self.target_se2_left, self.target_se2_right
        
        obj2left, obj2right, vs = inhand_planner(current_obj2left_se2, current_obj2right_se2, desired_obj2left_se2, desired_obj2right_se2, dls_params, steps = horizon, angle = 60.0, palm_radius=0.07, kv = 0.5, is_panda=True)
        
        print(np.round(desired_obj2left_se2 - obj2left[:,-1],4))
        print(np.round(desired_obj2right_se2 - obj2right[:,-1],4))
        
        print(np.round(obj2left,4))
        print(np.round(obj2right,4))
        
        desired_obj2left_se2s = []
        desired_obj2right_se2s = []
        for i in range(1,horizon):
            if i % 2 == 1:
                desired_obj2left_se2s.append(obj2left[:,i])
            else:
                desired_obj2right_se2s.append(obj2right[:,i] * np.array([1,1,-1]))
            
        #setup ground
        shape_lib.AddGround(plant)
        
        #load bimanual and finalize
        load_bimanual_setup(plant, scene_graph)
        
        
        plant.Finalize()
        
        # generate inhand 3d pose trajectory
        object_pose0 = self.object_pose
        left_pose0 = RigidTransform(object_pose0.rotation().ToQuaternion(), object_pose0.translation() + object_pose0.rotation().matrix() @ np.array([0,0,-self.box_width/2.0 - finger_length/2.0]))
        right_pose0 = RigidTransform(object_pose0.rotation().ToQuaternion(), object_pose0.translation() + object_pose0.rotation().matrix() @ np.array([0,0,self.box_width/2.0 + finger_length/2.0]))
        
        ts, left_poses, right_poses, obj_poses = run_full_inhand(desired_obj2left_se2s, desired_obj2right_se2s, left_pose0, right_pose0, object_pose0, rotation= (90-self.rotation) * np.pi/180, rotate_steps=40, rotate_time=0.1, se2_time=3.0, back_time=0.1, fix_right=False)
        left_piecewise, right_piecewise, object_piecewise = piecewise_traj(ts, left_poses, right_poses, obj_poses)
        # then solve ik for joint trajectory
        T = ts[-1]
        self.runtime = T + 0.5
        print("T: ", T)
        ts = np.linspace(0, T, 1_000)
        qs = solve_ik_inhand_no_collision(plant_arms, plant_arm_context, ts, left_piecewise, right_piecewise, "left_finger", "right_finger", self.seed_q0)
        
        q_piecewise = piecewise_joints(ts, qs)
        
        
        num_franka_joints = plant_arms.num_positions()
        kp = 800*np.ones(num_franka_joints)
        ki = 1*np.ones(num_franka_joints)
        kd = 4*np.sqrt(kp)
        controller_block = builder.AddSystem(InverseDynamicsController(plant_arms_arms, kp, ki, kd, False))
        
        feedforward_dual_block = builder.AddSystem(ApplyForce(plant_arms, obj_mass, force=30.0))
        adder_torque_block = builder.AddSystem(Adder(2, 14))
        arm_states_block = builder.AddSystem(CombineArmStates())
    
        iiwa_follower = builder.AddSystem(IIwaFollower(q_piecewise))
        
        franka_left_instance = plant.GetModelInstanceByName("franka_left")
        franka_right_instance = plant.GetModelInstanceByName("franka_right")
        object_instance = plant.GetModelInstanceByName("object")    
        
        self.pub = RecordPoses(plant)
        pub = builder.AddSystem(self.pub)
        self.contact_reporter = builder.AddSystem(ContactForceReporter(offset=0.0, period=0.1))
        
        builder.Connect(iiwa_follower.get_output_port(0), controller_block.get_input_port_desired_state())
        builder.Connect(controller_block.get_output_port(), adder_torque_block.get_input_port(0))
        builder.Connect(feedforward_dual_block.get_output_port(), adder_torque_block.get_input_port(1))
        builder.Connect(adder_torque_block.get_output_port(), plant.get_actuation_input_port())
        
        builder.Connect(plant.get_state_output_port(franka_left_instance), arm_states_block.get_input_port(0))
        builder.Connect(plant.get_state_output_port(franka_right_instance), arm_states_block.get_input_port(1))
        builder.Connect(arm_states_block.get_output_port(), controller_block.get_input_port_estimated_state())
        
        builder.Connect(plant.get_state_output_port(franka_left_instance), feedforward_dual_block.get_input_port(0))
        builder.Connect(plant.get_state_output_port(franka_right_instance), feedforward_dual_block.get_input_port(1))
        builder.Connect(plant.get_state_output_port(object_instance), feedforward_dual_block.get_input_port(2))
        
        builder.Connect(plant.get_state_output_port(franka_left_instance), pub.get_input_port(0))
        builder.Connect(plant.get_state_output_port(franka_right_instance), pub.get_input_port(1))
        builder.Connect(plant.get_state_output_port(object_instance), pub.get_input_port(2))
        builder.Connect(plant.get_contact_results_output_port(), self.contact_reporter.get_input_port())
        
        AddDefaultVisualization(builder, self.meshcat)
        
        AddMultibodyTriad(plant.GetFrameByName("left_finger"), scene_graph, length=0.03, radius=0.003)
        AddMultibodyTriad(plant.GetFrameByName("right_finger"), scene_graph, length=0.03, radius=0.003)
        AddMultibodyTriad(plant.GetFrameByName("object_body"), scene_graph, length=0.03, radius=0.003)
        
        
        self.qstart = q_piecewise.value(0)
    def initialize_simulate(self, plant: MultibodyPlant, plant_context):
        plant.SetPositions(plant_context, plant.GetModelInstanceByName("franka_left"), self.qstart[:7])
        plant.SetPositions(plant_context, plant.GetModelInstanceByName("franka_right"), self.qstart[7:14])
        plant.SetFreeBodyPose(plant_context, plant.GetBodyByName("object_body"), self.object_pose)
import os
if __name__ == '__main__':
    ANGLES = np.array([20, 30, 45, 60])
    PATHS = [
        (np.array([0.055,0,0]), np.array([0.00,0,-np.pi/2])), # for some reason final right is 0.04 and final left is 0.00 for positional
        (np.array([-0.05,0,0]), np.array([0.05,0,0])),
        (np.array([0.00,0,np.pi/4]), np.array([0.00,0,-np.pi/6])),
    ]
    angle_idx = 3
    path_idx = 0
    
    if path_idx == 0:
        horizon = 5
    elif path_idx == 2:
        horizon = 7
    else:
        horizon = 3
    
    test = ArmStation(use_custom_object=False, rotation=ANGLES[angle_idx], target_se2_left=PATHS[path_idx][0], target_se2_right=PATHS[path_idx][1], horizon=horizon, object="square")
    test.run(1e4)
    
    object2left_data = test.pub.object2left_data
    object2right_data = test.pub.object2right_data
    
    desired_obj2left = PATHS[path_idx][0]
    desired_obj2right = PATHS[path_idx][1]
    
    
    angles = ANGLES[angle_idx]
    if not os.path.exists(f"data/figure/open_loop/algo/"):
        os.makedirs(f"data/figure/open_loop/algo/")
    np.save(f"data/figure/open_loop/algo/desired_obj2left.npy", desired_obj2left)
    np.save(f"data/figure/open_loop/algo/desired_obj2right.npy", desired_obj2right)
    np.save(f"data/figure/open_loop/algo/obj2left_data.npy", object2left_data)
    np.save(f"data/figure/open_loop/algo/obj2right_data.npy", object2right_data)