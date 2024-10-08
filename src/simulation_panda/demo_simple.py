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
from pydrake.all import PiecewisePolynomial

from planning.ik_util_panda import solve_ik_inhand, piecewise_joints, run_full_inhand_og, piecewise_traj, run_full_inhand_naive
from record_lib import RecordPoses, ContactForceReporter
from arm_utils import ApplyForce, CombineArmStates
from load.panda_load import load_bimanual_setup



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
        self.target_se2_right = target_se2_right * np.array([-1,1,-1])
        self.object_pose = RigidTransform(RollPitchYaw(3.14, -1.58, 0.86), [0.46, -0.04, 0.48])
        
        self.qstart = self.seed_q0
        self.use_custom_object = use_custom_object
        self.rotation = rotation
        
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
            obj_mass = 0.3
            if self.object_name == "square":
                self.object = shape_lib.AddBox(plant, "object", lwh=(self.box_width*8, self.box_width*8,self.box_width), mass=obj_mass, mu=1.0, color=[0,0,1,0.3])
            else:
                self.object = shape_lib.AddCylinder(plant, self.box_width*8/2, self.box_width, name="object", mass = obj_mass, mu=1.0, color=[0,0,1,0.3])
            
        
        # self.object = shape_lib.AddBox(plant, "object", lwh=(0.25, 0.25,self.box_width), mass=obj_mass, mu=2.0, color=[0,0,1,0.3])
        
        #setup ground
        shape_lib.AddGround(plant)
        
        #load bimanual and finalize
        load_bimanual_setup(plant, scene_graph)
        
        #setup virtual arms
        plant_arms = MultibodyPlant(1e-3) # time step
        load_bimanual_setup(plant_arms)
        
        # generate inhand 3d pose trajectory
        object_pose0 = self.object_pose
        left_pose0 = RigidTransform(object_pose0.rotation().ToQuaternion(), object_pose0.translation() + object_pose0.rotation().matrix() @ np.array([0,0,-self.box_width/2.0 - finger_length/2.0]))
        right_pose0 = RigidTransform(object_pose0.rotation().ToQuaternion(), object_pose0.translation() + object_pose0.rotation().matrix() @ np.array([0,0,self.box_width/2.0 + finger_length/2.0]))
        ts, left_poses, right_poses, obj_poses = run_full_inhand_naive(self.target_se2_left, self.target_se2_right, left_pose0, right_pose0, object_pose0, rotation = (90 - self.rotation) * np.pi/180)
        # ts, left_poses, right_poses, obj_poses = run_full_inhand_og(self.target_se2_left, self.target_se2_right, left_pose0, right_pose0, object_pose0, rotation = (90 - self.rotation) * np.pi/180)
        left_piecewise, right_piecewise, object_piecewise = piecewise_traj(ts, left_poses, right_poses, obj_poses)
        # then solve ik for joint trajectory
        T = ts[-1]
        print("T: ", T)
        self.runtime = T + 1.0
        ts = np.linspace(0, T, 1_000)
        qs = solve_ik_inhand(plant_arms, ts, left_piecewise, right_piecewise, "left_finger", "right_finger", self.seed_q0)
        q_piecewise = piecewise_joints(ts, qs)
        
        
        num_franka_joints = plant_arms.num_positions()
        kp = 800*np.ones(num_franka_joints)
        ki = 1*np.ones(num_franka_joints)
        kd = 4*np.sqrt(kp)
        controller_block = builder.AddSystem(InverseDynamicsController(plant_arms, kp, ki, kd, False))
        
        feedforward_dual_block = builder.AddSystem(ApplyForce(plant_arms, obj_mass, force=20.0))
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
def run_test(input_arg):
    object = "circle"
    ANGLES = np.array([20, 30, 45, 60])
    PATHS = [
        (np.array([0.05,0,0]), np.array([-0.02,0,-np.pi/2])), # for some reason final right is 0.04 and final left is 0.00 for positional
        (np.array([-0.05,0,0]), np.array([0.05,0,0])),
        (np.array([0.00,0,np.pi/2]), np.array([0.00,0,-np.pi/6])),
    ]
    angle_idx = input_arg[0]
    path_idx = input_arg[1]    
    
    test = ArmStation(use_custom_object=False, rotation=ANGLES[angle_idx], target_se2_left=PATHS[path_idx][0], target_se2_right=PATHS[path_idx][1], object=object)
    test.run(1e4)

    object2left_data = test.pub.object2left_data
    object2right_data = test.pub.object2right_data
    
    final_object2left = object2left_data[-1]
    final_object2right = object2right_data[-1]
    
    desired_obj2left = PATHS[path_idx][0]
    desired_obj2right = PATHS[path_idx][1]
    
    angles = ANGLES[angle_idx]
    if not os.path.exists(f"data/naive/open_loop/{object}/even"):
        os.makedirs(f"data/naive/open_loop/{object}/even")
    np.save(f"data/naive/open_loop/{object}/even/naive_angle_{angles}_path_{path_idx}_MSE.npy", np.array([final_object2left, final_object2right, desired_obj2left, desired_obj2right]))

if __name__ == '__main__':
    from multiprocessing import Pool
    cores = 14
    # get every combination of angle indices and path indices
    pool = Pool(cores)
    pts = np.mgrid[0:4, 0:3].reshape(2, -1)
    pool.map(run_test, pts.T)
    print("Done")