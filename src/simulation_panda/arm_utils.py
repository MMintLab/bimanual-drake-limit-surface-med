###################################################################
# utils.py
#
# 
###################################################################

from pydrake.systems.framework import LeafSystem
from pydrake.multibody.plant import MultibodyPlant
from pydrake.math import RotationMatrix, RollPitchYaw, RigidTransform
from pydrake.common.eigen_geometry import Quaternion
from pydrake.systems.framework import LeafSystem
from pydrake.multibody.inverse_kinematics import InverseKinematics, GlobalInverseKinematics
from pydrake.solvers import Solve, GetAvailableSolvers, GetProgramType, NloptSolver, IpoptSolver, SnoptSolver
from pydrake.all import JacobianWrtVariable
import numpy as np


#################################
# Global Solver
#################################
#NOTE: too slow
def solveGlobalIK(plant: MultibodyPlant, target_pose: RigidTransform, frame_name: str, q0=1e-10*np.ones(14)):
    ik = GlobalInverseKinematics(plant)
    #get body index
    body_index = plant.GetBodyByName(frame_name).index()
    
    ik.AddWorldPositionConstraint(
        body_index,
        np.array([0,0,0]),
        target_pose.translation(),
        target_pose.translation()
    )
    
    ik.AddWorldOrientationConstraint(
        body_index,
        Quaternion(target_pose.rotation().matrix()),
        angle_tol=1e-3
    )
    
    prog = ik.get_mutable_prog()
    result = Solve(prog)
    
    #NOTE: did not add extra cost
    #NOTE: did not set initial guess
    
    print("Solving IK")
    if result.is_success():
        print("Finished solving IK")
        print(result.GetSolution())
        print(result.GetSolution().shape)
        return result.GetSolution()
    else:
        print("Failed to solve IK")
        return result.GetSolution()
    

def solveDualIK2(plant: MultibodyPlant, left_pose: RigidTransform, right_pose: RigidTransform, left_frame_name: str, right_frame_name: str, q0=1e-10*np.ones(14)):
    ik = InverseKinematics(plant, with_joint_limits=True)
    ik.AddPositionConstraint(
        plant.GetFrameByName(left_frame_name),
        np.array([0,0,0]),
        plant.world_frame(),
        left_pose.translation(),
        left_pose.translation())
    ik.AddOrientationConstraint(
        plant.GetFrameByName(left_frame_name),
        RotationMatrix(RollPitchYaw(0,0,0)),
        plant.world_frame(),
        left_pose.rotation(),
        0.0
    )
    ik.AddPositionConstraint(
        plant.GetFrameByName(right_frame_name),
        np.array([0,0,0]),
        plant.world_frame(),
        right_pose.translation(),
        right_pose.translation())
    ik.AddOrientationConstraint(
        plant.GetFrameByName(right_frame_name),
        RotationMatrix(RollPitchYaw(0,0,0)),
        plant.world_frame(),
        right_pose.rotation(),
        0.0
    )
    prog = ik.get_mutable_prog()
    q = ik.q()
    if np.abs(q0).sum() > 1e-4:
        prog.AddQuadraticErrorCost(np.eye(14),q0,q)
    prog.SetInitialGuess(q,q0)
    solver = SnoptSolver()
    result = solver.Solve(prog)
    return result.GetSolution(), result.is_success()

def solveDualIK(plant: MultibodyPlant, left_pose: RigidTransform, right_pose: RigidTransform, left_frame_name: str, right_frame_name: str, q0=1e-10*np.ones(14)):
    ik = InverseKinematics(plant, with_joint_limits=True)
    ik.AddPositionConstraint(
        plant.GetFrameByName(left_frame_name),
        np.array([0,0,0]),
        plant.world_frame(),
        left_pose.translation(),
        left_pose.translation())
    ik.AddOrientationConstraint(
        plant.GetFrameByName(left_frame_name),
        RotationMatrix(RollPitchYaw(0,0,0)),
        plant.world_frame(),
        left_pose.rotation(),
        0.0
    )
    ik.AddPositionConstraint(
        plant.GetFrameByName(right_frame_name),
        np.array([0,0,0]),
        plant.world_frame(),
        right_pose.translation(),
        right_pose.translation())
    ik.AddOrientationConstraint(
        plant.GetFrameByName(right_frame_name),
        RotationMatrix(RollPitchYaw(0,0,0)),
        plant.world_frame(),
        right_pose.rotation(),
        0.0
    )
    prog = ik.get_mutable_prog()
    q = ik.q()
    if np.abs(q0).sum() > 1e-4:
        prog.AddQuadraticErrorCost(np.eye(14),q0,q)
    print("Solving IK")
    prog.SetInitialGuess(q,q0)
    solver = SnoptSolver()
    result = solver.Solve(prog)
    if result.is_success():
        print("Finished solving IK")
        print(result.GetSolution())
        return result.GetSolution()
    else:
        print("Failed to solve IK")
        return result.GetSolution()


def solveDualIKplusplus(plant: MultibodyPlant, left_pose: RigidTransform, right_pose: RigidTransform, left_frame_name: str, right_frame_name: str, q0=1e-10*np.ones(14), gap=0.035):
    ik = InverseKinematics(plant, with_joint_limits=True)
    
    #ensure both arms are parallel in the optimization
    ik.AddAngleBetweenVectorsConstraint(
        frameA=plant.GetFrameByName(left_frame_name),
        na_A=np.array([[0,0,1]]).T,
        frameB=plant.GetFrameByName(right_frame_name),
        nb_B=np.array([[0,0,1]]).T,
        angle_lower=np.pi,
        angle_upper=np.pi
    )
    
    #ensure both arms have a gap between them in z direction
    ik.AddPositionConstraint(
        plant.GetFrameByName(right_frame_name),
        np.array([0,0,0]),
        plant.GetFrameByName(left_frame_name),
        np.array([-10,-10,gap]),
        np.array([10,10,gap])
    )
    ik.AddPositionConstraint(
        plant.GetFrameByName(left_frame_name),
        np.array([0,0,0]),
        plant.GetFrameByName(right_frame_name),
        np.array([-10,-10,gap]),
        np.array([10,10,gap])
    )
    
    ik.AddPositionConstraint(
        plant.GetFrameByName(left_frame_name),
        np.array([0,0,0]),
        plant.world_frame(),
        left_pose.translation(),
        left_pose.translation())
    ik.AddOrientationConstraint(
        plant.GetFrameByName(left_frame_name),
        RotationMatrix(RollPitchYaw(0,0,0)),
        plant.world_frame(),
        left_pose.rotation(),
        0.0
    )
    ik.AddPositionConstraint(
        plant.GetFrameByName(right_frame_name),
        np.array([0,0,0]),
        plant.world_frame(),
        right_pose.translation(),
        right_pose.translation())
    ik.AddOrientationConstraint(
        plant.GetFrameByName(right_frame_name),
        RotationMatrix(RollPitchYaw(0,0,0)),
        plant.world_frame(),
        right_pose.rotation(),
        0.0
    )
    prog = ik.get_mutable_prog()
    q = ik.q()
    if np.abs(q0).sum() > 1e-4:
        prog.AddQuadraticErrorCost(np.eye(14),q0,q)
    prog.SetInitialGuess(q,q0)
    solver = SnoptSolver()
    result = solver.Solve(prog)
    if result.is_success():
        return result.GetSolution()
    else:
        print("Failed to solve solveDualIKplusplus")
        return result.GetSolution()

def solveIK_Follower(plant: MultibodyPlant, pose: RigidTransform, left_frame_name: str, right_frame_name: str, q0=1e-10*np.ones(14), gap=0.035, left=False):
    '''
        given a pose and specified whether left or right arm moves
        
        move one arm to specified pose.
        other arm will follow the first arm and maintain relative transform
    '''
    
    ik = InverseKinematics(plant, with_joint_limits=True)
    
    #ensure that the arm is parallel to the other arm
    ik.AddAngleBetweenVectorsConstraint(
        frameA=plant.GetFrameByName(left_frame_name),
        na_A=np.array([[0,0,1]]).T,
        frameB=plant.GetFrameByName(right_frame_name),
        nb_B=np.array([[0,0,1]]).T,
        angle_lower=np.pi,
        angle_upper=np.pi
    )
    #ensure both arms have a gap between them in z direction
    ik.AddPositionConstraint(
        plant.GetFrameByName(right_frame_name),
        np.array([0,0,0]),
        plant.GetFrameByName(left_frame_name),
        np.array([-10,-10,gap]),
        np.array([10,10,gap])
    )
    ik.AddPositionConstraint(
        plant.GetFrameByName(left_frame_name),
        np.array([0,0,0]),
        plant.GetFrameByName(right_frame_name),
        np.array([-10,-10,gap]),
        np.array([10,10,gap])
    )
    
    if left:
        leader_frame = left_frame_name
        follower_frame = right_frame_name
    else:
        leader_frame = right_frame_name
        follower_frame = left_frame_name
        
    #move leader arm to specified pose
    ik.AddPositionConstraint(
        plant.GetFrameByName(leader_frame),
        np.array([0,0,0]),
        plant.world_frame(),
        pose.translation(),
        pose.translation()
    )
    ik.AddOrientationConstraint(
        plant.GetFrameByName(leader_frame),
        RotationMatrix(RollPitchYaw(0,0,0)),
        plant.world_frame(),
        pose.rotation(),
        0.0
    )
    
    #follower arm should follow leader arm and hold relative transform in SE(2)
    plant_context = plant.CreateDefaultContext()
    plant.SetPositions(plant_context, q0)
    leader_pose = plant.GetFrameByName(leader_frame).CalcPoseInWorld(plant_context)
    follower_pose = plant.GetFrameByName(follower_frame).CalcPoseInWorld(plant_context)
    follower2leader = leader_pose.inverse() @ follower_pose # relative transform from follower to target
    yaw_follower2leader = RotationMatrix.MakeZRotation(RollPitchYaw(follower2leader.rotation()).yaw_angle() + np.pi)
    
    ik.AddPositionConstraint(
        plant.GetFrameByName(follower_frame),
        np.array([0,0,0]),
        plant.GetFrameByName(leader_frame),
        np.array([follower2leader.translation()[0], follower2leader.translation()[1], gap]),
        np.array([follower2leader.translation()[0], follower2leader.translation()[1], gap])
    )
    ik.AddOrientationConstraint(
        plant.GetFrameByName(follower_frame),
        RotationMatrix(RollPitchYaw(0,0,0)),
        plant.GetFrameByName(leader_frame),
        yaw_follower2leader @ RotationMatrix.MakeYRotation(np.pi),
        0.0
    )
    
    prog = ik.get_mutable_prog()
    q = ik.q()
    if np.abs(q0).sum() > 1e-4:
        prog.AddQuadraticErrorCost(np.eye(14),q0,q)
    prog.SetInitialGuess(q,q0)
    solver = SnoptSolver()
    result = solver.Solve(prog)
    if result.is_success():
        return result.GetSolution()
    else:
        print("Failed to solve solveIK_Follower")
        return q0

def solveIK_SE2(plant: MultibodyPlant, pose_se2: RigidTransform, left_frame_name: str, right_frame_name: str, q0=1e-10*np.ones(14), gap=0.035, left=False):
    '''
        Move one arm while keeping the other still
    '''
    ik = InverseKinematics(plant, with_joint_limits=True)
    
    #ensure that the arm is parallel to the other arm
    ik.AddAngleBetweenVectorsConstraint(
        frameA=plant.GetFrameByName(left_frame_name),
        na_A=np.array([[0,0,1]]).T,
        frameB=plant.GetFrameByName(right_frame_name),
        nb_B=np.array([[0,0,1]]).T,
        angle_lower=np.pi,
        angle_upper=np.pi
    )
    
    #ensure both arms have a gap between them in z direction
    ik.AddPositionConstraint(
        plant.GetFrameByName(right_frame_name),
        np.array([0,0,0]),
        plant.GetFrameByName(left_frame_name),
        np.array([-10,-10,gap]),
        np.array([10,10,gap])
    )
    ik.AddPositionConstraint(
        plant.GetFrameByName(left_frame_name),
        np.array([0,0,0]),
        plant.GetFrameByName(right_frame_name),
        np.array([-10,-10,gap]),
        np.array([10,10,gap])
    )
    
    if left:
        use_frame = left_frame_name
        still_frame = right_frame_name
    else:
        use_frame = right_frame_name
        still_frame = left_frame_name
    
    #keep still arm still
    plant_context = plant.CreateDefaultContext()
    plant.SetPositions(plant_context, q0)
    still_pose = plant.GetFrameByName(still_frame).CalcPoseInWorld(plant_context)
    ik.AddPositionConstraint(
        plant.GetFrameByName(still_frame),
        np.array([0,0,0]),
        plant.world_frame(),
        still_pose.translation(),
        still_pose.translation()
    )
    ik.AddOrientationConstraint(
        plant.GetFrameByName(still_frame),
        RotationMatrix(RollPitchYaw(0,0,0)),
        plant.world_frame(),
        still_pose.rotation(),
        0.0
    )
    
    # move moving arm
    ts_se2 = np.array([pose_se2.translation()[0], pose_se2.translation()[1], gap])
    ik.AddPositionConstraint(
        plant.GetFrameByName(use_frame),
        np.array([0,0,0]),
        plant.GetFrameByName(still_frame),
        ts_se2,
        ts_se2
    )
    
    ik.AddOrientationConstraint(
        plant.GetFrameByName(use_frame),
        RotationMatrix(RollPitchYaw(0,0,0)),
        plant.GetFrameByName(still_frame),
        pose_se2.rotation() @ RotationMatrix.MakeYRotation(np.pi),
        0.0
    )
    
    prog = ik.get_mutable_prog()
    q = ik.q()
    if np.abs(q0).sum() > 1e-4:
        prog.AddQuadraticErrorCost(np.eye(14),q0,q)
    prog.SetInitialGuess(q,q0)
    solver = SnoptSolver()
    result = solver.Solve(prog)
    if result.is_success():
        return result.GetSolution()
    else:
        print("Failed to solve solveIK_SE2")
        return q0

#################################
# Old Solver for IK
#################################
def solveIK(plant: MultibodyPlant, target_pose: RigidTransform, frame_name: str, q0=1e-10*np.ones(14)):
    ik = InverseKinematics(plant, with_joint_limits=True)
    ik.AddPositionConstraint(
        plant.GetFrameByName(frame_name),
        np.array([0,0,0]),
        plant.world_frame(),
        target_pose.translation(),
        target_pose.translation())
    ik.AddOrientationConstraint(
        plant.GetFrameByName(frame_name),
        RotationMatrix(RollPitchYaw(0,0,0)),
        plant.world_frame(),
        target_pose.rotation(),
        0.0
    )
    prog = ik.get_mutable_prog()
    q = ik.q()
    prog.AddQuadraticErrorCost(np.eye(14),np.ones(14),q)
    
    
    print("Solving IK")
    prog.SetInitialGuess(q,q0)
    result = Solve(prog)
    # solver = IpoptSolver()
    # result = solver.Solve(prog, q0, None)
    
    #get solver details
    
    # available_solvers = GetAvailableSolvers(GetProgramType(prog))
    # print([solver.name() for solver in available_solvers])

    if result.is_success():
        print("Finished solving IK")
        print(result.GetSolution())
        return result.GetSolution()
    else:
        print("Failed to solve IK")
        return result.GetSolution()
    
    
#################################
# Arm Blocks for InHand System
#################################
class ApplyForce(LeafSystem):
    def __init__(self, plant_arms: MultibodyPlant, object_kg = 1.0, object_grav = 9.83, force=17.0):
        LeafSystem.__init__(self)
        self._plant = plant_arms
        self._plant_context = plant_arms.CreateDefaultContext()
        self._right_franka = plant_arms.GetModelInstanceByName("franka_right")
        self._left_franka = plant_arms.GetModelInstanceByName("franka_left")
        
        self._G_right = plant_arms.GetBodyByName("panda_link7", self._right_franka).body_frame()
        self._G_left = plant_arms.GetBodyByName("panda_link7", self._left_franka).body_frame()
        
        self._W = plant_arms.world_frame()
        
        self._force = np.array([0,0,force])
        
        self.left_pose = None
        self.right_pose = None
        
        self.grav_force = np.array([0, 0, -object_kg * object_grav])
        
        self.DeclareVectorInputPort("left_pos", 14)
        self.DeclareVectorInputPort("right_pos", 14)
        self.DeclareVectorInputPort("object_state", 13)
        self.DeclareVectorOutputPort("torque", 14, self.DoCalcOutput)
    
    def DoCalcOutput(self, context, output):
        qleft = self.get_input_port(0).Eval(context)
        qright = self.get_input_port(1).Eval(context)
        object_state = self.get_input_port(2).Eval(context)
        self._plant.SetPositions(self._plant_context, self._right_franka, qright[:7])
        self._plant.SetPositions(self._plant_context, self._left_franka, qleft[:7])
        
        object_state[:4] = object_state[:4] / np.linalg.norm(object_state[:4])
        object_pose = RigidTransform(
            Quaternion(object_state[:4]),
            object_state[4:7]
        )
        '''
            NOTE: Assumptions
            object rotation is aligned with left finger
        '''
        # gravity is world, so rotate from world -> object
        
        z_vector_object = object_pose.rotation().matrix() @ np.array([0,0,1])
        z_dir = z_vector_object[2]
        
        adder_left_force = np.zeros(3)
        adder_right_force = np.zeros(3)
        if z_dir >= 1e-3: # object weight on left finger
            rot_left_finger = self._plant.GetFrameByName("panda_link8", self._left_franka).CalcPoseInWorld(self._plant_context).rotation().matrix()
            adder_left_force = np.abs(rot_left_finger.T @ self.grav_force)
        elif z_dir < -1e-3: # object pushing on right finger
            rot_right_finger = self._plant.GetFrameByName("panda_link8", self._right_franka).CalcPoseInWorld(self._plant_context).rotation().matrix()
            adder_right_force = np.abs(rot_right_finger.T @ self.grav_force)
        else:
            pass
        # if z direction is negative, then it is pushing against left finger
        # if z direction is positive, then it is pushing against right finger
        # additive = np.array([0, 0, np.abs(rotated_grav[2])])
        # adder_left_force = additive if rotated_grav[2] <= 0 else np.array([0, 0, 0])
        # adder_right_force = additive if rotated_grav[2] > 0 else np.array([0, 0, 0])
        
        self.right_pose = self._plant.GetFrameByName("panda_link8", self._right_franka).CalcPoseInWorld(self._plant_context)
        rot_mat_right = self.right_pose.rotation().matrix()
        right_force = rot_mat_right @ (self._force + adder_right_force)
        
        self.left_pose = self._plant.GetFrameByName("panda_link8", self._left_franka).CalcPoseInWorld(self._plant_context)
        rot_mat_left = self.left_pose.rotation().matrix()
        left_force = rot_mat_left @ (self._force + adder_left_force)


        J_G_right = self._plant.CalcJacobianSpatialVelocity(
            self._plant_context,
            JacobianWrtVariable.kQDot,
            self._G_right,
            [0,0,0],
            self._W,
            self._W
        )[:,7:]
        
        J_G_left = self._plant.CalcJacobianSpatialVelocity(
            self._plant_context,
            JacobianWrtVariable.kQDot,
            self._G_left,
            [0,0,0],
            self._W,
            self._W
        )[:,:7]        
        
        if np.isnan(J_G_right[3:,:]).any() or np.linalg.matrix_rank(J_G_right[3:,:]) < 3:
            mat_right = np.zeros((7,3))
        else:
            mat_right = J_G_right[3:,:].T
            
        if np.isnan(J_G_left[3:,:]).any() or np.linalg.matrix_rank(J_G_left[3:,:]) < 3:
            mat_left = np.zeros((7,3))
        else:
            mat_left = J_G_left[3:,:].T
            
        tau_right = mat_right @ right_force
        tau_left = mat_left @ left_force
        
        output.SetFromVector(np.concatenate([tau_left, tau_right]))
        
class CombineArmStates(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        self.DeclareVectorInputPort("left_state", 14)
        self.DeclareVectorInputPort("right_state", 14)
        self.DeclareVectorOutputPort("combined_state", 28, self.DoCalcOutput)
    def DoCalcOutput(self, context, output):
        left_state = self.get_input_port(0).Eval(context)
        right_state = self.get_input_port(1).Eval(context)
        
        pos = np.concatenate([left_state[:7], right_state[:7]])
        vel = np.concatenate([left_state[7:], right_state[7:]])
        output.SetFromVector(np.concatenate([pos, vel]))
        
        
def CalculateRotation(pose: RigidTransform, object_pose: RigidTransform, desired_rotation: float = 0.0) -> RigidTransform:
    '''
    NOTE: hardcode inhand rotation to be in y-axis
    '''
    
    Hobj2world = object_pose.GetAsMatrix4()
    Hee2world = pose.GetAsMatrix4()
    
    # get object->ee
    Hobj2ee = np.linalg.inv(Hee2world) @ Hobj2world
    
    # rotate object to get new object-> world
    rot = RotationMatrix.MakeYRotation(desired_rotation).matrix()
    Hobj2world[:3, :3] = Hobj2world[:3, :3] @ rot # rotate around its y-axis from its current frame
    
    # calculate new ee -> world based on rotated object-> world
    Hee2world = Hobj2world @ np.linalg.inv(Hobj2ee)
    
    return RigidTransform(Hee2world)