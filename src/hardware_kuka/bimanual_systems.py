from pydrake.all import (
    LeafSystem,
    MultibodyPlant,
    JacobianWrtVariable,
    RigidTransform,
    RotationMatrix
)
from camera import CameraManager
from gamma import GammaManager
import numpy as np
from scipy.linalg import block_diag

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
                                                           self._plant.GetBodyByName("iiwa_link_ee_kuka", self._thanos_instance).body_frame(),
                                                           [0,0,0],
                                                           self._plant.world_frame(),
                                                           self._plant.world_frame())[:, :7]
        J_medusa = self._plant.CalcJacobianSpatialVelocity(self._plant_context,
                                                           JacobianWrtVariable.kQDot,
                                                           self._plant.GetBodyByName("iiwa_link_ee_kuka", self._medusa_instance).body_frame(),
                                                           [0,0,0],
                                                           self._plant.world_frame(),
                                                           self._plant.world_frame())[:, 7:]

        thanos_torque = J_thanos.T @ wrench_thanos
        medusa_torque = J_medusa.T @ wrench_medusa
        output.SetFromVector(np.concatenate([thanos_torque, medusa_torque]))
        
class ApplyForceCompensateGravity(LeafSystem):
    def __init__(self, plant_arms: MultibodyPlant, camera_manager: CameraManager, object_kg = 0.5, applied_force = 10.0, feedforward_z_force = 0.0):
        LeafSystem.__init__(self)
        self._plant = plant_arms
        self._plant_context = plant_arms.CreateDefaultContext()
        self._iiwa_thanos = plant_arms.GetModelInstanceByName("iiwa_thanos")
        self._iiwa_medusa = plant_arms.GetModelInstanceByName("iiwa_medusa")
        
        self._G_thanos = plant_arms.GetBodyByName("iiwa_link_ee_kuka", self._iiwa_thanos).body_frame()
        self._G_medusa = plant_arms.GetBodyByName("iiwa_link_ee_kuka", self._iiwa_medusa).body_frame()
        self._W = plant_arms.world_frame()
        
        self._applied_force = np.array([0,0,applied_force]) # additive force
        self.obj_grav_force = np.array([0, 0, -object_kg * 9.83])
        
        self.camera_manager = camera_manager
        
        self.DeclareVectorInputPort("thanos_position", 7)
        self.DeclareVectorInputPort("medusa_position", 7)
        self.DeclareVectorOutputPort("torque", 14, self.DoCalcOutput)
    def DoCalcOutput(self, context, output):
        q_thanos = self.GetInputPort("thanos_position").Eval(context)
        q_medusa = self.GetInputPort("medusa_position").Eval(context)
        self._plant.SetPositions(self._plant_context, self._iiwa_thanos, q_thanos)
        self._plant.SetPositions(self._plant_context, self._iiwa_medusa, q_medusa)
        
        
        obj2medusa_se2 = self.camera_manager.get_medusa_se2()
        
        zvec_thanos_finger = self._plant.GetFrameByName("thanos_finger").CalcPoseInWorld(self._plant_context).rotation().matrix() @ np.array([0,0,1])
        zdir_thanos_finger = zvec_thanos_finger[2]
        
        # adder forces in end-effector frame
        adder_left_wrench = np.zeros(6)
        adder_right_wrench = np.zeros(6)
        if zdir_thanos_finger >= 1e-3: # object weight on thanos finger
            obj2thanos_se2 = self.camera_manager.get_thanos_se2()
            x,y,_ = obj2thanos_se2
            z = 0.135 # distance from thanos finger to object
            
            # get gravity in iiwa_link_ee_kuka frame
            rot_thanos_finger = self._plant.GetFrameByName("thanos_finger").CalcPoseInWorld(self._plant_context).rotation().matrix()
            obj_gravity_eeframe_thanos = rot_thanos_finger.T @ self.obj_grav_force
            
            compensation_force = -obj_gravity_eeframe_thanos
            
            #cross product matrix
            phat = np.array([[0, -z, y],
                             [z, 0, -x],
                             [-y, x, 0]])
            compensation_torque = -(phat @ obj_gravity_eeframe_thanos)
            
            wrench = np.concatenate([compensation_torque, compensation_force])
            adder_left_wrench = wrench
        else:
            obj2medusa_se2 = self.camera_manager.get_medusa_se2()
            x,y,_ = obj2medusa_se2
            z = 0.135
            
            # get gravity in iiwa_link_ee_kuka frame
            rot_medusa_finger = self._plant.GetFrameByName("medusa_finger").CalcPoseInWorld(self._plant_context).rotation().matrix()
            obj_gravity_eeframe_medusa = rot_medusa_finger.T @ self.obj_grav_force
            
            compensation_force = -obj_gravity_eeframe_medusa
            
            #cross product matrix
            phat = np.array([[0, -z, y],
                             [z, 0, -x],
                             [-y, x, 0]])
            compensation_torque = phat @ compensation_force
            wrench = np.concatenate([compensation_torque, compensation_force])
            adder_right_wrench = wrench
        
        thanos_pose_rot = self._plant.GetFrameByName("iiwa_link_ee_kuka", self._iiwa_thanos).CalcPoseInWorld(self._plant_context).rotation().matrix()
        thanos_pose_rot = block_diag(thanos_pose_rot, thanos_pose_rot)
        medusa_pose_rot = self._plant.GetFrameByName("iiwa_link_ee_kuka", self._iiwa_medusa).CalcPoseInWorld(self._plant_context).rotation().matrix()
        medusa_pose_rot = block_diag(medusa_pose_rot, medusa_pose_rot)
        
        adder_left_wrench[3:] += self._applied_force
        adder_right_wrench[3:] += self._applied_force
        
        thanos_wrench = thanos_pose_rot @ adder_left_wrench
        medusa_wrench = medusa_pose_rot @ adder_right_wrench
        
        J_G_thanos = self._plant.CalcJacobianSpatialVelocity(
            self._plant_context,
            JacobianWrtVariable.kQDot,
            self._G_thanos,
            [0,0,0],
            self._W,
            self._W
        )[:,:7]
        
        J_G_medusa = self._plant.CalcJacobianSpatialVelocity(
            self._plant_context,
            JacobianWrtVariable.kQDot,
            self._G_medusa,
            [0,0,0],
            self._W,
            self._W
        )[:,7:]
        
        tau_thanos = J_G_thanos.T @ thanos_wrench
        tau_medusa = J_G_medusa.T @ medusa_wrench
        
        output.SetFromVector(np.concatenate([tau_thanos, tau_medusa]))