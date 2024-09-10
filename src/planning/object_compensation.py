from pydrake.systems.framework import LeafSystem
from pydrake.multibody.plant import MultibodyPlant
from pydrake.systems.framework import LeafSystem
from pydrake.all import JacobianWrtVariable
import numpy as np

class ApplyForce(LeafSystem):
    def __init__(self, plant_arms: MultibodyPlant, object_kg = 0.5, object_grav = 9.83, force= 30.0):
        LeafSystem.__init__(self)
        self._plant = plant_arms
        self._plant_context = plant_arms.CreateDefaultContext()
        self._iiwa_thanos = plant_arms.GetModelInstanceByName("iiwa_thanos")
        self._iiwa_medusa = plant_arms.GetModelInstanceByName("iiwa_medusa")
        
        self._G_thanos = plant_arms.GetBodyByName("iiwa_link_7", self._iiwa_thanos).body_frame()
        self._G_medusa = plant_arms.GetBodyByName("iiwa_link_7", self._iiwa_medusa).body_frame()
        self._W = plant_arms.world_frame()
        
        
        #NOTE: things worked nicely at 30.0 N
        self._force = np.array([0,0,force]) # additive force
        
        self.grav_force = np.array([0, 0, -object_kg * object_grav])
        
        self.DeclareVectorInputPort("thanos_position", 7)
        self.DeclareVectorInputPort("medusa_position", 7)
        self.DeclareVectorOutputPort("torque", 14, self.DoCalcOutput)
    
    def DoCalcOutput(self, context, output):
        q_thanos = self.GetInputPort("thanos_position").Eval(context)
        q_medusa = self.GetInputPort("medusa_position").Eval(context)
        self._plant.SetPositions(self._plant_context, self._iiwa_thanos, q_thanos)
        self._plant.SetPositions(self._plant_context, self._iiwa_medusa, q_medusa)
        
        # get where the z vector of thanos finger is pointing towards in world frame
        z_vector_thanos_finger = self._plant.GetFrameByName("thanos_finger").CalcPoseInWorld(self._plant_context).rotation().matrix() @ np.array([0,0,1])
        z_dir_thanos_finger = z_vector_thanos_finger[2]
        
        adder_left_force = np.array([0,0,0])
        adder_right_force = np.array([0,0,0])

        if z_dir_thanos_finger >= 1e-3: # object weight on thanos finger 
            rot_thanos_finger = self._plant.GetFrameByName("thanos_finger").CalcPoseInWorld(self._plant_context).rotation().matrix()
            adder_left_force[2] = np.abs(rot_thanos_finger.T @ self.grav_force)[2] # now this is in thanos finger frame
        elif z_dir_thanos_finger < -1e-3: # object pushing on medusa finger
            rot_medusa_finger = self._plant.GetFrameByName("medusa_finger").CalcPoseInWorld(self._plant_context).rotation().matrix()
            adder_right_force[2] = np.abs(rot_medusa_finger.T @ self.grav_force)[2] # now this is in medusa finger frame
        else:
            pass
        
        thanos_pose_rot = self._plant.GetFrameByName("iiwa_link_7", self._iiwa_thanos).CalcPoseInWorld(self._plant_context).rotation().matrix()
        medusa_pose_rot = self._plant.GetFrameByName("iiwa_link_7", self._iiwa_medusa).CalcPoseInWorld(self._plant_context).rotation().matrix()
        
        thanos_force = thanos_pose_rot @ (self._force + adder_left_force)
        medusa_force = medusa_pose_rot @ (self._force + adder_right_force)

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
        
        if np.isnan(J_G_thanos[3:,:]).any() or np.linalg.matrix_rank(J_G_thanos[3:,:]) < 3:
            mat_thanos = np.zeros((7,3))
        else:
            mat_thanos = J_G_thanos[3:,:].T
        
        if np.isnan(J_G_medusa[3:,:]).any() or np.linalg.matrix_rank(J_G_medusa[3:,:]) < 3:
            mat_medusa = np.zeros((7,3))
        else:
            mat_medusa = J_G_medusa[3:,:].T
            
        tau_thanos = mat_thanos @ thanos_force
        tau_medusa = mat_medusa @ medusa_force
        
        output.SetFromVector(np.concatenate([tau_thanos, tau_medusa]))