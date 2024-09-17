import numpy as np
from pydrake.math import RigidTransform
from pydrake.common.eigen_geometry import Quaternion

from pydrake.multibody.plant import MultibodyPlant
from pydrake.systems.framework import LeafSystem
from pydrake.common.value import Value
from collections import defaultdict
from pydrake.multibody.plant import ContactResults

class ContactForceReporter(LeafSystem):
    def __init__(self,period=0.1, offset=0.0):
        LeafSystem.__init__(self)
        self.DeclareAbstractInputPort(name='contact_results',
                                     model_value=Value(ContactResults()))
        
        self.DeclarePeriodicPublishEvent(period_sec=period, offset_sec=0.0, publish=self.Publish)

        self.wrench_hash = defaultdict(list)
        self.offset = offset
        self.ts = []
    def Publish(self, context):
        if context.get_time() > self.offset:
            contact_results = self.get_input_port().Eval(context)
            
            num_hydroelastic_contacts = contact_results.num_hydroelastic_contacts()
            if num_hydroelastic_contacts > 0:
                self.ts.append(context.get_time())
            for c in range(num_hydroelastic_contacts):
                hydroelastic_contact_info = contact_results.hydroelastic_contact_info(c)
                
                spatial_force = hydroelastic_contact_info.F_Ac_W()
                force = spatial_force.translational().reshape((3,1))
                torque = spatial_force.rotational().reshape((3,1))
                wrench = np.vstack((torque,force))

                self.wrench_hash[c].append(wrench)

class RecordPoses(LeafSystem):
    def __init__(self, plant: MultibodyPlant):
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        
        self._left_arm_port = self.DeclareVectorInputPort("left_arm", 14)
        self._right_arm_port = self.DeclareVectorInputPort("right_arm", 14)
        self._object_port = self.DeclareVectorInputPort("object", 13)
        
        self.DeclarePeriodicPublishEvent(period_sec=0.1, offset_sec=1.0, publish=self.Publish)
        
        self.object2right_data = []
        self.object2left_data = []
        self.force1_data = []
        self.force2_data = []
        
    def Publish(self, context):
        q_left = self._left_arm_port.Eval(context)[:7]
        q_right = self._right_arm_port.Eval(context)[:7]
        object = self._object_port.Eval(context)[:7]
        
        self._plant.SetPositions(self._plant_context, self._plant.GetModelInstanceByName("franka_left"), q_left)
        self._plant.SetPositions(self._plant_context, self._plant.GetModelInstanceByName("franka_right"), q_right)
        left_pose = self._plant.EvalBodyPoseInWorld(self._plant_context, self._plant.GetBodyByName("left_finger"))
        right_pose = self._plant.EvalBodyPoseInWorld(self._plant_context, self._plant.GetBodyByName("right_finger"))
        object[:4] /= np.linalg.norm(object[:4])
        object_pose = RigidTransform(Quaternion(object[:4]), object[4:])
        
        # check if fingers are 180 degrees facing apart (X angle)
        right2left = left_pose.inverse() @ right_pose
        right2left_pitch = np.round(right2left.rotation().ToRollPitchYaw().pitch_angle()*180/np.pi,4)
        right2left_roll = np.round(right2left.rotation().ToRollPitchYaw().roll_angle()*180/np.pi,4)
        right2left_yaw = np.round(right2left.rotation().ToRollPitchYaw().yaw_angle()*180/np.pi,4)
        
        
        
        object2left = left_pose.inverse() @ object_pose
        object2right = right_pose.inverse() @ object_pose
        
        object2left_ts = object2left.translation()[:2]
        object2left_yaw = object2left.rotation().ToRollPitchYaw().yaw_angle()
        object2left_se2 = np.array([object2left_ts[0], object2left_ts[1], object2left_yaw])        
        
        object2right_ts = object2right.translation()[:2]
        object2right_yaw = object2right.rotation().ToRollPitchYaw().yaw_angle()
        object2right_se2 = np.array([object2right_ts[0], object2right_ts[1], (object2right_yaw - np.pi)])
        
        self.object2left_data.append(object2left_se2)
        self.object2right_data.append(object2right_se2)