from pydrake.all import (
    LeafSystem,
    MultibodyPlant,
    JacobianWrtVariable
)
import numpy as np

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
                                                           self._plant.GetBodyByName("iiwa_link_7", self._thanos_instance).body_frame(),
                                                           [0,0,0],
                                                           self._plant.world_frame(),
                                                           self._plant.world_frame())[:, :7]
        J_medusa = self._plant.CalcJacobianSpatialVelocity(self._plant_context,
                                                           JacobianWrtVariable.kQDot,
                                                           self._plant.GetBodyByName("iiwa_link_7", self._medusa_instance).body_frame(),
                                                           [0,0,0],
                                                           self._plant.world_frame(),
                                                           self._plant.world_frame())[:, 7:]
        
        thanos_torque = J_thanos.T @ wrench_thanos
        medusa_torque = J_medusa.T @ wrench_medusa
        output.SetFromVector(np.concatenate([thanos_torque, medusa_torque]))