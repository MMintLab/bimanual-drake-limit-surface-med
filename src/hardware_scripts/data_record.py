from pydrake.all import (
    MultibodyPlant,
    LeafSystem,
    Context
)
        
class BenchmarkController(LeafSystem):
    def __init__(self, plant: MultibodyPlant):
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        
        self._target = self.DeclareVectorInputPort("target", 14)
        self._measured = self.DeclareVectorInputPort("measure", 14)
        
        
        self.thanos_instance = self._plant.GetModelInstanceByName("iiwa_thanos")
        self.medusa_instance = self._plant.GetModelInstanceByName("iiwa_medusa")
        
        self.DeclarePeriodicPublishEvent(period_sec=1e-2, offset_sec=0.0, publish=self.Publish)
        
        self.ts = []
        self.q_targets = []
        self.q_measures = []
        
        self.thanos_positions = []
        self.thanos_des_positions = []
        
        self.thanos_rotations = []
        self.thanos_des_rotations = []
        
        self.medusa_positions = []
        self.medusa_des_positions = []
        
        self.medusa_rotations = []
        self.medusa_des_rotations = []
        
    def Publish(self, context: Context):
        self.ts.append(context.get_time())
        q_target = self._target.Eval(context)
        q_measure = self._measured.Eval(context)
        
        self._plant.SetPositions(self._plant_context, q_measure)
        ee_pose_thanos = self._plant.CalcRelativeTransform(self._plant_context, self._plant.world_frame(), self._plant.GetBodyByName("iiwa_link_7", self.thanos_instance).body_frame())
        ee_pose_medusa = self._plant.CalcRelativeTransform(self._plant_context, self._plant.world_frame(), self._plant.GetBodyByName("iiwa_link_7", self.medusa_instance).body_frame())
        thanos_position = ee_pose_thanos.translation()
        thanos_rotation = ee_pose_thanos.rotation().matrix()
        medusa_position = ee_pose_medusa.translation()
        medusa_rotation = ee_pose_medusa.rotation().matrix()
            
        self._plant.SetPositions(self._plant_context, q_target)
        ee_pose_des_thanos = self._plant.CalcRelativeTransform(self._plant_context, self._plant.world_frame(), self._plant.GetBodyByName("iiwa_link_7", self.thanos_instance).body_frame())
        ee_pose_des_medusa = self._plant.CalcRelativeTransform(self._plant_context, self._plant.world_frame(), self._plant.GetBodyByName("iiwa_link_7", self.medusa_instance).body_frame())
        thanos_position_des = ee_pose_des_thanos.translation()
        thanos_rotation_des = ee_pose_des_thanos.rotation().matrix()
        medusa_position_des = ee_pose_des_medusa.translation()
        medusa_rotation_des = ee_pose_des_medusa.rotation().matrix()
        
        
        
        
        self.q_measures.append(q_measure)
        self.q_targets.append(q_target)
        self.thanos_positions.append(thanos_position)
        self.thanos_des_positions.append(thanos_position_des)
        self.thanos_rotations.append(thanos_rotation)
        self.thanos_des_rotations.append(thanos_rotation_des)
        
        self.medusa_positions.append(medusa_position)
        self.medusa_des_positions.append(medusa_position_des)
        self.medusa_rotations.append(medusa_rotation)
        self.medusa_des_rotations.append(medusa_rotation_des)