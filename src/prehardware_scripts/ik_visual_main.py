import numpy as np
from pydrake.systems.framework import DiagramBuilder
from pydrake.multibody.plant import MultibodyPlantConfig, AddMultibodyPlant
from pydrake.geometry import StartMeshcat
from pydrake.all import MeshcatVisualizerParams, MeshcatVisualizer, MultibodyPlant
from manipulation.meshcat_utils import MeshcatPoseSliders
from manipulation.scenarios import AddMultibodyTriad
from pydrake.math import RigidTransform, RotationMatrix, RollPitchYaw
from collections import namedtuple

import sys
sys.path.append('..')
from planning.ik_util import solveDualIK
from load.sim_setup import load_iiwa_setup
from load.finger_lib import AddSingleFinger

class InteractiveArm:
    def __init__(self):
        pass
    def run(self):
        config = MultibodyPlantConfig()
        meshcat = StartMeshcat()
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlant(config, builder)
        plant: MultibodyPlant = plant
        load_iiwa_setup(plant, scene_graph, package_file='../../package.xml', directive_path="../../config/bimanual_med.yaml")
        #no grav
        plant.mutable_gravity_field().set_gravity_vector([0, 0, 0])
        length = 0.01
        thanos_plate = AddSingleFinger(plant, radius=0.19/2.0, length=length, name="thanos_plate", mass=1.0, mu=1.0, color=[1,0,0,1])
        medusa_plate = AddSingleFinger(plant, radius=0.19/2.0, length=length, name="medusa_plate", mass=1.0, mu=1.0, color=[1,0,0,1])
        
        plant.WeldFrames(plant.GetFrameByName("thanos_finger"), plant.GetFrameByName("thanos_plate"), RigidTransform(np.array([0,0,0.13 - length/2 + 0.035])))
        plant.WeldFrames(plant.GetFrameByName("medusa_finger"), plant.GetFrameByName("medusa_plate"), RigidTransform(np.array([0,0,0.13 - length/2 + 0.035])))
        plant.Finalize()
        
        visualizer = MeshcatVisualizer.AddToBuilder(
            builder,
            scene_graph,
            meshcat,
            MeshcatVisualizerParams(delete_prefix_initialization_event=False),
        )
        
        AddMultibodyTriad(plant.GetFrameByName("iiwa_link_ee_kuka", plant.GetModelInstanceByName("iiwa_thanos")), scene_graph, length=0.1, radius=0.003)
        AddMultibodyTriad(plant.GetFrameByName("iiwa_link_ee_kuka", plant.GetModelInstanceByName("iiwa_medusa")), scene_graph, length=0.1, radius=0.003)
        
        diagram = builder.Build()
        context = diagram.CreateDefaultContext()
        plant_context = plant.GetMyContextFromRoot(context)
        
        q0 = [0.08232356364776336, 0.49329539590471605, 0.7554412443584381, -2.0426179181360524, 2.0754790345007996, 0.8874891667572512, -1.1673120760704268,
                 -1.4536369838514789, 0.5612986824682098, 0.8971038307962235, -2.003297518161298, 0.8415437358419539, -1.392097329426083, 0.7279235421513163]
        q0 = np.array(q0)
        # q0 = np.zeros(14)


        plant.SetPositions(plant_context, q0)
        
        def callback(context, pose):
            q0 = plant.GetPositions(plant_context)
            
            gap = 0.075 + 0.26
            modif = -30.0 * np.pi / 180 * 0
            left_pose = RigidTransform(pose.rotation().ToQuaternion(), pose.translation() + pose.rotation().matrix() @ np.array([0,0,-gap/2]))
            right_rot = RotationMatrix(pose.rotation().matrix()) @ RotationMatrix.MakeYRotation(np.pi)
            right_pose = RigidTransform( (right_rot @ RotationMatrix.MakeXRotation(modif)).ToQuaternion(), pose.translation() + right_rot.matrix() @ np.array([0,0,-gap/2]))
            sol, success = solveDualIK(plant, left_pose, right_pose, "medusa_finger", "thanos_finger", q0)
            
            # thanos q
            # format numpy print with commas
            if success:
                print("Thanos")
                print(sol[0:7].tolist())
                # medusa q
                print("Medusa") 
                print(sol[7:].tolist())
                print()
                print(sol.tolist())
            else:
                print("Fail")
                sol = np.zeros(14)
            plant.SetPositions(plant_context, sol)

        
        meshcat.DeleteAddedControls()
        MinRange = namedtuple("MinRange", ("roll", "pitch", "yaw", "x", "y", "z"))
        MinRange.__new__.__defaults__ = (0, 0, 0, 0.0, 0.0, 0.0)
        MaxRange = namedtuple("MaxRange", ("roll", "pitch", "yaw", "x", "y", "z"))
        MaxRange.__new__.__defaults__ = (np.pi, 0, 0, 1.0, 1.0, 2.0)
        sliders = MeshcatPoseSliders(meshcat,min_range=MinRange(), max_range=MaxRange())

        init_pose = RigidTransform(RollPitchYaw(np.pi/2, 0, 0), [0.32, 0.6096, 0.42])
        sliders.SetPose(init_pose)
        sliders.Run(visualizer, context, callback)
        
if __name__ == "__main__":
    app = InteractiveArm()
    app.run()