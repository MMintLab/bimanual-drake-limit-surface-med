###################################################################
# load.py
# 
###################################################################
import sys
sys.path.append("..")
from pydrake.multibody.parsing import LoadModelDirectives, ProcessModelDirectives
from pydrake.multibody.plant import MultibodyPlant, MultibodyPlantConfig
from pydrake.geometry import SceneGraph
from pydrake.multibody.parsing import Parser
from pydrake.math import RigidTransform, RotationMatrix
import load.finger_lib as finger_lib
import os

def RepoDir():
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def load_bimanual_setup(plant: MultibodyPlant, scene_graph: SceneGraph = None, radius_left_finger=0.1, radius_right_finger=0.1):
    directive_path = os.path.join(RepoDir(),"../urdf/franka_description/bimanual_franka_station.yaml")
    
    
    if scene_graph is None:
        parser = Parser(plant)
    else:
        parser = Parser(plant, scene_graph)
    directives = LoadModelDirectives(directive_path)
    models = ProcessModelDirectives(directives, plant, parser)
    
    left_finger_instance = finger_lib.AddSingleFinger(plant, radius=radius_left_finger, length=0.03, name="left_finger", mass=1.0, mu=1.0, color=[1,0,0,0.3])
    right_finger_instance = finger_lib.AddSingleFinger(plant, radius=radius_right_finger, length=0.03, name="right_finger", mass=1.0, mu=1.0, color=[0,1,0,0.3])    
    
    left_franka_instance = plant.GetModelInstanceByName("franka_left")
    right_franka_instance = plant.GetModelInstanceByName("franka_right")
    
    #offset by 0.02
    offset = RigidTransform(RotationMatrix().ToQuaternion(),[0,0,0.015])
    
    plant.WeldFrames(plant.GetFrameByName("panda_link8", left_franka_instance), plant.GetFrameByName("left_finger"), offset)
    plant.WeldFrames(plant.GetFrameByName("panda_link8", right_franka_instance), plant.GetFrameByName("right_finger"), offset)
    
    plant.Finalize()