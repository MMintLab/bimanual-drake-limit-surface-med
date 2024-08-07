import sys
sys.path.append('..')

import numpy as np

#drake-specific imports
from manipulation.scenarios import AddMultibodyTriad
from pydrake.systems.framework import DiagramBuilder
from pydrake.multibody.plant import MultibodyPlant
from pydrake.all import SceneGraph, InverseDynamicsController, ConstantVectorSource, AddDefaultVisualization, LinearSpringDamper, PrismaticJoint, PrismaticSpring

from load.sim_setup import load_iiwa_setup
from load.workstation import WorkStation
from load.shape_lib import AddBox
from load.finger_lib import AddSingleFinger
from pydrake.math import RigidTransform

def setup_plant(plant: MultibodyPlant, scene_graph: SceneGraph = None):
    
    load_iiwa_setup(plant, scene_graph, package_file="../../package.xml", directive_path="../../config/single_med.yaml")
    iiwa_instance = plant.GetModelInstanceByName("iiwa")
    # finger characteristics
    # -> desired length 0.2375
    # -> offset 0.0425 m
    # -> full length 0.199 m
    # -> base length 0.189 m
    # -> plate length 0.01 m
    # -> radius 0.19 m
    finger_radius = 0.19/2.0 # 9cm
    box_width = 0.08 # 8cm
    obj_mass = 0.2 # 200g

    if not (scene_graph is None):
        finger = AddSingleFinger(plant, radius=finger_radius, length=0.01, name="finger", mass=1.0, mu=1.0, color=[1,0,0,1])
        finger_z_joint = plant.AddJoint(
            PrismaticJoint(
                "finger_z",
                plant.GetFrameByName("iiwa_link_7"),
                plant.GetFrameByName("finger"),
                [0, 0, 1],
                -1.0,
                1.0,
                damping=100.0
            )
        )    
        plant.AddForceElement(PrismaticSpring(finger_z_joint, 0.2375 - 0.01/2.0, 5000.0))
        
        obj_instance = AddBox(plant, "box", lwh=(box_width, box_width, box_width), mass=obj_mass, mu=1.0, color=[0,1,0,1])
    # offset = RigidTransform(np.array([0.0, 0.0, 0.2375 - 0.01/2]))
    # plant.WeldFrames(plant.GetFrameByName("iiwa_link_7"), plant.GetFrameByName("finger"), offset)
    
    return iiwa_instance
    
class Station(WorkStation):
    def __init__(self):
        WorkStation.__init__(self, "hydroelastic_with_fallback", multibody_dt=1e-3,penetration_allowance=1e-3,visual=True)
        self.q0 = np.zeros(7)
        
    def setup_simulate(self, builder: DiagramBuilder, plant: MultibodyPlant, scene_graph: SceneGraph):
        plant.mutable_gravity_field().set_gravity_vector([0, 0, -9.83])
        iiwa_instance = setup_plant(plant, scene_graph)
        plant.Finalize()
        
        plant_arms = MultibodyPlant(1e-3)
        setup_plant(plant_arms, None)
        plant_arms.Finalize()
        
        num_joints = plant_arms.num_positions()
        kp = 800*np.ones(num_joints)
        ki = 1*np.ones(num_joints)
        kd = 4*np.sqrt(kp)
        controller_block = builder.AddSystem(InverseDynamicsController(plant_arms, kp, ki, kd, False))
        zero_state_block = builder.AddSystem(ConstantVectorSource(np.zeros(num_joints*2)))
        
        builder.Connect(zero_state_block.get_output_port(), controller_block.get_input_port_desired_state())
        builder.Connect(controller_block.get_output_port(), plant.get_actuation_input_port())
        builder.Connect(plant.get_state_output_port(iiwa_instance), controller_block.get_input_port_estimated_state())
        
        AddDefaultVisualization(builder, self.meshcat)
        AddMultibodyTriad(plant.GetFrameByName("finger"), scene_graph, length=0.03, radius=0.003)
    def initialize_simulate(self, plant: MultibodyPlant, plant_context):
        zeros = np.zeros(15)
        zeros[7] = 0.2375 - 0.01/2.0
        plant.SetPositions(plant_context, zeros)
        plant.SetPositions(plant_context, plant.GetModelInstanceByName("iiwa"), self.q0)
        
        object_rt = RigidTransform([0.0, 0.00, 1.8])
        plant.SetFreeBodyPose(
            plant_context,
            plant.GetBodyByName("box_body"),
            object_rt
        )
        
if __name__ == '__main__':
    test = Station()
    test.run(5.0, 10.0)
    input()