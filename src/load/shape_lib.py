from pydrake.multibody.plant import MultibodyPlant
from pydrake.geometry import ProximityProperties, Box
from pydrake.multibody.tree import RigidBody, SpatialInertia, UnitInertia
from pydrake.math import RigidTransform
import numpy as np

from load.contact_lib import AddContactModel

def RegisterVisualShape(plant: MultibodyPlant, name: str, body: RigidBody,
                        shape, color=[1,0,0,1]):
    plant.RegisterVisualGeometry(
        body, RigidTransform(), shape, name, color
    )

def RegisterShape(plant: MultibodyPlant, name:str, body: RigidBody, 
                  shape, prop: ProximityProperties, color=[1,0,0,1], rt=RigidTransform()):
    
    if plant.geometry_source_is_registered():
        plant.RegisterCollisionGeometry(
            body, rt, shape, name, prop
        )
        plant.RegisterVisualGeometry(
            body, rt, shape, name, color
        )
def AddBox(plant: MultibodyPlant, name: str, lwh=(1.0,1.0,1.0), mass=1.0, mu = 1.0, color=[1,0,0,1]):
    box_instance = plant.AddModelInstance(name)

    
    box_body = plant.AddRigidBody(f"{name}_body",
                box_instance,
                SpatialInertia(mass=mass,
                               p_PScm_E=np.array([0.0,0.0,0.0]),
                               G_SP_E=UnitInertia.SolidBox(*lwh)
                               )
                )
    box = Box(*lwh)
    
    dissip = 0
    hydro_mod = 5e4
    # hydro_mod = 1e6

    box_props = AddContactModel(plant, mu_static=mu, hydro_mod= hydro_mod, dissip = dissip, res_hint=0.01)
    # box_props = AddContactModel(plant, mu_static=mu, res_hint=0.01)
    RegisterShape(plant, name, box_body, box, box_props, color)
    return box_instance