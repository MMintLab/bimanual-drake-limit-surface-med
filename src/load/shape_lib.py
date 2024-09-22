from pydrake.multibody.plant import MultibodyPlant
from pydrake.geometry import ProximityProperties, Box, HalfSpace, Sphere, Cylinder
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
def AddGround(plant: MultibodyPlant):
    ground_color = [0.5, 1.0, 0.5, 1.0]
    ground_prop = AddContactModel(plant, halfspace_slab=0.5, hydro_mod = 3e4, dissip=1.0, mu_static=1.0, res_hint=10)
    RegisterShape(plant, "GroundVisualGeometry", plant.world_body(), HalfSpace(), ground_prop, ground_color)
    
def AddGhostBox(plant: MultibodyPlant, name: str, lwh=(1.0,1.0,1.0), mass=1.0, mu = 1.0, color=[1,0,0,1]):
    box_instance = plant.AddModelInstance(name)
    box_body = plant.AddRigidBody(f"{name}_body",
                box_instance,
                SpatialInertia(mass=mass,
                               p_PScm_E=np.array([0.0,0.0,0.0]),
                               G_SP_E=UnitInertia.SolidBox(*lwh)
                               )
                )
    box = Box(*lwh)
    RegisterVisualShape(plant, name, box_body, box, color)
    return box_instance
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
    # hydro_mod = 5e4
    hydro_mod = 1e6

    box_props = AddContactModel(plant, mu_static=mu, hydro_mod= hydro_mod, dissip = dissip, res_hint=0.01)
    # box_props = AddContactModel(plant, mu_static=mu, res_hint=0.01)
    RegisterShape(plant, name, box_body, box, box_props, color)
    return box_instance
def AddCylinder(plant: MultibodyPlant, radius: float, length: float, name: str, 
                mass: float = 1.0, mu=1.0, color=[1,0,0,1]):

    ## add cylinder to the plant
    book_instance = plant.AddModelInstance(name)
    cylinder = Cylinder(radius,length)

    book_body = plant.AddRigidBody(f"{name}_body",
                book_instance,
                SpatialInertia(mass=mass,
                               p_PScm_E=np.array([0.0,0.0,0.0]),
                               G_SP_E=UnitInertia.SolidCylinder(
        cylinder.radius(), cylinder.length(), [0, 0, 1]))
    )

    dissip = 0
    # hydro_mod = 5e4
    hydro_mod = 1e6
    ## put in hydroelastic properties
    book_props = AddContactModel(plant, dissip=dissip, hydro_mod=hydro_mod, mu_static=mu, mu_dynamic=mu, res_hint=0.01)

    ## register collision geometry (hydroelastic + geometric/visual)
    RegisterShape(plant, name, book_body, cylinder, book_props, color)
    return book_instance

def AddSphere(plant: MultibodyPlant, name: str, radius=1.0, mass=1.0, mu = 1):
    sphere_instance = plant.AddModelInstance(name)
    
    sphere_body = plant.AddRigidBody(f"{name}_body",
                sphere_instance,
                SpatialInertia(mass=mass,
                               p_PScm_E=np.array([0.0,0.0,0.0]),
                               G_SP_E=UnitInertia.SolidSphere(radius)
                               )
                )
    sphere = Sphere(radius)
    
    dissip = 0
    hydro_mod = 1e6
    
    sphere_props = AddContactModel(plant, mu_static=mu, hydro_mod= hydro_mod, dissip = dissip, res_hint=0.04)
    RegisterShape(plant, name, sphere_body, sphere, sphere_props)
    return sphere_instance

def AddCustomObject(plant: MultibodyPlant, name: str, largest_width: float, depth: float,smaller_percent: float, mass: float, mu_smaller = 1.0, mu_bigger=1.0):
    center_temp = AddSphere(plant, f"{name}", radius=depth/4, mass=1e-8)
    larger_object = AddBox(plant, f"{name}_2", lwh=(largest_width, largest_width, depth), mass=mass, color=[0,0,1,0.3], mu=mu_bigger)
    smaller_object = AddBox(plant, f"{name}_3", lwh=(largest_width*smaller_percent, largest_width*smaller_percent, depth), mass=mass, color=[1,0,0,0.3], mu=mu_smaller)
    
    plant.WeldFrames(plant.GetFrameByName(f"{name}_body"), plant.GetFrameByName(f"{name}_2_body"), RigidTransform([0,0,-depth/2]))
    plant.WeldFrames(plant.GetFrameByName(f"{name}_body"), plant.GetFrameByName(f"{name}_3_body"), RigidTransform([0,0,depth/2]))
    
    return center_temp, larger_object, smaller_object