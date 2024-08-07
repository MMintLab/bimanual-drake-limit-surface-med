from pydrake.multibody.plant import MultibodyPlant
from pydrake.geometry import Cylinder, ProximityProperties
from pydrake.multibody.tree import RigidBody, SpatialInertia, UnitInertia
from pydrake.math import RigidTransform
from pydrake.multibody.tree import PrismaticJoint, RevoluteJoint, WeldJoint, FixedOffsetFrame
import numpy as np

from load.contact_lib import AddContactModel
from load.shape_lib import RegisterShape, RegisterVisualShape


def AddSingleFinger(plant: MultibodyPlant, radius: float, length: float, name: str, 
                mass: float = 1.0, mu=1.0, color=[1,0,0,1], rt=RigidTransform()):

    ## add cylinder to the plant
    cylinder = Cylinder(radius,length)
    instance = plant.AddModelInstance(name)

    inertia = UnitInertia.SolidCylinder(
        cylinder.radius(), cylinder.length(), [0, 0, 1]
    )
    spatial_inertia = SpatialInertia(
            mass=mass, p_PScm_E=np.array([0.0, 0.0, 0.0]), G_SP_E=inertia
        )
    body = plant.AddRigidBody(
        name,
        instance,
        spatial_inertia,
    )

    dissip = 0
    hydro_mod = 1e6
    ## put in hydroelastic properties
    cylinder_prop = AddContactModel(plant, dissip=dissip, hydro_mod=hydro_mod, mu_static=mu, mu_dynamic=mu, res_hint=0.01)
    # cylinder_prop = AddContactModel(plant, mu_static=mu, res_hint=0.01)
    
    ## register collision geometry (hydroelastic + geometric/visual)
    RegisterShape(plant, name, body, cylinder, cylinder_prop, color, rt=rt)
    return instance, spatial_inertia