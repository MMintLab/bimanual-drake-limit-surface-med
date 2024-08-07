#===============================================
# contact_lib.py
# 
# File for working with hydroelastic contact
# 
#===============================================
from pydrake.geometry import (
    AddContactMaterial, 
    ProximityProperties, 
    AddCompliantHydroelasticPropertiesForHalfSpace, 
    AddCompliantHydroelasticProperties, 
    AddRigidHydroelasticProperties,
    Meshcat
)

from pydrake.multibody.plant import ContactModel, ContactResults, HydroelasticContactInfo
from pydrake.geometry import MeshcatVisualizer, StartMeshcat, TriangleSurfaceMesh, TriangleSurfaceMeshFieldLinear
from pydrake.multibody.meshcat import ContactVisualizer, ContactVisualizerParams
from pydrake.geometry import TriangleSurfaceMesh, SurfaceTriangle


from pydrake.systems.framework import LeafSystem, DiagramBuilder
from pydrake.common.value import Value
from pydrake.multibody.plant import ContactResults
from pydrake.multibody.plant import MultibodyPlant, CoulombFriction
from pydrake.multibody.meshcat import ContactVisualizer, ContactVisualizerParams
import numpy as np
import open3d as o3d
from collections import defaultdict

def AddContactModel(plant: MultibodyPlant, halfspace_slab=0.0, mu_static=0.0, **kwargs):
    '''
    Paramters:
    ==========
    @param plant
    @param half_space_slab
    @param mu_static

    @param kwargs (dictionary)
    ===============================================================================
    | Compliant Parameters                                                        |
    ===============================================================================
    | Key       | Value
    -------------------------------------------------------------------------------
    | hydro_mod   | - measure of stiffness of material (pressure over penetration). 
    |             | - float (Pa (N/m^2))
    -------------------------------------------------------------------------------
    | dissip      | - energy dampening property of contact on object.
    |             | - >0 float (s/m) recommended dissip \in [0,3] and default: 1
    -------------------------------------------------------------------------------
    | res_hint    | - controls fineness of meshes from shapes. 
    |             | - coarse (fast but not accurate), fine (slow but accurate)
    |             | - float (meters)
    -------------------------------------------------------------------------------
    | mu_static   | - coefficient of static friction
    |             | - >0 float.
    -------------------------------------------------------------------------------
    | mu_dynamic  | - coefficient of dynamic friction
    |             | - >0 float recommended to keep same as mu_static
    -------------------------------------------------------------------------------
    ===============================================================================
    | Rigid Parameters                                                            |
    ===============================================================================
    | Key       | Value
    -------------------------------------------------------------------------------
    | res_hint    | - controls fineness of meshes from shapes. 
    |             | - coarse (fast but not accurate), fine (slow but accurate)
    |             | - float (meters)
    -------------------------------------------------------------------------------
    | mu_static   | - coefficient of static friction
    |             | - >0 float. Doesn't just apply to hydroelastic
    -------------------------------------------------------------------------------
    | mu_dynamic  | - coefficient of dynamic friction
    |             | - >0 float recommended to keep same as mu_static
    -------------------------------------------------------------------------------
    '''
    prop = ProximityProperties()
    contact_type = "compliant" if "hydro_mod" in kwargs else "rigid"
    mu_dynamic = (kwargs["mu_dynamic"] if "mu_dynamic" in kwargs else mu_static)
    res_hint   = (kwargs["res_hint"] if "res_hint" in kwargs else 1.0)
    dissip     = (kwargs["dissip"] if "dissip" in kwargs else 1.0)
    AddContactMaterial(dissip, None, CoulombFriction(mu_static, mu_dynamic), prop)

    if contact_type == "rigid":
        AddRigidHydroelasticProperties(res_hint, prop)        

    elif contact_type == "compliant":
        hydro_mod = kwargs["hydro_mod"]
        if halfspace_slab == 0.0:
            AddCompliantHydroelasticProperties(res_hint, hydro_mod, prop)
        else:
            AddCompliantHydroelasticPropertiesForHalfSpace(halfspace_slab, hydro_mod, prop)

    return prop