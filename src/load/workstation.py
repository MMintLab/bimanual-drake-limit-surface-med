from pydrake.geometry import StartMeshcat
from pydrake.multibody.plant import MultibodyPlant, MultibodyPlantConfig, AddMultibodyPlant
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.analysis import Simulator

from abc import ABC, abstractmethod

class WorkStation(ABC):
    '''
        Abstract Base Class for Simulation
    '''
    def __init__(self, contact_model="hydroelastic_with_fallback",multibody_dt=1e-3,penetration_allowance=1e-3, visual=True):
        if contact_model != "hydroelastic_with_fallback" and contact_model != "point" and contact_model != "hydroelastic":
            raise ValueError("Wrong Contact Model")
        
        CONTACT_SURF = "polygon"
        self.config  = MultibodyPlantConfig()
        self.config.time_step = multibody_dt
        self.config.penetration_allowance = penetration_allowance
        self.config.contact_model = contact_model
        self.config.contact_surface_representation = CONTACT_SURF
        # self.config.sap_near_rigid_threshold = 0.01
        self.config.discrete_contact_approximation = "tamsi"
        '''
            TAMSI is way more stable than SAP.
            SAP is way faster.
        '''
        self.visual = visual
        self.runtime = 5.0

    #must be overloaded or else you can't simulate
    @abstractmethod
    def setup_simulate(self, builder: DiagramBuilder, plant: MultibodyPlant, scene_graph):
        raise NotImplementedError

    @abstractmethod
    def initialize_simulate(self, plant: MultibodyPlant, plant_context):
        raise NotImplementedError
    
    def run(self, realtime_rate=1.0):
        if self.visual:
            self.meshcat = StartMeshcat()
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlant(self.config, builder)

        self.setup_simulate(builder, plant, scene_graph)

        diagram   = builder.Build()
        simulator = Simulator(diagram)
        plant_context = plant.GetMyContextFromRoot(simulator.get_mutable_context())
        self.initialize_simulate(plant, plant_context)
        diagram.ForcedPublish(simulator.get_context())

        simulator.Initialize()
        simulator.set_target_realtime_rate(realtime_rate)
        self.meshcat.StartRecording()
        simulator.AdvanceTo(self.runtime)
        self.meshcat.StopRecording()
        self.meshcat.PublishRecording()