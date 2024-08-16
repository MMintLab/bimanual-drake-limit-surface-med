from pydrake.all import (
    DiagramBuilder,
    MultibodyPlant,
    ApplyMultibodyPlantConfig,
    Parser,
    ProcessModelDirectives,
    ModelDirectives,
    AddMultibodyPlantSceneGraph,
    LoadModelDirectives,
    MeshcatVisualizerParams,
    Role,
    MeshcatVisualizer,
    Diagram
)
from manipulation.station import MakeHardwareStation, _MakeHardwareStationInterface, load_scenario
from typing import Tuple
def get_hardware_blocks(hardware_builder, scenario, meshcat = None, package_file='./package.xml'):
    real_station = hardware_builder.AddNamedSystem(
        "real_station",
        _MakeHardwareStationInterface(
            scenario,
            meshcat=meshcat,
            package_xmls=[package_file]
        )
    )
    fake_station = hardware_builder.AddNamedSystem(
        "fake_station",
        MakeHardwareStation(
            scenario,
            meshcat=meshcat,
            package_xmls=[package_file]
        )
    )
    hardware_plant = MultibodyPlant(scenario.plant_config.time_step)
    ApplyMultibodyPlantConfig(scenario.plant_config, hardware_plant)
    parser = Parser(hardware_plant)
    parser.package_map().AddPackageXml(package_file)
    ProcessModelDirectives(
        directives=ModelDirectives(directives=scenario.directives),
        plant=hardware_plant,
        parser=parser
    )
    
    
    return real_station, fake_station

def create_hardware_diagram_plant(scenario_filepath, meshcat, position_only=True) -> Tuple[Diagram, MultibodyPlant]:
    hardware_builder = DiagramBuilder()
    scenario = load_scenario(filename=scenario_filepath, scenario_name="Demo")
    real_station, fake_station = get_hardware_blocks(hardware_builder, scenario, meshcat)
    
    hardware_plant = fake_station.GetSubsystemByName("plant")
    
    hardware_builder.ExportInput(
        real_station.GetInputPort("iiwa.position"), "iiwa.position"
    )
    hardware_builder.ExportInput(
        fake_station.GetInputPort("iiwa.position"), "iiwa_fake.position"
    )
    if not position_only:
        hardware_builder.ExportInput(
            real_station.GetInputPort("iiwa.feedforward_torque"), "iiwa.feedforward_torque"
        )
        hardware_builder.ExportInput(
            fake_station.GetInputPort("iiwa.feedforward_torque"), "iiwa_fake.feedforward_torque"
        )

    
    hardware_builder.ExportOutput(
        real_station.GetOutputPort("iiwa.position_commanded"), "iiwa.position_commanded"
    )
    hardware_builder.ExportOutput(
        real_station.GetOutputPort("iiwa.position_measured"), "iiwa.position_measured"
    )
    
    hardware_diagram = hardware_builder.Build()
    return hardware_diagram, hardware_plant

def create_hardware_diagram_plant_bimanual(scenario_filepath, meshcat = None, position_only=True) -> Tuple[Diagram, MultibodyPlant]:
    hardware_builder = DiagramBuilder()
    scenario = load_scenario(filename=scenario_filepath, scenario_name="Demo")
    real_station, fake_station = get_hardware_blocks(hardware_builder, scenario, meshcat, package_file='../../package.xml')
    
    hardware_plant = fake_station.GetSubsystemByName("plant")
    
    
    #medusa
    hardware_builder.ExportInput(
        real_station.GetInputPort("iiwa_medusa.position"), "iiwa_medusa.position"
    )
    hardware_builder.ExportInput(
        fake_station.GetInputPort("iiwa_medusa.position"), "iiwa_medusa_fake.position"
    )
    if not position_only:
        hardware_builder.ExportInput(
            real_station.GetInputPort("iiwa_medusa.feedforward_torque"), "iiwa_medusa.feedforward_torque"
        )
        hardware_builder.ExportInput(
            fake_station.GetInputPort("iiwa_medusa.feedforward_torque"), "iiwa_medusa_fake.feedforward_torque"
        )
    hardware_builder.ExportOutput(
        real_station.GetOutputPort("iiwa_medusa.position_commanded"), "iiwa_medusa.position_commanded"
    )
    hardware_builder.ExportOutput(
        real_station.GetOutputPort("iiwa_medusa.position_measured"), "iiwa_medusa.position_measured"
    )
    
    
    hardware_builder.ExportInput(
        real_station.GetInputPort("iiwa_thanos.position"), "iiwa_thanos.position"
    )
    hardware_builder.ExportInput(
        fake_station.GetInputPort("iiwa_thanos.position"), "iiwa_thanos_fake.position"
    )
    if not position_only:
        hardware_builder.ExportInput(
            real_station.GetInputPort("iiwa_thanos.feedforward_torque"), "iiwa_thanos.feedforward_torque"
        )
        hardware_builder.ExportInput(
            fake_station.GetInputPort("iiwa_thanos.feedforward_torque"), "iiwa_thanos_fake.feedforward_torque"
        )
    hardware_builder.ExportOutput(
        real_station.GetOutputPort("iiwa_thanos.position_commanded"), "iiwa_thanos.position_commanded"
    )
    hardware_builder.ExportOutput(
        real_station.GetOutputPort("iiwa_thanos.position_measured"), "iiwa_thanos.position_measured"
    )
    
    hardware_diagram = hardware_builder.Build()
    return hardware_diagram, hardware_plant

def create_visual_diagram(directives_filepath: str, meshcat, package_file='./package.xml') -> Diagram:
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    # run simulator.set_target_realtime_rate(1.0)
    parser = Parser(plant)
    parser.package_map().AddPackageXml(package_file)
    directives = LoadModelDirectives(directives_filepath)
    models = ProcessModelDirectives(directives, plant, parser)
    plant.Finalize()
    
    meshcat_visual_params = MeshcatVisualizerParams()
    meshcat_visual_params.delete_on_initialization_event = False
    meshcat_visual_params.role = Role.kIllustration
    meshcat_visual_params.prefix = "visual"
    meshcat_visual = MeshcatVisualizer.AddToBuilder(
        builder, scene_graph, meshcat, meshcat_visual_params)
    
    meshcat_collision_params = MeshcatVisualizerParams()
    meshcat_collision_params.delete_on_initialization_event = False
    meshcat_collision_params.role = Role.kProximity
    meshcat_collision_params.prefix = "collision"
    meshcat_collision_params.visible_by_default = False
    meshcat_collision = MeshcatVisualizer.AddToBuilder(
        builder, scene_graph, meshcat, meshcat_collision_params)
    vis_diagram = builder.Build()
    return vis_diagram

