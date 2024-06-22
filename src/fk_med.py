# get forward kinematic visualization of med robot
import numpy as np
from IPython.display import clear_output
from pydrake.all import (
    AbstractValue,
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    JointSliders,
    LeafSystem,
    MeshcatVisualizer,
    Parser,
    RigidTransform,
    RollPitchYaw,
    StartMeshcat,
    MultibodyPlant,
    FixedOffsetFrame,
    RotationMatrix
)

from manipulation import ConfigureParser, running_as_notebook
from manipulation.scenarios import AddMultibodyTriad
meshcat = StartMeshcat()

class PrintPose(LeafSystem):
    def __init__(self, body_index):
        LeafSystem.__init__(self)
        self._body_index = body_index
        self.DeclareAbstractInputPort(
            "body_poses", AbstractValue.Make([RigidTransform()])
        )
        self.DeclareForcedPublishEvent(self.Publish)

    def Publish(self, context):
        pose = self.get_input_port().Eval(context)[self._body_index]
        print(pose)
        print(
            "gripper position (m): "
            + np.array2string(
                pose.translation(),
                formatter={"float": lambda x: "{:3.2f}".format(x)},
            )
        )
        print(
            "gripper roll-pitch-yaw (rad):"
            + np.array2string(
                RollPitchYaw(pose.rotation()).vector(),
                formatter={"float": lambda x: "{:3.2f}".format(x)},
            )
        )
        clear_output(wait=True)


def gripper_forward_kinematics_example():
    builder = DiagramBuilder()

    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0)
    plant : MultibodyPlant = plant
    parser = Parser(plant)
    parser.package_map().AddPackageXml("../package.xml")
    ConfigureParser(parser)
    parser.AddModelsFromUrl("package://bimanual/urdf/med.urdf")
    
    apriltag_rot = RotationMatrix.MakeXRotation(-np.pi/2) @ RotationMatrix.MakeZRotation(np.pi)
    apriltag_trans = np.array([-0.025, 0.0365, 0.0865])
    apriltag_rt = RigidTransform(apriltag_rot, apriltag_trans)
    frame_apriltag = FixedOffsetFrame("apriltag", plant.GetFrameByName("iiwa_link_7"), apriltag_rt)
    plant.AddFrame(frame=frame_apriltag)
    
    plant.Finalize()
    # Draw the frames
    for body_name in [
        "iiwa_link_1",
        "iiwa_link_2",
        "iiwa_link_3",
        "iiwa_link_4",
        "iiwa_link_5",
        "iiwa_link_6",
        "apriltag"
    ]:
        AddMultibodyTriad(plant.GetFrameByName(body_name), scene_graph)

    meshcat.Delete()
    meshcat.DeleteAddedControls()
    visualizer = MeshcatVisualizer.AddToBuilder(
        builder, scene_graph.get_query_output_port(), meshcat
    )

    # wsg = plant.GetModelInstanceByName("wsg")
    # gripper = plant.GetBodyByName("body", wsg)
    # print_pose = builder.AddSystem(PrintPose(gripper.index()))
    # builder.Connect(plant.get_body_poses_output_port(), print_pose.get_input_port())

    # default_interactive_timeout = None if running_as_notebook else 1.0
    sliders = builder.AddSystem(JointSliders(meshcat, plant))
    diagram = builder.Build()
    sliders.Run(diagram, None)
    meshcat.DeleteAddedControls()

if __name__ == '__main__':
    gripper_forward_kinematics_example()
    input()