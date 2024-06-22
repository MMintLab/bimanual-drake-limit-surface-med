import numpy as np
from pydrake.multibody.parsing import LoadModelDirectives, ProcessModelDirectives
from pydrake.multibody.plant import MultibodyPlant
from pydrake.geometry import SceneGraph
from pydrake.multibody.parsing import Parser


def load_iiwa_setup(plant: MultibodyPlant, scene_graph: SceneGraph = None, package_file='./package.xml', directive_path="single_med.yaml"):
    parser = Parser(plant, scene_graph)
    parser.package_map().AddPackageXml(package_file)
    directives = LoadModelDirectives(directive_path)
    models = ProcessModelDirectives(directives=directives, plant=plant, parser=parser)