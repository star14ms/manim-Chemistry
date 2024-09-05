from manim import Scene, ThreeDScene, config
from manim import *
from manim_chemistry import (
    MMoleculeObject,
    GraphMolecule,
)

from pathlib import Path
from manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVGroup


script_path = Path(__file__).absolute().parent
files_path = script_path / "glycolysis"


# 2D Molecule example
class Draw2DMorphine(Scene):

    # Two D Manim Chemistry objects require Cairo renderer
    config.renderer = "cairo"
    def construct(self):    
        molecules = [
            # 'd-glucose',
            # 'glucose-6-phosphate',
            # 'fructose-6-phosphate',
            # 'fructose-1,6-bisphosphate',
            # 'dihydroxyacetone phosphate',
            # 'glyceraldehyde-3-phosphate',
            # '1,3-bisphosphoglycerate',
            # '3-phosphoglycerate',
            # '2-phosphoglycerate',
            'phosphoenolpyruvate',
            'pyruvate',
        ]

        # from manim_chemistry.utils import mol_parser
        # atoms1, bonds1 = mol_parser(files_path / (molecules[0] + '.sdf'))
        # atoms2, bonds2 = mol_parser(files_path / (molecules[1] + '.sdf'))

        initial_molecule = MMoleculeObject.from_mol_file(filename=files_path /(molecules[0] + '.sdf'))
        # initial_molecule = GraphMolecule.build_from_mol(mol_file=files_path / (molecules[0] + '.sdf'))
        self.play(Create(initial_molecule))
        prev_moleucule = initial_molecule

        for molecule in molecules[1:]:
            next_molecule = MMoleculeObject.from_mol_file(filename=files_path / (molecule + '.sdf'))
            # next_molecule = GraphMolecule.build_from_mol(mol_file=files_path / (molecule + '.sdf'))
            self.play(TransformMatchingShapes(prev_moleucule, next_molecule, key_map={}), run_time=4)
            prev_moleucule = next_molecule
    
    def render(self):
        super().render(preview=True)


# # 2D Graph Molecule example
# class DrawGraphMorphine(Scene):
#     # Two D Manim Chemistry objects require Cairo renderer
#     config.renderer = "cairo"
#     def construct(self):
#         self.add(GraphMolecule.build_from_mol(mol_file=files_path / "Structure2D_COMPOUND_CID_5958.sdf"))
