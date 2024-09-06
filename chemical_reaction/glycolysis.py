from manim import Scene, ThreeDScene, config
from manim import *
from manim_chemistry import (
    MMoleculeObject,
    GraphMolecule,
)

from pathlib import Path


script_path = Path(__file__).absolute().parent.parent
files_path = script_path / "glycolysis"
font = 'Comic Sans MS'
molecules = [
    'd-glucose',
    'glucose-6-phosphate',
    'fructose-6-phosphate',
    'fructose-1,6-bisphosphate',
    'dihydroxyacetone phosphate',
    'glyceraldehyde-3-phosphate',
    '1,3-bisphosphoglycerate',
    '3-phosphoglycerate',
    '2-phosphoglycerate',
    'phosphoenolpyruvate',
    'pyruvate',
]


# 2D Molecule example
class Draw2DMorphine(Scene):
    # Two D Manim Chemistry objects require Cairo renderer
    config.renderer = "cairo"
    def construct(self):    
        initial_title = Text(molecules[0], font_size=64, font=font).to_edge(UP)
        initial_molecule = MMoleculeObject.from_mol_file(filename=files_path /(molecules[0] + '.sdf'))
        self.play(Write(initial_title), Create(initial_molecule), run_time=2)
        self.wait(duration=2)
        prev_title = initial_title
        prev_moleucule = initial_molecule

        for title in molecules[1:]:
            next_title = Text(title, font_size=64, font=font).to_edge(UP)
            next_molecule = MMoleculeObject.from_mol_file(filename=files_path / (title + '.sdf'))
            self.play(TransformMatchingShapes(prev_title, next_title), TransformMatchingShapes(prev_moleucule, next_molecule, key_map={}), run_time=2)
            self.wait(duration=2)
            prev_moleucule = next_molecule
            prev_title = next_title

    def render(self):
        super().render(preview=True)


# # 2D Graph Molecule example
# class DrawGraphMorphine(Scene):
#     # Two D Manim Chemistry objects require Cairo renderer
#     config.renderer = "cairo"
#     def construct(self):
#         self.add(GraphMolecule.build_from_mol(mol_file=files_path / "Structure2D_COMPOUND_CID_5958.sdf"))
