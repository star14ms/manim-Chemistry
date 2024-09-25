# manim -p -qm chem.py ChanimScene

# pdflatex test.tex
# pdf2svg test.pdf test.svg

# latex test.tex 
# dvisvgm test.dvi --no-fonts

from chanim import *

class ChanimScene(Scene):
    def construct(self):
        ## ChemWithName creates a chemical diagram with a name label
        chem = ChemWithName("*6((=O)-N(-CH_3)-*5(-N=-N(-CH_3)-=)--(=O)-N(-H_3C)-)", "Caffeine")
        # chem = ChemWithName("CH_3-C(=O)-C(=[::-90]O)O^{-}", "Pyruvate")

        self.play(chem.creation_anim())
        self.wait()