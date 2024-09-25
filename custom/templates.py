"""A few TexTemplate subclasses used internally by chanim but can be used by a user if they want to as well.
Note that setting the TeX template for a scene will affect all subsequent TexMobjects until another change in template.
"""

from manim.utils.tex import TexTemplate


class ChemTemplate(TexTemplate):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        print(1)
        self.add_to_preamble("\\usepackage{chemfig}")
        # self.add_to_preamble("\\usepackage{pgf}")
        # self.add_to_preamble("\\usepackage{tikz}")

    def set_chemfig(
        self,
        atom_sep: str = "2em", ## all of these are the defaults in chemfig
        chemfig_style:str="",
        atom_style:str="",
        angle_increment:int=45,
        bond_offset:str="2pt",
        double_bond_sep:str="2pt",
        node_style:str="",
        bond_style:str="",
    ):
        self.atom_sep = atom_sep
        self.chemfig_style = chemfig_style
        self.atom_style = atom_style
        self.angle_increment = angle_increment
        self.bond_offset = bond_offset
        self.double_bond_sep = double_bond_sep
        self.node_style = node_style
        self.bond_style = bond_style

        # set_chemfig = "\\setchemfig{atom sep=%s,chemfig style=%s, atom style=%s,angle increment=%d,bond offset=%s,double bond sep=%s, node style=%s, bond style=%s}" % (atom_sep,chemfig_style,atom_style,angle_increment,bond_offset,double_bond_sep,node_style,bond_style)
        set_chemfig = self.create_chemfig_settings()
        self.add_to_preamble(set_chemfig)

    def create_chemfig_settings(self):
        settings = {
            "atom sep": self.atom_sep,
            "chemfig style": self.chemfig_style,
            "atom style": self.atom_style,
            "angle increment": self.angle_increment,
            "bond offset": self.bond_offset,
            "double bond sep": self.double_bond_sep,
            "node style": self.node_style,
            "bond style": self.bond_style,
        }
        
        # Build the LaTeX command string, excluding empty settings
        set_chemfig_parts = []
        for key, value in settings.items():
            if value:  # Only add setting if it has a non-empty value
                if isinstance(value, int):  # Handle integers differently
                    set_chemfig_parts.append(f"{key}={value}")
                else:
                    set_chemfig_parts.append(f"{key}={{{value}}}")  # Wrap strings in braces

        set_chemfig_command = "\\setchemfig{" + ", ".join(set_chemfig_parts) + "}"
        return set_chemfig_command


class ChemReactionTemplate(TexTemplate):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.add_to_preamble("\\usepackage{chemfig}")

    def set_chemfig(
        self,

        ##Individual molecule params
        atom_sep: str = "2em", ## all of these are the defaults in chemfig
        chemfig_style:str="",
        atom_style:str="",
        angle_increment:int=45,
        bond_offset:str="2pt",
        double_bond_sep:str="2pt",
        node_style:str="",
        bond_style:str="",

        ## Reaction scheme params
        arrow_length:int=1,
        arrow_angle:int=0,
        arrow_style:int="",
        debug:str="false",
    ):
        set_chemfig = "\\setchemfig{atom sep=%s,chemfig style=%s, atom style=%s,angle increment=%d,bond offset=%s,double bond sep=%s, node style=%s, bond style=%s,scheme debug=%s, atom sep=2em, arrow angle={%d}, arrow coeff={%s}, arrow style={%s}}" % (atom_sep,chemfig_style,atom_style,angle_increment,bond_offset,double_bond_sep,node_style,bond_style,debug, arrow_angle, arrow_length, arrow_style)

        self.add_to_preamble(set_chemfig)

    def get_texcode_for_expression(self, expression):
        """Inserts expression verbatim into TeX template.

        Parameters
        ----------
        expression : :class:`str`
            The string containing the expression to be typeset, e.g. ``$\\sqrt{2}$``

        Returns
        -------
        :class:`str`
            LaTeX code based on current template, containing the given ``expression`` and ready for typesetting
        """
        return self.body.replace(self.placeholder_text, "\n\\schemestart\n"+expression+"\n\\schemestop\n\n")

    def get_texcode_for_expression_in_env(self,expression,environment):
        """Inserts an expression wrapped in a given environment into the TeX template.

        Parameters
        ----------
        environment : :class:`str`
            The environment in which we should wrap the expression.
        expression : :class:`str`
            The string containing the expression to be typeset, e.g. ``"$\\sqrt{2}$"``

        Returns
        -------
        :class:`str`
            LaTeX code based on template, containing the given expression and ready for typesetting
        """
        print(environment)
        begin = r"\begin{" + environment + "}" + "\n\\schemestart"
        end = "\\schemestop\n" + r"\end{" + environment + "}"
        print(begin,end)
        return self.body.replace(
            self.placeholder_text, "{0}\n{1}\n{2}".format(begin,expression, end)
        )
