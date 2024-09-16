"""
Module to handle interface to python from optimisation tools
"""

class PythonLatticeInterface():
    """Interface from python to a generic lattice for optimisation"""

    def __init__(self):
        """
        Initialisation, which sets up members.

        Call setup to setup the lattice.
        self.lattice_loaders: list of subclasses
        self.verbose: integer, 0 is silent, >0 is progressively more verbose
        self.substitutions: dictionary of mappings from key to value, for
             optimisations. Keys should be strings like:
                {
                    "__my_key_1__":value_1,
                    "__my_key_2__":value_2
                }
             At runtime, we apply:
                lattice_loader_element.my_key_1 = value_1
                lattice_loader_element.my_key_2 = value_2
        The substitution is also applied to self.
        """
        self.lattice_loaders = []
        self.verbose = 0 # silent
        self.substitutions = None

    def setup(self, substitutions):
        """
        Setup self. Calls setup_loader on each element of lattice_loaders.
        """
        self.setup_loader(substitutions, self)
        for loader in self.lattice_loaders:
            self.setup_loader(substitutions, loader)
        self.momentum = ((self.energy+self.mass)**2-self.mass**2)**0.5

    def setup_loader(self, substitutions, an_instance):
        """
        Setup a module
        - substitutions: dictionary of substituions to apply
        - an_instance: instance to which the substitutions will be applied

        Apply substitutions to an_instance as described in __init__
        """
        if an_instance == self:
            if "__verbose__" in subs:
                self.verbose = subs["__verbose__"]
        if self.verbose > 1:
            print("Setting up", an_instance)
        an_instance.substitutions = substitutions
        for key in an_instance.__dict__:
            subs_key = "__"+key+"__"
            if subs_key in an_instance.substitutions:
                my_type = type(an_instance.__dict__[key])
                alt_type = type(subs[subs_key])
                if self.verbose:
                    print("Applying substitution", key, subs[subs_key], "replacing", an_instance.__dict__[key], "with types", my_type, alt_type)
                an_instance.__dict__[key] = subs[subs_key]
                if self.verbose > 1:
                    print("   ... check", an_instance.__dict__[key])

