#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created Oct 26, 2023 by Melvin Storbacka.

Last edited Oct 26, 2023.

"""


# for the time being, I will assume one reactant (ask Chong, do we want to include 
# reactions with more than one nucleus? what is the long-term goal?)



class Nucleus:
    def __init__(self, Z, N, mass):
        self.Z = Z
        self.N = N
        self.mass = mass


class Reaction:     
    def __init__(self, reactant_nucleus, reaction_type):    # possible reaction types: "alpha", "(n, g)", "beta" ... (and so on)
        self.reactant_nucleus = reactant_nucleus
        self.reaction_type = reaction_type

    def products(self):
        """Returns the reaction products of the reaction"""
        
        products = []

        if self.reaction_type == "(n, g)":
            products.append(Nucleus(self.reactant_nucleus.Z, self.reactant_nucleus.N))
        pass

    def Q(self):
        """Returns the Q-value of the reaction"""
        pass
    