#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created Oct 26, 2023 by Melvin Storbacka.

Last edited Oct 27, 2023.

"""

from input_parser import element_Z_vs_name, element_name_vs_Z


# for the time being, I will assume one reactant (ask Chong, do we want to include
# reactions with more than one nucleus? what is the long-term goal?)



class Nucleus:
    """Object instance of each nuclei, containing its nuclear data"""
    def __init__(self, name, z, n, mass):
        self.name = name
        self.z = z
        self.n = n
        self.mass = mass
        # the mass must be converted to MeV at this stage
        # TODO: add mass conversion and unit option in the input parser


class NuclearNetwork:
    """Create network of nuclei based on input given from the input parser"""
    def __init__(self, list_of_nuclear_data, reaction_type_list):
        self.nuclei = self.__read_nuclear_list(list_of_nuclear_data)
        self.reactions, self.products = self.__list_of_reactions(reaction_type_list)
        self.qs = self.__q_list() # check if needed - write data_fetch.py

    def __read_nuclear_list(self, list_of_nuclei_data):
        """Reads raw input from input_parser into nuclei list in NuclearNetwork"""
        nuclei_dict = {}
        # for every nucleus in the list, we find its chemical symbol, z, and n, and
        # add a Nucleus object to nuclei_dict
        for nuc in list_of_nuclei_data:
            symb = "".join(list(filter(lambda x: not x.isdigit(), nuc[0])))
            z = element_Z_vs_name[symb]
            a = int(nuc[0].replace(symb, ""))
            n = a - z
            mass = nuc[1]
            nuclei_dict.update({(n, z) : Nucleus(symb, z, n, mass)})
        return nuclei_dict

    def __list_of_reactions(self, reaction_type_list):
        """
        Given the reaction types, returns a list of reaction objects.
        Currenty implemented reaction types: "(n, g)"
        To be implemented in the future: alpha, beta... """
        reaction_list = []
        reaction_products = []

        # allows for multiple reaction types to be called at once
        for reaction_type in reaction_type_list:
            if reaction_type == "(n, g)":
                for (n, z), nuc in self.nuclei.items():
                    # checks if the necessary data exists: if so, add reaction and products to lists
                    if (n + 1, z) in self.nuclei:
                        products = self.nuclei[(n+1, z)]
                        reaction_list.append(Reaction(nuc, "(n, g)", products))
                        reaction_products.append(products)
            elif reaction_type == "alpha":
                pass
        return reaction_list, reaction_products

    def __q_list(self): # have to determine if this is even needed
        return [reaction.Q for reaction in self.reactions]





class Reaction:
    """Object of reaction, containing necessary reaction information."""     
    def __init__(self, reactant_nucleus, reaction_type, reaction_products):
        self.reactant_nucleus = reactant_nucleus
        self.type = reaction_type
        self.products = reaction_products
        self.q = self.__q()

    def __q(self):
        """Returns the Q-value of the reaction"""
        rmass = self.reactant_nucleus.mass
        pmass = sum(product.mass for product in self.products)
        return rmass - pmass
    