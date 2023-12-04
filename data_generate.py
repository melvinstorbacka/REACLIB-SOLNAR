#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
To use any data_generator functions, TALYS needs to be installed and the correct path to TALYS 
binary must be supplied.

Created Oct 26, 2023 by Melvin Storbacka.

Last edited Dec 4, 2023.

"""

import xml
import numpy as np
import os

# cores to be used for parallell computations
NUM_CORES = 10

# TALYS binary path (will be copied into calculation folders) NB: Intended to
# use TALYS with 108 temperature steps -- see documentation. TODO: add some sort of documentation
TALYS_PATH = './talys'

# path to xml file for the baseline masses to be used
XML_PATH = "input_data/webnucleo-nuc2021.xml"


def read_xml_baseline_masses(xml_path):
    """Reads the baseline masses from XML webnucleo file and returns array 
    with entries on form (N, Z, mass_excess).
    xml_path : path to webnucleo library file"""

def init_calculations(num_calculation_folders):
    """Checks whether a sufficient amount of folders have been created. If not, creates them
    and copies the TALYS binary into the folder.
    num_calculation_folders : the number of calculations that should run in parallel"""

def perform_calculation(calculation_idx, n, z, baseline_masses, be_step, num_qs):
    """Runs calculations for rates for a certain nucleus.
        calculation_idx : calculation number from 1 to num_cores
        n               : number of neutrons
        z               : number of protons
        baseline_masses : tuple of baseline masses for the calculation, (m(N), m(N+1))
        be_step         : percentual step in binding energy per nucleon between each calculation
        num_qs          : number of Q-values to be used (odd number)"""

def save_calculation_results(calculation_idx, n, z):
    """Moves calculation results from calculations/calculation{i}/ to
    data/{z}-{n}/, and automatically assigns the Q-value from the output file.
    calculation_idx : calculation number from 1 to num_cores
    n               : number of neutrons
    z               : number of protons
    """
