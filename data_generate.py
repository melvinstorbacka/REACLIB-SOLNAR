#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
To use any data_generator functions, TALYS needs to be installed and the correct path to TALYS 
binary must be supplied.

Created Oct 26, 2023 by Melvin Storbacka.

Last edited Dec 4, 2023.

"""

from xml.etree import ElementTree as ET
import os
import shutil
import multiprocessing


import numpy as np




# maximal number of cores to be used for parallell computations
MAX_NUM_CORES = 10

# path to TALYS binary to be used in the calculations (will be copied into calculation folders) NB:
# Intended to use TALYS with 108 temperature steps -- see documentation.
# TODO: add some sort of documentation
TALYS_PATH = './talys'

# path to xml file for the baseline masses to be used
XML_PATH = "input_data/webnucleo-nuc2021.xml"

# number of Q-value steps
NUM_QS = 20

# binding energy per nucleon fractional step
BE_STEP = 0.02


def read_xml_baseline_masses(xml_path):
    """Reads the baseline masses from XML webnucleo file and returns array 
    with entries on form (N, Z, mass_excess).
    xml_path : path to webnucleo library file"""
    tree = ET.ElementTree(file=xml_path)
    root = tree.getroot()
    out_array = np.empty(len(root), dtype=tuple)
    for i, child in enumerate(root):
        out_array[i] = (int(child[1].text), int(child[0].text), float(child[3].text))
    return out_array

def baseline_mass_excess(nzme_array, ns, zs):
    """Returns the baseline mass excess from the stored array of N, Z and mass excess.
    nzme_array      : array formatted as (N, Z, mass_excess)
    ns              : list of neutron numbers to search for, with each z
    zs              : list of protons to search for, with each n"""
    be_out_array = np.empty(len(ns))
    for nzme in nzme_array:
        for i, (n, z) in enumerate(zip(ns, zs)):
            if nzme[0] == n and nzme[1] == z:
                be_out_array[i] = nzme[2]
    return be_out_array

def init_calculations(num_calculation_folders):
    """Checks whether a sufficient amount of folders have been created. If not, creates them
    and copies the TALYS binary into the folder.
    num_calculation_folders : the number of calculations that should run in parallel"""
    return

def perform_calculation(n, z, baseline_me, be_step, num_qs, talys_path):
    """Runs calculations for rates for a certain nucleus in calculations/calculation{PID}, then removes it.
    n               : number of neutrons
    z               : number of protons
    baseline_me     : tuple of baseline mass excesses for the calculation, (me(N), me(N+1))
    be_step         : fractional step in binding energy per nucleon between each calculation
    num_qs          : number of Q-values to be used (odd number)
    talys_path      : path to TALYS binary to be used in the calculations"""
    calculation_idx = os.getpid()
    return

def save_calculation_results(calculation_idx, n, z):
    """Moves calculation results from calculations/calculation{i}/ to
    data/{z}-{n}/, and automatically assigns the Q-value from the output file.
    calculation_idx : calculation number id (PID)
    n               : number of neutrons
    z               : number of protons"""
    return

def prepare_input(calculation_idx, n, z, mass_excesses):
    """Prepares the input file for the current calulation run.
    calculation_idx : calculation number id (PID)
    n               : number of neutrons
    z               : number of protons
    mass_excesses   : tuple of current mass excesses for the calculation, (me(N), me(N+1))"""
    return

def move_to_long_term_storage(n, z, storage_path):
    """Moves calculation results from data/{z}-{n}/ to storage_path/SONAR_data/{z}-{n}.
    n               : number of neutrons
    z               : number of protons
    storage_path    : path to long-term storage directory"""
    return


def execute(nuclei_lst, max_num_cores, talys_path, xml_path, num_qs, be_step):
    """Generates reaction rate data for passed nuclei and parameters.
    nuclei_lst      : full list of all nuclei to be changed, with entries formatted as (N, Z)
    max_num_cores   : the maximal number of cores to be allocated to the calculations
    talys_path      : path to TALYS binary to be used in the calculations
    xml_path        : path to xml file for the baseline masses to be used
    be_step         : fractional step in binding energy per nucleon between each calculation
    num_qs          : number of Q-values to be used (odd number)"""

    # preparation for calculations

    ns = []
    zs = []

    for nuc in nuclei_lst:
        zs.append(nuc[1])
        ns.append(nuc[0])

    total_baseline_me_array = read_xml_baseline_masses(xml_path)

    baseline_me = baseline_mass_excess(total_baseline_me_array, zs, ns)


    arguments = []
    for n, z, me in zip(ns, zs, baseline_me):
        arguments.append((n, z, me, be_step, num_qs, talys_path))


    # parallel computation 
    num_cores = multiprocessing.cpu_count()
    if num_cores > max_num_cores:
        num_cores = max_num_cores
    print(f"Number of available cores: {num_cores}")
    pool = multiprocessing.Pool(num_cores)


    pool.map_async(perform_calculation, arguments)

    pool.close()
    pool.join()


if __name__ == "__main__":
    a = (read_xml_baseline_masses(XML_PATH))

    print(baseline_mass_excess(a, [101, 102], [50, 50]))

