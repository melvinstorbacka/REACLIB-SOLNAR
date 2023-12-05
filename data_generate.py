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
import signal
import shutil
import multiprocessing
import logging


import numpy as np



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

# proton and neutron mass in MeV, reference: https://www.nist.gov/pml/fundamental-physical-constants
PROTON_MASS_IN_MEV = 938.27208816
NEUTRON_MASS_IN_MEV = 939.56542052
ELECTRON_MASS_IN_MEV = 0.51099895000


# MeV to a.m.u. conversion, reference: https://www.nist.gov/pml/fundamental-physical-constants
MEVTOU = 931.49410242



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

def init_calculation(calculation_idx):
    """Creates a folder for a particular calculation and copies the 
    TALYS binary into the folder.
    calculation_idx   : the PID of the current process"""
    if "calculations" not in os.listdir():
        os.mkdir("calculations")
    try:
        os.mkdir(f"calculations/calculation{calculation_idx}")
    except FileExistsError:
        logging.warning("Directory calculations/calculation%s already exists. Will be overwritten.",
                        str(calculation_idx))
    return

def clean_calculation(calculation_idx):
    """Removes the folder for the particular calculation
    calculation_idx   : the PID of the current process"""
    os.rmdir(f"calculations/calculation{calculation_idx}")
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
    # TODO: add calculation of mass excess to be used in each run
    return

def save_calculation_results(calculation_idx, n, z):
    """Moves calculation results from calculations/calculation{i}/ to
    data/{z}-{n}/, and automatically assigns the Q-value from the output file.
    calculation_idx : calculation number id (PID)
    n               : number of neutrons
    z               : number of protons"""
    if not "data" in os.listdir():
        os.mkdir("data")
    if not f"{z}-{n}" in os.listdir("data"):
        os.mkdir(f"data/{z}-{n}")
    try:
        shutil.copy(f"calculations/calculation{calculation_idx}/astrorate.g",
                    "data/{z}-{n}/astrorate.g")
        # TODO: add saving of Q-value from talys.out to this file
    except FileNotFoundError:
        logging.error("Could not copy 'astrorate.g' from calculations/calculation%s/" +
                      "Does it exist? Terminating...", str(calculation_idx))
        os.kill(os.getpid(), signal.SIGTERM)
    return

def prepare_input(calculation_idx, n, z, mass_excesses, num_ldmodel):
    """Prepares the input file for the current calulation run.
    calculation_idx : calculation number id (PID)
    n               : number of neutrons
    z               : number of protons
    mass_excesses   : tuple of current mass excesses for the calculation, (me(N), me(N+1))
    num_ldmodel     : the number for ldmodel option to be used in TALYS"""
    try:
        with open("def_input", "r", encoding="utf8") as f:
            with open(f"calculations/calculation{calculation_idx}/input", "w",
                      encoding="utf8") as g:
                mecount = 0
                while True:
                    line = f.readline()
                    if not line:
                        break
                    line = line.split()
                    if "element" in line:
                        line[-1] = str(z)
                    elif "mass" in line:
                        line[-1] = str(z + n)
                    elif "massexcess" in line:
                        line[-3], line[-2] = str(z) + ",", str(n + z) + ","
                        line[-1] = str(mass_excesses[mecount])
                        mecount += 1
                    elif "ldmodel" in line:
                        line[-1] = str(num_ldmodel)
                    line.append("\n")
                    g.write(" ".join(line))
    except FileNotFoundError:
        logging.error("File 'def_input' not found. Does it exist? Terminating...")
        os.kill(os.getpid(), signal.SIGTERM)
    return

def mass_excess_to_bindning_energy(n, z, mass_excess):
    """Converts the input mass excess to binding energy.
    n               : number of neutrons
    z               : number of protons
    mass_excess     : mass excess in MeV"""

    nuclear_mass = mass_excess + (n + z)*MEVTOU
    binding_energy = - nuclear_mass + n*NEUTRON_MASS_IN_MEV + z*(PROTON_MASS_IN_MEV + ELECTRON_MASS_IN_MEV)
    return binding_energy

def binding_energy_to_mass_excess(n, z, binding_energy):
    """Converts the input binding energy to mass excess.
    n               : number of neutrons
    z               : number of protons
    binding_energy  : binding energy in MeV"""

    nuclear_mass = -binding_energy + n*NEUTRON_MASS_IN_MEV + z*(PROTON_MASS_IN_MEV + ELECTRON_MASS_IN_MEV)
    mass_excess = nuclear_mass - (n + z)*MEVTOU
    print(nuclear_mass/MEVTOU)
    return mass_excess

def move_to_long_term_storage(n, z, storage_path):
    """Moves calculation results from data/{z}-{n}/ to storage_path/SONAR_data/{z}-{n}.
    n               : number of neutrons
    z               : number of protons
    storage_path    : path to long-term storage directory"""
    return


def execute(nuclei_lst, talys_path, xml_path, num_qs, be_step):
    """Generates reaction rate data for passed nuclei and parameters.
    nuclei_lst      : full list of all nuclei to be changed, with entries formatted as (N, Z)
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
    print(f"Number of available cores: {num_cores}")
    pool = multiprocessing.Pool(num_cores)


    pool.map_async(perform_calculation, arguments)

    pool.close()
    pool.join()


if __name__ == "__main__":
    init_calculation(1)

    print(mass_excess_to_bindning_energy(62, 50, -88.6579))
    print(binding_energy_to_mass_excess(62, 50, mass_excess_to_bindning_energy(62, 50, -88.6579)))

