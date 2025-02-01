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
from  src.dz10fit1_1 import DZ10
import traceback


import numpy as np



# path to TALYS binary to be used in the calculations (will be copied into calculation folders) NB:
# Intended to use TALYS with 108 temperature./talys steps -- see documentation.
# TODO: add some sort of documentation
# needs to be full path
TALYS_PATH = '~/REACLIB-SOLNAR/talys'

# path to xml file for the baseline masses to be used
XML_PATH = "input_data/webnucleo-nuc2021.xml"

# number of Q-value steps (must be odd)
NUM_QS = 21

# Q-value step [MeV]
Q_STEP = 0.1

# proton and neutron mass in MeV, reference: https://www.nist.gov/pml/fundamental-physical-constants
PROTON_MASS_IN_MEV = 938.27208816
NEUTRON_MASS_IN_MEV = 939.56542052
ELECTRON_MASS_IN_MEV = 0.51099895000


# MeV to a.m.u. conversion, reference: https://www.nist.gov/pml/fundamental-physical-constants
MEVTOU = 931.49410242

# mass excess of a neutron, from REACLIB data
MENEUTRON = 8.07132

# standard parameters for DZ10 model
dz10_standard_params = [17.74799982094152, 16.25161355526155, 0.705100090804503,
          37.378328815961694, 52.40309615915015, 5.192531960013464,
          0.46472710051933575, -2.1083462345707162, 0.020788744907550675,
          41.1572619187368]


def FRDM_masses(xml_path, *other):
    """Reads the baseline masses from XML webnucleo file and returns array 
    with entries on form (N, Z, mass_excess).
    xml_path : path to webnucleo library file"""
    tree = ET.ElementTree(file=xml_path)
    root = tree.getroot()
    out_array = np.zeros((len(root), 4))
    for i, child in enumerate(root):
        out_array[i, 0], out_array[i, 1], out_array[i, 2], out_array[i, 3] = int(child[1].text) - int(child[0].text), int(child[0].text), float(child[3].text), 0 # uncertainty set to 0 - add check
    return out_array

def DZ10_masses(ame20_path, params, nuclei_lst):
    """Calculates baseline mass excess of non-measured masses using DZ10 model,
    and takes theoretical masses from AME20.
    ame20_path  : path to ame20 data file
    params      : parameters of dz10 model
    nuclei_lst  : list of nuclei for which we want to calculate rates """
    out_array = np.zeros((20000, 4)) # up to 10000 nuclei currently
    counter = 0
    with open(ame20_path, "r") as f:
        for _ in range(0, 36):
            f.readline()
        while True:
            line = f.readline()
            if not line:
                break
            n = int(line[6:10])
            z = int(line[11:15])
            ME = line[29:56]
            if "#" not in ME:
                ME_line = ME.split()
                out_array[counter, 0], out_array[counter, 1], out_array[counter, 2], out_array[counter, 3] = n, z, float(ME_line[0])/1000, float(ME_line[1])/1000
                counter += 1
    for nucleus in nuclei_lst:
        experimental = False
        experimental1 = False
        n = nucleus[0]
        z = nucleus[1]
        n1 = nucleus[0] + 1
        for entry in out_array:
            if entry[0] == n and entry[1] == z:
                experimental = True
            elif entry[0] == n1 and entry[1] == z:
                experimental1 = True
        if not experimental:
            out_array[counter, 0], out_array[counter, 1], out_array[counter, 2], out_array[counter, 3] = n, z, binding_energy_to_mass_excess(n, z, DZ10(n, z, params)), 0
            counter += 1
        if not experimental1:
            out_array[counter, 0], out_array[counter, 1], out_array[counter, 2], out_array[counter, 3] = n1, z, binding_energy_to_mass_excess(n1, z, DZ10(n1, z, params)), 0
            counter += 1

    # sort by neutron number, and proton number second
    column_sort = np.lexsort((out_array[:,0], out_array[:,1]))

    out_array = out_array[column_sort]

    return out_array
        

def baseline_mass_excess(nzme_array, ns, zs):
    """Returns the baseline mass excess from the stored array of N, Z and mass excess.
    nzme_array      : array formatted as (N, Z, mass_excess)
    ns              : list of neutron numbers to search for, with each z
    zs              : list of protons to search for, with each n"""
    be_out_array = np.empty(len(ns), dtype=np.ndarray)
    for i, (n, z) in enumerate(zip(ns, zs)):
        for idx, nzme in enumerate(np.vsplit(nzme_array, len(nzme_array))):
            if int(nzme[0][0]) == n and int(nzme[0][1]) == z:
                if int(nzme_array[idx + 1][0]) == n+1 and int(nzme_array[idx+1][1]) == z:
                    # The total uncertainty is here assumed to be the maximal absolute uncertainty (i.e. sum of uncertainties)
                    be_out_array[i] = np.array((nzme[0][2], nzme_array[idx+1][2], (nzme[0][3] + nzme_array[idx+1][3])))
                else:
                    be_out_array[i] = None  # if we do not have data for the product, do not calculate
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
    shutil.rmtree(f"calculations/calculation{calculation_idx}")
    return

def perform_calculation(arguments):
    """Runs calculations for rates for a certain nucleus in calculations/calculation{PID}, then removes it.
    n               : number of neutrons
    z               : number of protons
    baseline_me     : array of baseline mass excesses for the calculation, (me(N), me(N+1))
    q_step          : step in Q-value between each calculation
    num_qs          : number of Q-values to be used (odd number)
    talys_path      : path to TALYS binary to be used in the calculations"""
    n, z, baseline_mes, q_step, num_qs, talys_path, q_num, ld_idx, exp = arguments

    calculation_idx = os.getpid()

    #print("Starting" + str(arguments), flush=True)
 
    # confirmed to give a +/- (num_qs - 1)/2 even spread
    current_me = (baseline_mes[0] + (q_step)*(q_num - (num_qs - 1)/2), baseline_mes[1])
    # test what the Q-value "should" be
    q_value = (current_me[0] + MENEUTRON) - current_me[1]

    remaining_args = []

    #with open("remaining_list1.txt", "r") as f:
     #   while True:
      #      line = f.readline()
       #     if not line:
        #        break
         #   elements = line.split(",")
          #  remaining_args.append([int(elements[0]), int(elements[1]), float(elements[2]), int(elements[3]), bool(elements[4])])


    #if [n, z, np.round(q_value, 5), ld_idx, exp] not in remaining_args:
     #   print([n, z, np.round(q_value, 5), ld_idx, exp])
      #  return

    name = "|" + str(round(q_value, 5)) + "|" + f"{ld_idx:03d}" + "|"

    #with open("total_calculations_list.txt", "a+") as f: # to be used for creating list of nuclei
     #   f.write(f"{n}-{z}-{name}{exp}\n")


   #print("Checking existence of previous calculations.", flush=True)


                # NOTE: Switch back to data
    if exp:
        if os.path.exists(f"data/{z}-{n}/rate{name}-exp.g"):
            if z == 25 and n == 81:
                print(f"data/{z}-{n}/rate{name}-exp.g")
            #logging.warning(f"Skipping data/{z}-{n}/rate{name}-exp.g, already found!")
            return
    else:
        if os.path.exists(f"data/{z}-{n}/rate{name}.g"):
            if z == 25 and n == 81:
                print(f"data/{z}-{n}/rate{name}-exp.g")
            #logging.warning("Skipping " +  f"data/{z}-{n}/rate{name}.g, already found!")
            return
            
    #print(arguments, flush=True)
    with open("remaining_args", "a+") as f:
        f.write(str(arguments) + "\n")
    
    #print(q_value, flush=True)

    #print(arguments)

    """print("Initiating.", flush=True)

    init_calculation(calculation_idx)

    print("Preparing.", flush=True)

    prepare_input(calculation_idx, n, z, current_me, ld_idx, talys_path)
    def_path = os.path.abspath(os.getcwd())
    os.chdir(f"calculations/calculation{calculation_idx}")
    os.system(f"{talys_path} < input > talys.out")
    os.chdir(def_path)

    print("Done! Saving.", flush=True)

    save_calculation_results(calculation_idx, n, z, "|" + str(round(q_value, 5)) + "|" + f"{ld_idx:03d}" + "|", exp)

    print("Cleaning!", flush=True)

    clean_calculation(calculation_idx)

    print("Returning!", flush=True)
"""
    return

def save_calculation_results(calculation_idx, n, z, name, exp=False):
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
        if not exp:
            shutil.copy(f"calculations/calculation{calculation_idx}/astrorate.g",
                   f"data/{z}-{n}/rate{name}.g")
        else:
            shutil.copy(f"calculations/calculation{calculation_idx}/astrorate.g",
                   f"data/{z}-{n}/rate{name}-exp.g")
    except FileNotFoundError:
        logging.warning("Could not copy 'astrorate.g' from calculations/calculation%s/" +
                      "Does it exist? Terminating...", str(calculation_idx))
        #os.kill(os.getpid(), signal.SIGTERM)
    try:
        with open(f"calculations/calculation{calculation_idx}/talys.out") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                if "Q(n,g)" in line:
                    QVal = line
                    break
        if not exp:
            with open(f"data/{z}-{n}/rate{name}.g", "a") as f:
                f.write(QVal)
        else:
            with open(f"data/{z}-{n}/rate{name}-exp.g", "a") as f:
                f.write(QVal)
    except FileNotFoundError:
        logging.warning("Could not copy 'talys.out' from calculations/calculation%s/" +
                      "Does it exist? Terminating...", str(calculation_idx))
        #os.kill(os.getpid(), signal.SIGTERM)
    return

def prepare_input(calculation_idx, n, z, mass_excesses, num_ldmodel, talys_path):
    """Prepares the input file for the current calulation run.
    calculation_idx : calculation number id (PID)
    n               : number of neutrons
    z               : number of protons
    mass_excesses   : array of current mass excesses for the calculation, (me(N), me(N+1))
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
                        line[-3], line[-2] = str(z) + ",", str(n + z + mecount) + ","
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

def mass_excess_to_binding_energy(n, z, mass_excess):
    """Converts the input mass excess to binding energy.
    n               : number of neutrons
    z               : number of protons
    mass_excess     : mass excess in MeV"""

    nuclear_mass = mass_excess + (n + z)*MEVTOU
    binding_energy = - nuclear_mass + n*NEUTRON_MASS_IN_MEV + z*(PROTON_MASS_IN_MEV + 
                                                                 ELECTRON_MASS_IN_MEV)
    return binding_energy

def binding_energy_to_mass_excess(n, z, binding_energy):
    """Converts the input binding energy to mass excess.
    n               : number of neutrons
    z               : number of protons
    binding_energy  : binding energy in MeV
    NOTE: the binding energy and mass excess converters can be used
    interchageably, since they do exactly the same thing."""

    nuclear_mass = -binding_energy + n*NEUTRON_MASS_IN_MEV + z*(PROTON_MASS_IN_MEV + 
                                                                ELECTRON_MASS_IN_MEV)
    mass_excess = nuclear_mass - (n + z)*MEVTOU
    return mass_excess

def move_to_long_term_storage(n, z, storage_path):
    """Moves calculation results from data/{z}-{n}/ to storage_path/SONAR_data/{z}-{n}.
    n               : number of neutrons
    z               : number of protons
    storage_path    : path to long-term storage directory"""
    return


def execute(nuclei_lst, talys_path, data_path, num_qs, num_qs_exp, q_step, mass_function, params=None, exp_uncertainty_multiple=2):
    """Generates reaction rate data for passed nuclei and parameters.
    nuclei_lst                  : full list of all nuclei to be changed, with entries formatted as (N, Z)
    talys_path                  : path to TALYS binary to be used in the calculations
    data_path                   : path to xml file for the baseline masses to be used
    num_qs                      : number of Q-values to be used (odd number)
    num_qs_exp                  : number of Q-values to be used for experimental nuclei (odd number)
    q_step                      : step in Q-value between each calculation
    mass_function               : function through which to obtain the baseline masses. Currently DZ10/AME or FRDM
    params                      : parameters to use for mass function
    exp_uncertainty_multiple    : multiple of experimental uncertainty within which experimental nuclei will be sampled"""

    if num_qs % 2 == 0:
        logging.error("Number of Q-steps is not odd. Terminating...")
        os.kill(os.getpid(), signal.SIGTERM)

    # preparation for calculations

    ns = []
    zs = []

    print("test0")

    total_baseline_me_array = mass_function(data_path, params, nuclei_lst)

    for nuc in nuclei_lst:
        zs.append(nuc[1])
        ns.append(nuc[0])

    print("test1")

    baseline_me = baseline_mass_excess(total_baseline_me_array, ns, zs)

    print("test2")

    # create full list of arguments
    arguments = []
    for n, z, me in zip(ns, zs, baseline_me):
        for idx in range(num_qs):
            for ld_idx in range(1, 7): 
                if me is not None: # checks that we have data for the product
                    arguments.append((n, z, me, q_step, num_qs, talys_path, idx, ld_idx, False))
        # checks if we have uncertainty from AME20. If so, run more refined calculations
        # within +- 2*uncertainty
        if me[-1] != 0:
            for idx in range(num_qs_exp):
                for ld_idx in range(1, 7):
                    # TODO: add flag for experimental values?
                    arguments.append((n, z, me, exp_uncertainty_multiple*2*me[-1]/(num_qs_exp-1), num_qs_exp, talys_path, idx, ld_idx, True))
    
    print("test3")

    print(len(arguments))


    for arg in arguments: # to be used to creating list of required nuclei
        perform_calculation(arg)

    # parallel computation
    """num_cores = multiprocessing.cpu_count()
    print(f"Running with {num_cores} cores.")
    pool = multiprocessing.Pool(num_cores)

    pool.map_async(perform_calculation, arguments)

    pool.close()
    pool.join()"""

if __name__ == "__main__":
    print("File should be imported, then run 'execute()' with the proper arguments.")



