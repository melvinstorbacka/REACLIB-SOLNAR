from src import data_generate
from src import dz10fit1_1
from math import floor, ceil

from random import randint

TALYS_PATH = '~/REACLIB-SOLNAR/talys'

# path to xml file for the baseline masses to be used
XML_PATH = "input_data/webnucleo-nuc2021.xml"

# path to ame20 data
AME_PATH = "input_data/ame20.txt"

# number of Q-value steps (must be odd)
NUM_QS = 21

# number of Q-value steps if either of the involved nucleus has experimental masses
NUM_QS_EXP = 21

# binding energy per nucleon fractional step
Q_STEP = 0.5

nuclei_lst = [[123, 82]]#[[80, 49], [93, 60], [13, 11]]

dz10_standard_params = [17.74799982094152, 16.25161355526155, 0.705100090804503,
          37.378328815961694, 52.40309615915015, 5.192531960013464,
          0.46472710051933575, -2.1083462345707162, 0.020788744907550675,
          41.1572619187368]


nuclei_lst = []

# estimate number of bound nuclei with 2 ≤ Z ≤ 120. TODO: add padding of additional nuclei

for z in range(2, 121):
    for n in range(2, 4*z):
        if dz10fit1_1.DZ10(n, z, dz10_standard_params) - dz10fit1_1.DZ10(n-1, z, dz10_standard_params) >= 0 and dz10fit1_1.DZ10(n, z, dz10_standard_params) - dz10fit1_1.DZ10(n, z-1, dz10_standard_params) >= 0:
            nuclei_lst.append([n, z])


comp_nuclei_lst = []

for i in range(0, 100):
    idx = randint(0, len(nuclei_lst))
    comp_nuclei_lst.append(nuclei_lst[idx])

print(comp_nuclei_lst)

data_generate.execute(comp_nuclei_lst, TALYS_PATH, AME_PATH, NUM_QS, NUM_QS_EXP, Q_STEP, data_generate.DZ10_masses, dz10_standard_params)
