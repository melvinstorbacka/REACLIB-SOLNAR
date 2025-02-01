from src import data_generate
from src import dz10fit1_1
from math import floor, ceil
import os
import json

from random import randint
import matplotlib.pyplot as plt

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

# estimate number of bound nuclei with 3 ≤ Z ≤ 120. TODO: add padding of additional nuclei

for z in range(3, 121):
    for n in range(3, 4*z): # this MUST be a three
        if dz10fit1_1.DZ10(n+1, z, dz10_standard_params) - dz10fit1_1.DZ10(n, z, dz10_standard_params) >= -1.5 and dz10fit1_1.DZ10(n, z, dz10_standard_params) - dz10fit1_1.DZ10(n, z-1, dz10_standard_params) >= -1.5:
            nuclei_lst.append([n, z])



print(len(nuclei_lst))

#plt.scatter([nuc[0] for nuc in nuclei_lst], [nuc[1] for nuc in nuclei_lst ], s=2)
#plt.scatter([126], [50], color="black")
#plt.savefig("testchart")

with open("filelist.json") as f:
    file_dict = json.load(f)

data_path = 'data/'

print(f'Completed nuclei: {len([nuc for nuc in os.listdir(f"{data_path}") if len(os.listdir(f"{data_path}{nuc}/")) == file_dict[nuc]])} of {len(nuclei_lst)}')
print(f'Completed nuclei: {sum([len(os.listdir(f"{data_path}{nuc}/"))/sum(file_dict.values()) for nuc in os.listdir(f"{data_path}")])}')

with open("completed_list.txt", "w") as f:
    for nuc in os.listdir(f"{data_path}"):
        if len(os.listdir(f"{data_path}{nuc}/"))/sum(file_dict.values()):
            f.write(str(nuc) + "\n")


for n in range(1, 11):
    print(f"{n = }")
    print(f'Completed nuclei: {len([nuc for nuc in os.listdir(f"{data_path}") if len(os.listdir(f"{data_path}{nuc}/")) == file_dict[nuc] and nuc in list(file_dict.keys())[1000*(n-1):min(1000*n, len(file_dict))]])} of {len((list(file_dict)[1000*(n-1):min(1000*n, len(file_dict))]))}')
    print(f'Completed calculations: {sum([len(os.listdir(f"{data_path}{nuc}/"))/sum((list(file_dict.values())[1000*(n-1):min(1000*n, len(file_dict))])) for nuc in os.listdir(data_path) if nuc in list(file_dict.keys())[1000*(n-1):min(1000*n, len(file_dict))]])}')



print(sum([len(os.listdir(f"{data_path}{nuc}/")) for nuc in os.listdir(f"{data_path}")]), sum(file_dict.values()))

#comp_nuclei_lst = []

"""for i in range(0, 100):
    idx = randint(0, len(nuclei_lst))
    comp_nuclei_lst.append(nuclei_lst[idx])"""

#print(comp_nuclei_lst)

#comp_nuclei_lst = []
"""
for nucdir in os.listdir("data/"):
    z, n = nucdir.split("-")
    comp_nuclei_lst.append([int(n), int(z)])
""" 
print(len(nuclei_lst))

n = 3

#with open("filelist.json") as f:
 #   file_dict = json.load(f)

#with open("nuclist.json") as f:
 #   nuclei_lst = json.load(f)

"""
data_path = 'data/'

print(f'Completed nuclei: {len([nuc for nuc in os.listdir(f"{data_path}") if len(os.listdir(f"{data_path}{nuc}/")) == file_dict[nuc] and nuc in list(file_dict.keys())[1000*(n-1):min(1000*n, len(file_dict))]])} of {len((list(file_dict)[1000*(n-1):min(1000*n, len(file_dict))]))}')
print(f'Completed calculations: {sum([len(os.listdir(f"{data_path}{nuc}/"))/sum((list(file_dict.values())[1000*(n-1):min(1000*n, len(file_dict))])) for nuc in os.listdir(data_path) if nuc in list(file_dict.keys())[1000*(n-1):min(1000*n, len(file_dict))]])}')

print(nuclei_lst)

with open("nuclist.json", "w") as f:
    json.dump(nuclei_lst, f)"""

#'print([145, 91] in nuclei_lst)

comp_nuclei_lst = nuclei_lst#[1000*(n-1):min(1000*n, len(nuclei_lst))]

print(comp_nuclei_lst[-1])

data_generate.execute(comp_nuclei_lst, TALYS_PATH, AME_PATH, NUM_QS, NUM_QS_EXP, Q_STEP, data_generate.DZ10_masses, dz10_standard_params)

# NOTE: I will have to copy all non-exp with rsync, then pick out the correct exp masses later (at the end)
