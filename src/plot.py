import os
import matplotlib.pyplot as plt
import json
import numpy as np

with open("src/symb_vs_z.json") as f:
    symbol_vs_z = json.load(f)

z_vs_symbol = {v: k for k, v in symbol_vs_z.items()}

z = 11
n = 13

dir_path = f"./data/{z}-{n}/"

files = os.listdir(dir_path)

files.sort()

num_non_exp_qs = 21
num_exp_qs = 21


z_array = np.zeros((6, num_non_exp_qs, 108))

z_array_exp = np.zeros((6, num_exp_qs, 108))

column_q_sort_exp = np.zeros(num_exp_qs)

column_q_sort = np.zeros(num_non_exp_qs)



ldmodel = 1

expid = 0
nonexpid = 0
first = True

# probably a good idea to rewrite this later...

templist = []

for idx, file_path in enumerate(files):
    print(file_path.split("-"))
    if len(file_path.split("-")) == 4:
        Q = file_path.split("-")[3].strip(".g")
        ldmodel = int(file_path.split("-")[2]) - 1
        id = int(file_path.split("-")[1])
    elif len(file_path.split("-")) == 5 and "exp" not in file_path.split("-")[-1]:
        Q = -float(file_path.split("-")[4].strip(".g"))
        ldmodel = int(file_path.split("-")[3]) - 1
        id = int(file_path.split("-")[2])
    elif len(file_path.split("-")) == 5:
        Q = -float(file_path.split("-")[3].strip(".g"))
        ldmodel = int(file_path.split("-")[2]) - 1
        id = int(file_path.split("-")[1])
    else:
        Q = -float(file_path.split("-")[4].strip(".g"))
        ldmodel = int(file_path.split("-")[3]) - 1
        id = int(file_path.split("-")[2])
    #if f"-00{ldmodel}-{Q}" in file_path:
    with open(dir_path + file_path, "r") as f:
        f.readline()
        f.readline()
        while True:
            line = f.readline()
            if not line or "Q" in line:
                break
            if first:
                templist.append(float(line.split()[0]))
            if "exp" in file_path:
                if line.split()[0] == "0.0001":
                    if float(Q) not in column_q_sort_exp:
                        column_q_sort_exp[id] = float(Q)
                    z_array_exp[ldmodel, id, 0] = float(line.split()[1])
                    for i in range(1,108):
                        line = f.readline()
                        if first:
                            templist.append(float(line.split()[0]))
                        z_array_exp[ldmodel, id, i] =  float(line.split()[1])
            else:
                print(Q, file_path)
                if line.split()[0] == "0.0001":
                    if float(Q) not in column_q_sort:
                        column_q_sort[id] = float(Q)
                    z_array[ldmodel, id, 0] = float(line.split()[1])
                    for i in range(1,108):
                        line = f.readline()
                        if first:
                            templist.append(float(line.split()[0]))
                        z_array[ldmodel, id, i] =  float(line.split()[1])
        if first:
            first = False



array = z_array
column = column_q_sort
num = num_non_exp_qs
"""

#print(array)
plt.imshow(array[:,10:], norm="log", vmin=np.min(array[:,10:]), vmax=np.max(array[:,10:]), aspect='auto', origin='lower')
cbar = plt.colorbar()
print(array[:,10:])
_cs2 = plt.contour(array[:,10:], levels = [10**(i/10) for i in range(10*int(np.floor(np.log10(np.min(array[:,10:])))), int(10*np.ceil(np.log10(np.max(array[:,10:]))))+1, 4)], origin='lower', 
                   linestyles = "dashed", colors=['black'], alpha=0.5)
plt.yticks(ticks=list(range(0, num, 5)), labels=[round((column[i]-column[10]), 5) for i in range(0, num, 5)])
plt.xticks(ticks=[i for i in range(7, 104, 10)], labels=[i/10 for i in range(10, 110, 10)])
#cbar.add_lines(_cs2)
#plt.xscale("log")
plt.title(f"Reaction rate constant vs Temperature and Q-value for {z_vs_symbol[z]}-{z + n}")
plt.xlabel(f"Temperature (GK)")
plt.ylabel(f"Difference from standard Q-value [MeV]")
cbar.ax.set_ylabel(f"Reaction rate constant", rotation=270)
plt.savefig('test.png')

# problematic width here...
# why zig-zag?

plt.clf()
plt.plot(column_q_sort, array[:, 80]/array[10,80])
#plt.yscale("log")

plt.savefig('test2.png')
"""
