import os
import matplotlib.pyplot as plt
import json
import numpy as np

with open("src/symb_vs_z.json") as f:
    symbol_vs_z = json.load(f)

z_vs_symbol = {v: k for k, v in symbol_vs_z.items()}

z = 60
n = 93

dir_path = f"./data/{z}-{n}/"

files = os.listdir(dir_path)

files.sort()

num_non_exp_qs = 21
num_exp_qs = 21


z_array = np.zeros((num_non_exp_qs, 108))

z_array_exp = np.zeros((num_exp_qs, 108))

column_q_sort_exp = np.zeros(num_exp_qs)

column_q_sort = np.zeros(num_non_exp_qs)



ldmodel = 4

expid = 0
nonexpid = 0

for idx, file_path in enumerate(files):
    Q = file_path.split("-")[3].strip(".g")
    if f"-00{ldmodel}-{Q}" in file_path:
        with open(dir_path + file_path, "r") as f:
            f.readline()
            f.readline()
            while True:
                line = f.readline()
                if not line or "Q" in line:
                    break
                if "exp" in file_path:
                    if line.split()[0] == "0.0001":
                        column_q_sort_exp[expid] = float(Q)
                        z_array_exp[expid, 0] = float(line.split()[1])
                        for i in range(1,108):
                            line = f.readline()
                            z_array_exp[expid, i] =  float(line.split()[1])
                    expid += 1
                else:
                    print(Q, file_path)
                    if line.split()[0] == "0.0001":
                        column_q_sort[nonexpid] = float(Q)
                        z_array[nonexpid, 0] = float(line.split()[1])
                        for i in range(1,108):
                            line = f.readline()
                            z_array[nonexpid, i] =  float(line.split()[1])
                    nonexpid += 1


array = z_array
column = column_q_sort
num = num_non_exp_qs

print(array)

plt.imshow(array[:,10:], norm='log', vmin=np.min(array[:,10:]), vmax=np.max(array[:,10:]), aspect='auto', origin='lower')
cbar = plt.colorbar()
print(array[:,10:])
_cs2 = plt.contour(array[:,10:], levels = [10**(i/10) for i in range(10*int(np.floor(np.log10(np.min(array[:,10:])))), int(10*np.ceil(np.log10(np.max(array[:,10:]))))+1, 1)], origin='lower', 
                   linestyles = "dashed", colors=['black'], alpha=0.5)
plt.yticks(ticks=list(range(0, num, 5)), labels=[round((column[i]-column[10])*1000, 5) for i in range(0, num, 5)])
plt.xticks(ticks=[i for i in range(7, 104, 10)], labels=[i/10 for i in range(10, 110, 10)])
#cbar.add_lines(_cs2)
#plt.xscale("log")
plt.title(f"Reaction rate constant vs Temperature and Q-value for {z_vs_symbol[z]}-{z + n}")
plt.xlabel(f"Temperature (GK)")
plt.ylabel(f"Difference from standard Q-value [keV]")
cbar.ax.set_ylabel(f"Reaction rate constant", rotation=270)
plt.savefig('test.png')

# problematic width here...
# why zig-zag?

plt.clf()
plt.plot(column_q_sort, array[:, 37]/array[10,37]) # TODO: really gotta check if this is properly sorted... go through by hand?
# There's clearly something wrong here! Wohooo! Check later :D (this is not irony)
# that is, I have to sort them before plotting (of course)

plt.savefig('test2.png')