import numpy as np
import signal
import os


def read(files, dir_path):

    QT_points = [[] for i in range(1, 7)]
    rate_points = [[] for i in range(1, 7)]
    templist = [[] for i in range(1, 7)]
    qlist = [[] for i in range(1, 7)]
    error_lst = []

    for file_path in files:
        Q, ld_idx = float(file_path.split("|")[1]), int(file_path.split("|")[2])
        with open(dir_path + file_path, "r") as f:
            for _ in range(20):
                f.readline()
            line = f.readline()
            line = line.split(" ")
            temperature, rate = float(line[3]), float(line[6])
            QT_points[ld_idx-1].append((Q, temperature))
            if temperature not in templist[ld_idx - 1]:
                templist[ld_idx-1].append(temperature)
            ind = templist[ld_idx - 1].index(temperature)
            qlist[ld_idx - 1].append(Q)
            if rate != 0:
                rate = np.log2(rate)
            else:
                rate = np.log2(1e-30)
            rate_points[ld_idx-1].append(rate)
            while True:
                line = f.readline()
                if not line or "Q" in line:
                    break
                line = line.split(" ")
                try:
                    temperature, rate = float(line[3]), float(line[6])
                    if temperature not in templist[ld_idx - 1]:
                        templist[ld_idx-1].append(temperature)
                    if rate != 0:
                        rate = np.log2(rate)
                    else:
                        rate = np.log2(1e-30)
                    rate_points[ld_idx-1].append(rate)
                    ind = templist[ld_idx - 1].index(temperature)
                    QT_points[ld_idx-1].append((Q, temperature))
                except:
                    print(f"WARNING, NaN caluclation results detected in {dir_path}/{file_path}!", flush=True)
                    with open("data_read_warnings.out", "a+") as g:
                        g.write(f"WARNING, NaN caluclation results detected in {dir_path}/{file_path}!\n")
                        g.close()
                    rate_points[ld_idx-1].append(-10000)
                    QT_points[ld_idx-1].append((Q, temperature))
                    if ld_idx not in error_lst:
                        error_lst.append(ld_idx)

    # these arrays need to be flattened or select one of the dimensions (different LD models)
    return np.array(QT_points), np.array(rate_points), qlist[1], templist[1], error_lst