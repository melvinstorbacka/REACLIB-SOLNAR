import numpy as np


def read(files, dir_path):

    QT_points = [[] for i in range(1, 7)]
    rate_points = [[] for i in range(1, 7)]
    templist = [[] for i in range(1, 7)]
    qlist = [[] for i in range(1, 7)]

    for file_path in files:
        Q, idx, ld_idx = float(file_path.split("|")[1]), int(file_path.split("|")[2]), int(file_path.split("|")[3])
        with open(dir_path + file_path, "r") as f:
            f.readline()
            f.readline()
            line = f.readline()
            line = line.split(" ")
            if len(line) == 5:
                    temperature, rate = float(line[2]), float(line[3])
            else:
                temperature, rate = float(line[1]), float(line[2])
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
                if len(line) == 5:
                    temperature, rate = float(line[2]), float(line[3])
                else:
                    temperature, rate = float(line[1]), float(line[2])
                if temperature not in templist[ld_idx - 1]:
                    templist[ld_idx-1].append(temperature)
                if rate != 0:
                    rate = np.log2(rate)
                else:
                    rate = np.log2(1e-30)
                rate_points[ld_idx-1].append(rate)
                ind = templist[ld_idx - 1].index(temperature)
                QT_points[ld_idx-1].append((Q, temperature))
        

    # these arrays need to be flattened or select one of the dimensions (different LD models)
    return np.array(QT_points), np.array(rate_points), qlist[1], templist[1]