import numpy as np
from sklearn.utils import shuffle
import argparse
import os

parser = argparse.ArgumentParser(description='convert txt to numpy for model')
parser.add_argument('--file', type=str, default='VFG-2706')  # VFG-2706/VFG-740/VFG-2706-1066/VFG-566/COG-755
parser.add_argument('--feature', type=str, default='aac')   # aac, dpc, ctd, pseaac1, pseaac2, label
args = parser.parse_args()

features_dir = os.getcwd() + "/data/" + args.file + "_features/"
f_label = open(features_dir + "labels.txt")

# lables
all_labes = []
for line in f_label.readlines():
    all_labes.append(line.strip())
all_labes = np.array(all_labes)

# features
if args.feature == "aac":
    f_open = open(features_dir + "propy_AAC.txt", 'r')
elif args.feature == "dpc":
    f_open = open(features_dir + "propy_DPC.txt", 'r')
elif args.feature == "ctd":
    f_open = open(features_dir + "propy_CTD.txt", 'r')
elif args.feature == "pseaac1":
    f_open = open(features_dir + "propy_pseaac1.txt", 'r')
    # f_open = open(features_dir + "pseb_pseaac1.txt", 'r')
elif args.feature == "pseaac2":
    f_open = open(features_dir + "propy_pseaac2.txt", 'r')
    # f_open = open(features_dir + "pseb_pseaac2.txt", 'r')

all_data = []
for line in f_open.readlines():
    line = line.strip()
    lists = line.split("\t")

    lists_float = list(map(float, lists))
    # each_data = np.array(lists).reshape(1, len(lists))
    all_data.append(lists_float)
all_data = np.array(all_data)
all_data, all_labes = shuffle(all_data, all_labes, random_state=43)
np.savez(features_dir + args.feature + "_ml", data=all_data, labels=all_labes)
f_open.close()
f_label.close()