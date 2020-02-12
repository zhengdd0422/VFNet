import numpy as np
from sklearn.utils import shuffle
import argparse
import os

parser = argparse.ArgumentParser(description='convert txt to numpy for model')
parser.add_argument('--feature', type=str, default='dpc')   # aac, dpc, ctd, pseaac1, pseaac2
parser.add_argument('--file', type=str, default='VFG-2706-iid')
parser.add_argument('--dataset', type=str, default='train')    # train, indep
args = parser.parse_args()

data_dir = os.getcwd() + "/data/"
features_dir = data_dir + args.file + "_features/"
f_label = open(features_dir + args.dataset + "_labels.txt", "r")

# lables
all_labes = []
for line in f_label.readlines():
    all_labes.append(line.strip())
all_labes = np.array(all_labes)

# features
if args.feature == "aac":
    f_open = open(features_dir + args.dataset + "_" + "propy_AAC.txt", 'r')
elif args.feature == "dpc":
    f_open = open(features_dir + args.dataset + "_" + "propy_DPC.txt", 'r')
elif args.feature == "ctd":
    f_open = open(features_dir + args.dataset + "_" + "propy_CTD.txt", 'r')
elif args.feature == "pseaac1":
    f_open = open(features_dir + "propy_pseaac1.txt", 'r')
    # f_open = open(features_dir + args.dataset + "_" + "pseb_pseaac1.txt", 'r')
elif args.feature == "pseaac2":
    f_open = open(features_dir + "propy_pseaac2.txt", 'r')
    # f_open = open(features_dir + args.dataset + "_" + "pseb_pseaac2.txt", 'r')


all_data = []
for line in f_open.readlines():
    line = line.strip()
    lists = line.split("\t")
    lists_float = list(map(float, lists))
    # each_data = np.array(lists).reshape(1, len(lists))
    all_data.append(lists_float)
all_data = np.array(all_data)
seq_data_new, labels_new = shuffle(all_data, all_labes, random_state=43)  # same with onehot datasets
np.savez(features_dir + args.dataset + "_" + args.feature + "_ml", data=seq_data_new, labels=labels_new)
f_open.close()
f_label.close()
