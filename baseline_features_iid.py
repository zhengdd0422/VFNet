import pickle
from propy.PyPro import GetProDes
import argparse
import os

parser = argparse.ArgumentParser(description='extract features')
parser.add_argument('--file', type=str, default='VFG-2706-iid')
parser.add_argument('--dataset', type=str, default='train')  # train, test
parser.add_argument('--feature', type=str, default='aac')   # aac, dpc, ctd, pseaac1, pseaac2, label
args = parser.parse_args()

data_dir = os.getcwd() + "/data/"
f_file = open(data_dir + args.file + "_" + args.dataset, 'r')
all_data = pickle.load(f_file)
features_dir = data_dir + args.file + "_features/"
if not os.path.exists(features_dir):
    os.makedirs(features_dir)

if args.feature == 'aac':
    f_save = open(features_dir + args.dataset + "_propy_AAC.txt", 'a')
elif args.feature == 'dpc':
    f_save = open(features_dir + args.dataset + "_propy_DPC.txt", 'a')
elif args.feature == 'ctd':
    f_save = open(features_dir + args.dataset + "_propy_CTD.txt", 'a')
elif args.feature == 'pseaac1':
    f_save = open(features_dir + args.dataset + "_propy_pseaac1.txt", 'a')
elif args.feature == 'pseaac2':
    f_save = open(features_dir + args.dataset + "_propy_pseaac2.txt", 'a')
elif args.feature == 'label':
    f_save = open(features_dir + args.dataset + "_labels.txt", 'a')

for i, each_compid in enumerate(all_data.keys()):
    print(i)
    for j, each_sequence in enumerate(all_data[each_compid]):
        if args.feature == 'label':
            f_save.write('{}\n'.format(i))
        else:
            Des = GetProDes(each_sequence)
            # print(j)
            if args.feature == 'aac':  # group 1 AAC:20,   # DC:400
                each_value = Des.GetAAComp().values()
            elif args.feature == 'dpc':
                each_value = Des.GetDPComp().values()
            elif args.feature == 'ctd':
                each_value = Des.GetCTD().values()
            elif args.feature == 'pseaac1':  # pseaac type1
                each_value = Des.GetPAAC(lamda=10, weight=0.05).values()
            elif args.feature == 'pseaac2':  # pseaac type2
                each_value = Des.GetAPAAC(lamda=20, weight=0.05).values()

            each_value_p = '\t'.join(str(n) for n in each_value)
            f_save.write('{}\n'.format(each_value_p))

f_save.close()
