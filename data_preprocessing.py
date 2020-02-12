import pickle
from sklearn.utils import shuffle
import argparse
import gc
import numpy as np
import os
from collections import defaultdict

parser = argparse.ArgumentParser(description='VFs data_prepocess')
parser.add_argument('--file', type=str, default='VFG-2706')  # VFG-740 or VFG-2706, VFG-2706-1066, VFG-566
parser.add_argument('--process', type=str, default='make_dict')
parser.add_argument('--length_cutoff', type=int, default=0)
parser.add_argument('--number_cutoff', type=int, default=0)
args = parser.parse_args()

VFs_data_dir = os.getcwd() + "/data/"

if args.process == "make_dict":
    """make dictionary from fasta"""
    f_data = open(VFs_data_dir + args.file + ".fasta", "r")
    compid_seq = defaultdict(list)
    id_name = ""
    for line in f_data.readlines():
        line = line.strip()
        if line.find(">") >= 0:
            id_name = line.replace('>', '')
        else:
            compid_seq[id_name].append(line)
            id_name = ""

    f_dict = open(VFs_data_dir + args.file, 'ab')
    pickle.dump(compid_seq, f_dict)
    f_dict.close()
    f_data.close()

elif args.process == "seq_save":
    """dictionary to numpy(sequences + labels)"""
    f_file = open(VFs_data_dir + args.file, 'rb')
    f_class_name = open(VFs_data_dir + args.file + '_class_name', 'a')
    all_data = pickle.load(f_file)
    labels = []
    seq_data = []
    for i, each_label in enumerate(all_data.keys()):
        if len(all_data[each_label]) < 4:
            print(each_label)
        f_class_name.write("{}\n".format(each_label))
        for j in range(len(all_data[each_label])):
            labels.append(i)
            seq_data.append(all_data[each_label][j])
    del all_data
    gc.collect()
    seq_data_new, labels_new = shuffle(seq_data, labels, random_state=43)
    np.savez_compressed(VFs_data_dir + args.file + "_seq", data=seq_data_new,
                        labels=labels_new)
    f_class_name.close()
    f_file.close()

elif args.process == 'number_filter':
    """extract the classes with no more than or more than xx samples"""
    # open files
    if args.length_cutoff != 0:
        # data_name = VFs_data_dir + 'compid_seq_lm' + str(args.length_cutoff)
        data_name = VFs_data_dir + 'compid_seq_lt' + str(args.length_cutoff)
    else:
        data_name = VFs_data_dir + 'compid_seq'
    if args.number_cutoff != 0:
        save_name = data_name + '_mt' + str(args.number_cutoff)
    else:
        save_name = data_name
    # read datas
    f_data = open(data_name, 'r')
    all_data = pickle.load(f_data)
    for k in all_data.keys():
        # if len(all_data[k]) > args.number_cutoff:
        if len(all_data[k]) <= args.number_cutoff:
            all_data.pop(k)
    # save
    f_save = open(save_name, 'a')
    pickle.dump(all_data, f_save)
    f_data.close()
    f_save.close()

elif args.process == "filter_length":
    f_data = open(VFs_data_dir + args.file + ".fasta", "r")
    f_seq = open(VFs_data_dir + args.file + "_lt" + str(args.length_cutoff) + ".fasta", "a")
    id_name = ""
    for eachline in f_data.readlines():
        eachline = eachline.strip("\n")
        if eachline.find(">") >= 0:
            id_name = eachline.split("\t")[0].replace(">", "", 1)
        else:
            if len(eachline) <= 2500:
                f_seq.write(">{}\n{}\n".format(id_name, eachline))
            id_name = ""

elif args.process == 'fasta_output':
    """convert dictionary to fasta again"""
    f_file = open(VFs_data_dir + args.file, 'r')
    f_file_save = open(VFs_data_dir + args.file + ".fasta", 'a')
    all_data = pickle.load(f_file)
    for i, each_label in enumerate(all_data.keys()):
        for j in range(len(all_data[each_label])):
            f_file_save.write(">{}\n{}\n".format(each_label, all_data[each_label][j]))

    f_file_save.close()
    f_file.close()

elif args.process == 'sys':
    f_file = open(VFs_data_dir + str(args.file), 'r')
    f_result_sys = open(VFs_data_dir + str(args.file) + '_number_length_sys.txt', 'a')
    # read datas
    all_sequences_number = 0
    f_result_sys.write("compid\tnumber\tlength_min\tlength_max\n")
    all_data = pickle.load(f_file)
    for i, each_compid in enumerate(all_data.keys()):
        sizes = [len(rec) for rec in all_data[each_compid]]
        all_sequences_number += len(sizes)
        f_result_sys.write('{}\t{}\t{}\t{}\n'.format(each_compid, len(all_data[each_compid]), min(sizes), max(sizes)))

    f_result_sys.write('Total sequences number is {}'.format(all_sequences_number))
    f_result_sys.close()
    f_file.close()
