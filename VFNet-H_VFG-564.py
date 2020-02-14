# -*- coding: utf-8 -*-
import numpy as np
import os
import logging
from keras import backend as K
from keras.models import load_model
from sklearn.metrics import classification_report
import tensorflow as tf
import random as rn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras import optimizers
from models import *
from utils import *
import gc
from sklearn import metrics
import DataGenerator_all
import argparse
from encodes import onehot_encode

parser = argparse.ArgumentParser(description='VFs_classifier')
parser.add_argument('--encode', type=str, default='onehot')
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--nbt', type=int, default=128)
parser.add_argument('--opt', type=str, default='adam')
parser.add_argument('--feature', type=str, default='aac')  # aac, dpc, ctd, pseaac1, pseaac2, all
parser.add_argument('--file', type=str, default='VFG-564')
parser.add_argument('--signal', type=int, default=13)  # 13, 23, 33, 43, 53
parser.add_argument('--split_cutoff', type=float, default=0.4)
args = parser.parse_args()
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['PYTHONHASHSEED'] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
np.random.seed(42)
rn.seed(12345)
session_conf = tf.ConfigProto(allow_soft_placement=True)
session_conf.gpu_options.allow_growth = True
# session_conf.gpu_options.per_process_gpu_memory_fraction = 0.3
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
# parameters
max_sequence_length = 2500  # 5000
input_dim = 20

data_dir = os.getcwd() + "/data/"
features_dir = data_dir + args.file + "_features/"
all_data = np.load(data_dir + str(args.file) + "_seq.npz", allow_pickle=True)['data']
all_labels = np.load(data_dir + str(args.file) + "_seq.npz", allow_pickle=True)['labels']

class_name_dir = data_dir + args.file + "_class_name"
class_name = load_class_name(class_name_dir)

X_train, X_test, Y_train, Y_test = train_test_split(all_data, all_labels, test_size=args.split_cutoff, stratify=all_labels, random_state=args.signal)

if args.feature == 'all':
    feature_data = np.load(features_dir + "aac_ml.npz", allow_pickle=True)['data']
    feature_label = np.load(features_dir + "aac_ml.npz", allow_pickle=True)['labels']
    dpc_data = np.load(features_dir + "dpc_ml.npz", allow_pickle=True)['data']
    feature_data = np.concatenate((feature_data, dpc_data), axis=1)
    ctd_data = np.load(features_dir + "ctd_ml.npz", allow_pickle=True)['data']
    feature_data = np.concatenate((feature_data, ctd_data), axis=1)
    pseaac1_data = np.load(features_dir + "pseaac1_ml.npz", allow_pickle=True)['data']
    feature_data = np.concatenate((feature_data, pseaac1_data), axis=1)
    pseaac2_data = np.load(features_dir + "pseaac2_ml.npz", allow_pickle=True)['data']
    feature_data = np.concatenate((feature_data, pseaac2_data), axis=1)
else:
    # feature_data
    feature_data = np.load(features_dir + args.feature + "_ml.npz", allow_pickle=True)['data']
    feature_label = np.load(features_dir + args.feature + "_ml.npz", allow_pickle=True)['labels']

feature_label = map(int, feature_label)
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
feature_data = min_max_scaler.fit_transform(feature_data)

X_train_feature, X_test_feature, Y_train_feature, Y_test_feature = train_test_split(feature_data, feature_label, test_size=args.split_cutoff,
                                                        stratify=feature_label,
                                                        random_state=args.signal)

max_sequence_length_feature = X_train_feature.shape[1]

# check whether feature_label is same with all_labels, whether labels are same after train_test_split.==> answer is yes.
if [all_labels[i] == feature_label[i] for i in range(len(all_labels))]:
    print("all_labels is same with feature_label\n ")
else:
    print("all_labels is different with feature_label\n")

if [Y_train[i] == Y_train_feature[i] for i in range(len(Y_train))]:
    print("Y_train is same with Y_train_feature\n ")
else:
    print("Y_train is different with Y_train_feature\n")

# dataloader
# Parameters

params = {'batch_size': args.nbt,
          'n_classes': len(class_name),
          'encode': args.encode,
          }
# Generators
training_generator = DataGenerator_all.DataGenerator2inputs(X_train, Y_train, feature_data=X_train_feature, feature_label=Y_train_feature, **params)
val_generator = DataGenerator_all.DataGenerator2inputs(X_test, Y_test, feature_data=X_test_feature, feature_label=Y_test_feature, **params)

# model = two_inputs_model_fc1024(max_sequence_length, input_dim, max_sequence_length_feature, len(class_name))
model = vfneth(max_sequence_length, input_dim, max_sequence_length_feature, len(class_name))

if args.opt == "adam":
    # adam = optimizers.Adam(decay=0.01)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
elif args.opt == "sgd":
    opt = optimizers.SGD(lr=0.1, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

sig = "VFNet-H_seed" + str(args.signal) + "_" + args.feature + "_bt" + str(args.nbt)
train_info_record = data_dir + args.file + "_record/VFNet-H/"

if not os.path.exists(train_info_record):
    os.makedirs(train_info_record)

f = open(train_info_record + sig + '.txt', 'a')
VFs_model_dir = train_info_record + sig + '_bestmodel'
# plot_model(model, to_file=train_info_record + sig + '_model.png', show_shapes=True, show_layer_names=False)

save_point = 20
for epoch_idx in range(1, args.epoch + 1):
    print("Epoch: {}\n".format(epoch_idx))
    # early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=2)
    history = model.fit_generator(generator=training_generator, epochs=1, verbose=2)
    if epoch_idx % save_point == 0:
        best_model_save_dir = VFs_model_dir + str(int(epoch_idx/save_point))
        model.save(best_model_save_dir)
    del history
    gc.collect()
# test
X_test = np.array(onehot_encode(X_test))
best_model = 0
best_acc = 0
best_test_cla_report = {}
best_recall = 0
best_precision = 0
best_f1_score = 0
best_recall2 = 0
best_precision2 = 0
best_f1_score2 = 0

for j in range(1, int(args.epoch/save_point) + 1):
    trained_model_dir = VFs_model_dir + str(j)
    t_model = load_model(trained_model_dir)
    test_pred = t_model.predict([X_test, X_test_feature], batch_size=args.nbt, verbose=0)
    test_pred_a = test_pred.tolist()
    test_pred_labels = [i.index(max(i)) for i in test_pred_a]
    test_y_labels = Y_test.tolist()
    f1_score_value_2 = metrics.f1_score(test_y_labels, test_pred_labels, average='macro')
    if f1_score_value_2 > best_f1_score2:
        best_f1_score2 = f1_score_value_2
        best_model = j
        best_test_cla_report = classification_report(test_y_labels, test_pred_labels, target_names=list(class_name))
        best_acc = metrics.accuracy_score(test_y_labels, test_pred_labels)
        best_recall = metrics.recall_score(test_y_labels, test_pred_labels, average='micro')
        best_precision = metrics.precision_score(test_y_labels, test_pred_labels, average='micro')
        best_f1_score = metrics.f1_score(test_y_labels, test_pred_labels, average='micro')
        best_recall2 = metrics.recall_score(test_y_labels, test_pred_labels, average='macro')
        best_precision2 = metrics.precision_score(test_y_labels, test_pred_labels, average='macro')


f.write('best model idx: {}\nTest_acc is: {:.4f}\nClassfication report:\n{}\n'.format(best_model, best_acc, best_test_cla_report))
f.write('Micro\nprecision is: {:.4f}\nRecall is: {:.4f}\nF1_score is: {:.4f}\n'.format(best_precision, best_recall, best_f1_score))
f.write('Macro\nprecision is: {:.4f}\nRecall is: {:.4f}\nF1_score is: {:.4f}\n'.format(best_precision2, best_recall2, best_f1_score2))
f.close()
print("finish ")
