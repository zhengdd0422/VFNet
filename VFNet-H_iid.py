# -*- coding: utf-8 -*-
import numpy as np
import os
import time
import logging
from keras import backend as K
from keras.models import load_model
from sklearn.metrics import classification_report
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import random as rn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from keras.utils import plot_model
from keras import optimizers
from models import *
from utils import *
import gc
from keras.callbacks import ReduceLROnPlateau
from sklearn import metrics
import DataGenerator_all
import argparse
from encodes import *
from sklearn.model_selection import StratifiedKFold


parser = argparse.ArgumentParser(description='VFs_classifier')
parser.add_argument('--encode', type=str, default='onehot')   # onehot or word2vec or embed(1,2..)
parser.add_argument('--epoch', type=int, default=2)
parser.add_argument('--nbt', type=int, default=128)
parser.add_argument('--opt', type=str, default='adam')
parser.add_argument('--feature', type=str, default='aac')  # aac, dpc, ctd, pseaac1, pseaac2, all
parser.add_argument('--file', type=str, default='VFG-2706-iid')
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

data_dir = os.getcwd() + "/data/"
features_dir = data_dir + args.file + "_features/"
class_name_dir = data_dir + args.file + "_train_class_name"
class_name = load_class_name(class_name_dir)

train_data = np.load(data_dir + args.file + "_train_seq.npz", allow_pickle=True)['data']
train_labels = np.load(data_dir + args.file + "_train_seq.npz", allow_pickle=True)['labels']
# train feature

if args.feature == 'all':
    feature_data = np.load(features_dir + "train_aac_ml.npz", allow_pickle=True)['data']
    feature_label = np.load(features_dir + "train_aac_ml.npz", allow_pickle=True)['labels']

    dpc_data = np.load(features_dir + "train_dpc_ml.npz", allow_pickle=True)['data']
    feature_data = np.concatenate((feature_data, dpc_data), axis=1)

    ctd_data = np.load(features_dir + "train_ctd_ml.npz", allow_pickle=True)['data']
    feature_data = np.concatenate((feature_data, ctd_data), axis=1)

    pseaac1_data = np.load(features_dir + "train_pseaac1_ml.npz", allow_pickle=True)['data']
    feature_data = np.concatenate((feature_data, pseaac1_data), axis=1)

    pseaac2_data = np.load(features_dir + "train_pseaac2_ml.npz", allow_pickle=True)['data']
    feature_data = np.concatenate((feature_data, pseaac2_data), axis=1)
else:
    # feature_data
    feature_data = np.load(features_dir + "train_" + args.feature + "_ml.npz", allow_pickle=True)['data']
    feature_label = np.load(features_dir + "train_" + args.feature + "_ml.npz", allow_pickle=True)['labels']

feature_label = map(int, feature_label)
if feature_label != train_labels.tolist():
    print("Error\n")

scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
scaler.fit(feature_data)
feature_data = scaler.transform(feature_data)

# indep
indep_data = np.load(data_dir + args.file + "_indep_seq.npz", allow_pickle=True)['data']
indep_label = np.load(data_dir + args.file + "_indep_seq.npz", allow_pickle=True)['labels']

if args.feature == 'all':
    indep_feature_data = np.load(features_dir + "indep_aac_ml.npz", allow_pickle=True)['data']
    indep_feature_label = np.load(features_dir + "indep_aac_ml.npz", allow_pickle=True)['labels']
    indep_dpc_data = np.load(features_dir + "indep_dpc_ml.npz", allow_pickle=True)['data']
    indep_feature_data = np.concatenate((indep_feature_data, indep_dpc_data), axis=1)
    indep_ctd_data = np.load(features_dir + "indep_ctd_ml.npz", allow_pickle=True)['data']
    indep_feature_data = np.concatenate((indep_feature_data, indep_ctd_data), axis=1)
    indep_pseaac1_data = np.load(features_dir + "indep_pseaac1_ml.npz", allow_pickle=True)['data']
    indep_feature_data = np.concatenate((indep_feature_data, indep_pseaac1_data), axis=1)
    indep_pseaac2_data = np.load(features_dir + "indep_pseaac2_ml.npz", allow_pickle=True)['data']
    indep_feature_data = np.concatenate((indep_feature_data, indep_pseaac2_data), axis=1)

else:
    indep_feature_data = np.load(features_dir + "indep_" + args.feature + "_ml.npz", allow_pickle=True)['data']
    indep_feature_label = np.load(features_dir + "indep_" + args.feature + "_ml.npz", allow_pickle=True)['labels']

indep_feature_label = map(int, indep_feature_label)

if indep_feature_label != indep_label.tolist():
    print("Error\n")

indep_feature_data = scaler.transform(indep_feature_data)

params = {'batch_size': args.nbt,
          'n_classes': len(class_name),
          'encode': args.encode,
          }

# parameters
max_sequence_length = 2500  # 5000
input_dim = 20

sig = "VFNet-H_seed" + str(args.signal) + "_" + args.feature + "_bt" + str(args.nbt)
train_info_record = data_dir + args.file + "_record_iid/VFNet-H/" + args.feature + "/"

if not os.path.exists(train_info_record):
    os.makedirs(train_info_record)

f_train = open(train_info_record + sig + '_train.txt', 'a')
f_train_time = time.time()
j = 0
cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=43)
for train, test in cv_outer.split(train_data, train_labels):
    print("Fold number: ", j)
    f_train.write("Fold number: {}\n".format(j))
    X_train, X_test = train_data[train], train_data[test]
    Y_train, Y_test = train_labels[train], train_labels[test]
    X_train_feature, X_test_feature = feature_data[train], feature_data[test]
    Y_train_feature, Y_test_feature = np.array(feature_label)[train], np.array(feature_label)[test]
    max_sequence_length_feature = X_train_feature.shape[1]
    if Y_train_feature.tolist() != Y_train.tolist() or (Y_test_feature.tolist() != Y_test.tolist()):
        print("Error\n")
    training_generator = DataGenerator_all.DataGenerator2inputs(X_train, Y_train,
                                        feature_data=X_train_feature, feature_label=Y_train_feature, **params)
    val_generator = DataGenerator_all.DataGenerator2inputs(X_test, Y_test,
                                        feature_data=X_test_feature, feature_label=Y_test_feature, **params)
    model = None
    model = vfneth(max_sequence_length, input_dim, max_sequence_length_feature, len(class_name))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    VFs_model_dir = train_info_record + sig + '_bestmodel' + str(j + 1)
    # plot_model(model, to_file=train_info_record + sig + '_model.png', show_shapes=True, show_layer_names=False)

    checkpoint = ModelCheckpoint(filepath=VFs_model_dir, monitor='val_accuracy', save_best_only='True', mode='max',
                                 save_weights_only=False)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', patience=10, mode='auto', factor=0.1)
    history = model.fit_generator(generator=training_generator, validation_data=val_generator, epochs=args.epoch,
                                       verbose=2, callbacks=[checkpoint, reduce_lr])
    loss_values = history.history['loss']  # one value per epoch
    acc_values = history.history['accuracy']
    val_loss_values = history.history['val_loss']
    val_acc_values = history.history['val_accuracy']  # one value per epoch

    epochs = range(1, len(history.history['accuracy']) + 1)
    for i in range(0, len(history.history['accuracy'])):
        f_train.write('epoch:{}, train_loss:{:.4f}, train_acc:{:.4f}, val_loss:{:.4f}, val_acc:{:.4f}'.
                format(i + 1, loss_values[i], acc_values[i], val_loss_values[i], val_acc_values[i]) + '\n')
    # plot
    # plot_acc_loss(train_info_record, sig, acc_values, val_acc_values, epochs, ll='acc')  # plot acc
    # plot_acc_loss(train_info_record, sig, loss_values, val_loss_values, epochs, ll='loss')  # plot loss
    test_model = load_model(VFs_model_dir)
    X_test = np.array(map(seq2onehot, X_test))
    test_pred = test_model.predict([X_test, X_test_feature], batch_size=args.nbt, verbose=0)
    test_pred_a = test_pred.tolist()
    test_pred_labels = [i.index(max(i)) for i in test_pred_a]
    Y_test = Y_test.tolist()
    test_cm = confusion_matrix(Y_test, test_pred_labels)
    acc = metrics.accuracy_score(Y_test, test_pred_labels)
    # test_cla_report = classification_report(test_y_labels, test_pred_labels, target_names=list(class_name))
    # f_train.write('Test_acc is: {:.4f}\nClassfication report:\n{}\n'.format(acc, test_cla_report))
    f_train.write('val_acc is: {:.4f}\n'.format(acc))
    recall_value = metrics.recall_score(Y_test, test_pred_labels, average='micro')
    precision_value = metrics.precision_score(Y_test, test_pred_labels, average='micro')
    f1_score_value = metrics.f1_score(Y_test, test_pred_labels, average='micro')
    recall_value_2 = metrics.recall_score(Y_test, test_pred_labels, average='macro')
    precision_value_2 = metrics.precision_score(Y_test, test_pred_labels, average='macro')
    f1_score_value_2 = metrics.f1_score(Y_test, test_pred_labels, average='macro')
    f_train.write(
        'Micro\nprecision is: {:.4f}\nRecall is: {:.4f}\nF1_score is: {:.4f}\n'.format(precision_value, recall_value,
                                                                                       f1_score_value))
    f_train.write('Macro\nprecision is: {:.4f}\nRecall is: {:.4f}\nF1_score is: {:.4f}\n'.format(precision_value_2,
                                                                                           recall_value_2,
                                                                                           f1_score_value_2))

    del history, loss_values, acc_values, val_loss_values, val_acc_values, X_train, X_test, Y_train, Y_test, \
        acc, recall_value, precision_value, f1_score_value, recall_value_2, precision_value_2, f1_score_value_2
    del model
    K.clear_session()
    gc.collect()
    j += 1


del train_data, train_labels
gc.collect()

s_train_time = time.time()
train_time = format_time(s_train_time - f_train_time)
f_train.write("5folds training time is {}\n".format(train_time))
f_train.close()
# independent test
for n_fold in range(5):
    f_indep = open(train_info_record + sig + '_indep_bestmodel' + str(n_fold+1) + '.txt', 'a')
    f_indep.write("\n\n{} bestmodel\n".format(n_fold+1))
    X_indep = np.array(map(seq2onehot, indep_data))
    indep_model = load_model(train_info_record + sig + '_bestmodel' + str(n_fold + 1))
    indep_pred = indep_model.predict([X_indep, indep_feature_data], batch_size=args.nbt, verbose=0)
    indep_pred = indep_pred.tolist()
    indep_pred_labels = [i.index(max(i)) for i in indep_pred]
    indep_y_labels = indep_label.tolist()
    indep_cm = confusion_matrix(indep_y_labels, indep_pred_labels)
    indep_acc = metrics.accuracy_score(indep_y_labels, indep_pred_labels)
    indep_cla_report = classification_report(indep_y_labels, indep_pred_labels, target_names=list(class_name))
    f_indep.write('indep_acc is: {:.4f}\nClassfication report:\n{}\n'.format(indep_acc, indep_cla_report))
    # f_indep.write('Test_acc is: {:.4f}\n'.format(indep_acc))
    recall_value = metrics.recall_score(indep_y_labels, indep_pred_labels, average='micro')
    precision_value = metrics.precision_score(indep_y_labels, indep_pred_labels, average='micro')
    f1_score_value = metrics.f1_score(indep_y_labels, indep_pred_labels, average='micro')
    recall_value_2 = metrics.recall_score(indep_y_labels, indep_pred_labels, average='macro')
    precision_value_2 = metrics.precision_score(indep_y_labels, indep_pred_labels, average='macro')
    f1_score_value_2 = metrics.f1_score(indep_y_labels, indep_pred_labels, average='macro')
    f_indep.write('Micro\nprecision is: {:.4f}\nRecall is: {:.4f}\nF1_score is: {:.4f}\n'.format(precision_value, recall_value, f1_score_value))
    f_indep.write('Macro\nprecision is: {:.4f}\nRecall is: {:.4f}\nF1_score is: {:.4f}\n'.format(precision_value_2, recall_value_2, f1_score_value_2))

    if n_fold == 0:
        s_test_time = time.time()
        indep_times = format_time(s_test_time - s_train_time)
        f_indep.write('indep test time is :{}\n'.format(indep_times))

    f_indep.close()

print("finish\n")
