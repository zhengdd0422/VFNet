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
from encodes import onehot_encode

parser = argparse.ArgumentParser(description='VFs_classifier')
parser.add_argument('--encode', type=str, default='onehot')   # onehot or word2vec or embed(1,2..)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--nbt', type=int, default=128)
parser.add_argument('--opt', type=str, default='adam')
parser.add_argument('--feature', type=str, default='pseaac1')  # aac, dpc, ctd, pseaac1, pseaac2, all
parser.add_argument('--file', type=str, default='VFG-2706')  # VFG-2706, VFG-740 or VFG-2706-1066
parser.add_argument('--signal', type=int, default=13)  # 13, 23, 33, 43, 53
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
all_data = np.load(data_dir + str(args.file) + "_seq.npz", allow_pickle=True)['data']
all_labels = np.load(data_dir + str(args.file) + "_seq.npz", allow_pickle=True)['labels']

class_name_dir = data_dir + args.file + "_class_name"
class_name = load_class_name(class_name_dir)
features_dir = data_dir + args.file + "_features/"
X_train, X_test_a, Y_train, Y_test_a = train_test_split(all_data, all_labels, test_size=0.4, stratify=all_labels,
                                                random_state=args.signal)
X_test, X_val, Y_test, Y_val = train_test_split(X_test_a, Y_test_a, test_size=0.5, stratify=Y_test_a,
                                               random_state=args.signal)
print('onehot train number is {}\ntest number is {}\n val number is {}\n'.format(len(X_train), len(X_test), len(X_val)))

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
data_minmax = min_max_scaler.fit_transform(feature_data)

X_train_feature, X_test_a_feature, Y_train_feature, Y_test_a_feature = train_test_split(data_minmax, feature_label,
                test_size=0.4, stratify=feature_label, random_state=args.signal)
X_test_feature, X_val_feature, Y_test_feature, Y_val_feature = train_test_split(X_test_a_feature,
                test_size=0.5, stratify=Y_test_a_feature, random_state=args.signal)
print('features train number is {}\ntest number is {}\n val number is {}\n'.format(len(X_train_feature), len(X_test_feature),
                                                                          len(X_val_feature)))
max_sequence_length_feature = X_train_feature.shape[1]

# dataloader, Parameters
params = {'batch_size': args.nbt,
          'n_classes': len(class_name),
          'encode': args.encode,
          }
# Generators
training_generator = DataGenerator_all.DataGenerator2inputs(X_train, Y_train,
                                    feature_data=X_train_feature, feature_label=Y_train_feature, **params)
val_generator = DataGenerator_all.DataGenerator2inputs(X_val, Y_val,
                                    feature_data=X_val_feature, feature_label=Y_val_feature, **params)
model = None
model = vfneth(max_sequence_length, input_dim, max_sequence_length_feature, len(class_name))

if args.opt == "adam":
    # adam = optimizers.Adam(decay=0.01)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
elif args.opt == "sgd":
    opt = optimizers.SGD(lr=0.1, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

sig = "VFNet-H_seed" + str(args.signal) + "_" + args.feature + "_bt" + str(args.nbt)
train_info_record = data_dir + args.file + "_record/VFNet-H/seed" + str(args.signal) + "/"
if not os.path.exists(train_info_record):
    os.makedirs(train_info_record)

f = open(train_info_record + sig + '.txt', 'a')
VFs_model_dir = train_info_record + sig + '_bestmodel'

f_train_time = time.time()
plot_model(model, to_file=train_info_record + sig + '_model.png', show_shapes=True, show_layer_names=False)

checkpoint = ModelCheckpoint(filepath=VFs_model_dir, monitor='val_accuracy', save_best_only='True', mode='max',
                             save_weights_only=False)
# reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', patience=5, mode='auto', factor=0.1)
history = model.fit_generator(generator=training_generator, validation_data=val_generator, epochs=args.epoch,
                                   verbose=2, callbacks=[checkpoint])
loss_values = history.history['loss']  # one value per epoch
acc_values = history.history['accuracy']
val_loss_values = history.history['val_loss']
val_acc_values = history.history['val_accuracy']  # one value per epoch

epochs = range(1, len(history.history['accuracy']) + 1)
for i in range(0, len(history.history['accuracy'])):
    f.write('epoch:{}, train_loss:{:.4f}, train_acc:{:.4f}, val_loss:{:.4f}, val_acc:{:.4f}'.
            format(i + 1, loss_values[i], acc_values[i], val_loss_values[i], val_acc_values[i]) + '\n')
# plot
plot_acc_loss(train_info_record, sig, acc_values, val_acc_values, epochs, ll='acc')  # plot acc
plot_acc_loss(train_info_record, sig, loss_values, val_loss_values, epochs, ll='loss')  # plot loss

del history, loss_values, acc_values, val_loss_values, val_acc_values, X_train, X_val, Y_train, Y_val
del model
K.clear_session()
gc.collect()
s_train_time = time.time()
train_time = format_time(s_train_time - f_train_time)
f.write("training time is {}\n".format(train_time))

# independent test
X_test = np.array(onehot_encode(X_test))
t_model = load_model(VFs_model_dir)
test_pred = t_model.predict([X_test, X_test_feature], batch_size=args.nbt, verbose=0)
test_pred_a = test_pred.tolist()
test_pred_labels = [i.index(max(i)) for i in test_pred_a]
test_y_labels = Y_test.tolist()
test_cm = confusion_matrix(test_y_labels, test_pred_labels)
acc = metrics.accuracy_score(test_y_labels, test_pred_labels)
test_cla_report = classification_report(test_y_labels, test_pred_labels, target_names=list(class_name))
f.write('Test_acc is: {:.4f}\nClassfication report:\n{}\n'.format(acc, test_cla_report))
recall_value = metrics.recall_score(test_y_labels, test_pred_labels, average='micro')
precision_value = metrics.precision_score(test_y_labels, test_pred_labels, average='micro')
f1_score_value = metrics.f1_score(test_y_labels, test_pred_labels, average='micro')
recall_value_2 = metrics.recall_score(test_y_labels, test_pred_labels, average='macro')
precision_value_2 = metrics.precision_score(test_y_labels, test_pred_labels, average='macro')
f1_score_value_2 = metrics.f1_score(test_y_labels, test_pred_labels, average='macro')
f.write('Micro\nprecision is: {:.4f}\nRecall is: {:.4f}\nF1_score is: {:.4f}\n'.format(precision_value, recall_value, f1_score_value))
f.write('Macro\nprecision is: {:.4f}\nRecall is: {:.4f}\nF1_score is: {:.4f}\n'.format(precision_value_2, recall_value_2, f1_score_value_2))
s_test_time = time.time()
test_times = format_time(s_test_time - s_train_time)
f.write('Test time is :{}\n'.format(test_times))
f.close()
print("finish\n")
