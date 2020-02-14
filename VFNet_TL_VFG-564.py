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
from keras.utils import plot_model
from keras import optimizers
from models import *
from utils import *
from encodes import *
import gc
from DataGenerator_all import DataGenerator
from sklearn import metrics
import argparse

parser = argparse.ArgumentParser(description='VFs_classifier')
parser.add_argument('--encode', type=str, default='onehot')   # onehot or word2vec or embed(1,2..)
parser.add_argument('--model', type=str, default='ly2')   #
parser.add_argument('--epoch', type=int, default=800)
parser.add_argument('--nbt', type=int, default=128)
parser.add_argument('--opt', type=str, default='adam')
parser.add_argument('--file', type=str, default='VFG-564')
parser.add_argument('--signal', type=int, default=13)  # 13, 23, 33, 43, 53
parser.add_argument('--gpuid', type=str, default="0")  
parser.add_argument('--split_cutoff', type=float, default=0.4)
parser.add_argument('--pretrain_method', type=str, default="fixnodense")
parser.add_argument('--lr', type=float, default=0.0001)
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
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

# directory
data_dir = os.getcwd() + "/data/"
class_name_dir = data_dir + args.file + "_class_name"
class_name = load_class_name(class_name_dir)
all_data = np.load(data_dir + str(args.file) + "_seq.npz")["data"]
all_labels = np.load(data_dir + str(args.file) + "_seq.npz")["labels"]

X_train,  X_test, Y_train, Y_test, = train_test_split(all_data, all_labels, test_size=args.split_cutoff, stratify=all_labels, random_state=args.signal)
print('train number is {}\ntest number is {}\n'.format(len(X_train), len(X_test)))

# dataloader parameters
params = {'batch_size': args.nbt,
          'n_classes': len(class_name),
          'encode': args.encode,
          'shuffle': True}
training_generator = DataGenerator(X_train, Y_train, **params)
val_generator = DataGenerator(X_test, Y_test, **params)

model = None
pretrained_model = load_model(os.getcwd() + "COG-755_record/VFNet/VFNet_seed13_bestmodel")
if args.pretrain_method == "cnnonly":
    # only take concatenate layer
    x = pretrained_model.layers[-4].output
    x = Dense(len(class_name), activation='softmax', name='prediction', trainable=True)(x)
    model = Model(inputs=pretrained_model.inputs, output=x)
    for layer in model.layers[:-1]:
        layer.trainable = False

# fix non dense
elif args.pretrain_method == "fixnodense":
    x = pretrained_model.layers[-2].output
    x = Dense(len(class_name), activation='softmax', name='prediction', trainable=True)(x)
    model = Model(inputs=pretrained_model.inputs, output=x)
    for layer in model.layers[:-3]:
        layer.trainable = False

elif args.pretrain_method == "fixall":
    # fix all
    x = pretrained_model.layers[-2].output
    x = Dense(len(class_name), activation='softmax', name='prediction', trainable=True)(x)
    model = Model(inputs=pretrained_model.inputs, output=x)
    for layer in model.layers[:-1]:
        layer.trainable = False

print(model.summary())
adam = optimizers.Adam(lr=args.lr)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

sig = "VFNet-H_TL_seed" + str(args.signal) + "_" + args.feature + "_bt" + \
      str(args.nbt) + args.pretrain_method + "_lr" + str(args.lr) + "_split" + str(args.split_cutoff)

train_info_record = data_dir + args.file + "_record/VFNet_TL/"
if not os.path.exists(train_info_record):
    os.makedirs(train_info_record)

f = open(train_info_record + sig + '.txt', 'a')
VFs_model_dir = train_info_record + sig + '_bestmodel'
plot_model(model, to_file=train_info_record + sig + '_model.png', show_shapes=True, show_layer_names=False)

save_point = 100
for epoch_idx in range(1, args.epoch + 1):
    print("Epoch: {}\n".format(epoch_idx))
    history = model.fit_generator(generator=training_generator, epochs=1, verbose=2)
    if epoch_idx % save_point == 0:
        best_model_save_dir = VFs_model_dir + str(int(epoch_idx/save_point))
        model.save(best_model_save_dir)
    del history
    gc.collect()

# independent test
X_test = np.array(map(seq2onehot, X_test))
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
    test_pred = t_model.predict(X_test, batch_size=args.nbt, verbose=0)
    test_pred_a = test_pred.tolist()
    test_pred_labels = [i.index(max(i)) for i in test_pred_a]
    test_y_labels = Y_test.tolist()
    f1_score_value_2 = metrics.f1_score(test_y_labels, test_pred_labels, average='macro')
    if f1_score_value_2 > best_f1_score2:
        best_model = j
        best_test_cla_report = classification_report(test_y_labels, test_pred_labels, target_names=list(class_name))
        best_acc = metrics.accuracy_score(test_y_labels, test_pred_labels)
        best_recall = metrics.recall_score(test_y_labels, test_pred_labels, average='micro')
        best_precision = metrics.precision_score(test_y_labels, test_pred_labels, average='micro')
        best_f1_score = metrics.f1_score(test_y_labels, test_pred_labels, average='micro')
        best_recall2 = metrics.recall_score(test_y_labels, test_pred_labels, average='macro')
        best_precision2 = metrics.precision_score(test_y_labels, test_pred_labels, average='macro')
        best_f1_score2 = f1_score_value_2

f.write('best model idx: {}\nTest_acc is: {:.4f}\nClassfication report:\n{}\n'.format(best_model, best_acc, best_test_cla_report))
f.write('Micro\nprecision is: {:.4f}\nRecall is: {:.4f}\nF1_score is: {:.4f}\n'.format(best_precision, best_recall, best_f1_score))
f.write('Macro\nprecision is: {:.4f}\nRecall is: {:.4f}\nF1_score is: {:.4f}\n'.format(best_precision2, best_recall2, best_f1_score2))
f.close()
print("finish ")
