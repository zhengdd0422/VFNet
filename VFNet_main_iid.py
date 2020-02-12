"""VFNet on VFG2706 or VFG740"""
import logging
from keras import backend as K
from keras.models import load_model
from sklearn.metrics import classification_report
from keras.callbacks import ModelCheckpoint
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
from keras.callbacks import LearningRateScheduler,  ReduceLROnPlateau
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import argparse
import os
import time
from sklearn.model_selection import StratifiedKFold

parser = argparse.ArgumentParser(description='VFs_classifier')
parser.add_argument('--encode', type=str, default='onehot')
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--nbt', type=int, default=128)
parser.add_argument('--file', type=str, default='VFG-2706')
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

# directory
data_dir = os.getcwd() + "/data/"
class_name_dir = data_dir + args.file + "_iid_train_class_name"
class_name = load_class_name(class_name_dir)
train_data = np.load(data_dir + args.file + "_iid_train_seq.npz")["data"]
train_labels = np.load(data_dir + args.file + "_iid_train_seq.npz")["labels"]

gc.collect()
print('train number is {}\n'.format(len(train_data)))

params = {'batch_size': args.nbt,
          'n_classes': len(class_name),
          'encode': args.encode,
          'shuffle': True}

sig = "VFNet_iid"
train_info_record = data_dir + args.file + "_record_iid/VFNet/"

if not os.path.exists(train_info_record):
    os.makedirs(train_info_record)

f_train = open(train_info_record + sig + '_train.txt', 'a')

f_train_time = time.time()
j = 0
cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=43)
for train, test in cv_outer.split(train_data, train_labels):
    print("Fold number: ", j)
    f_train.write("\n\nFold number: {}\n".format(j))
    X_train, X_test = train_data[train], train_data[test]
    Y_train, Y_test = train_labels[train], train_labels[test]
    training_generator = DataGenerator(X_train, Y_train, **params)
    validation_generator = DataGenerator(X_test, Y_test, **params)
    model = None
    model = vfnet(max_sequence_length, input_dim, len(class_name))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    VFs_model_dir = train_info_record + sig + "bestmodel_fold" + str(j + 1)
    # plot_model(model, to_file=train_info_record + sig + '_model.png', show_shapes=True, show_layer_names=False)
    checkpoint = ModelCheckpoint(filepath=VFs_model_dir, monitor='val_accuracy', save_best_only='True', mode='max',
                             save_weights_only=False)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', patience=10, mode='auto', factor=0.1)

    history = model.fit_generator(training_generator, callbacks=[checkpoint, reduce_lr],
                                  epochs=args.epoch, validation_data=validation_generator,
                                  shuffle=True, verbose=2)

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
    # test
    test_model = load_model(VFs_model_dir)
    X_test = np.array(map(seq2onehot, X_test))
    test_model = load_model(VFs_model_dir)
    test_pred = test_model.predict(X_test, batch_size=args.nbt, verbose=0)
    test_pred_a = test_pred.tolist()
    test_pred_labels = [i.index(max(i)) for i in test_pred_a]
    test_y_labels = Y_test.tolist()
    test_cm = confusion_matrix(test_y_labels, test_pred_labels)
    acc = metrics.accuracy_score(test_y_labels, test_pred_labels)
    test_cla_report = classification_report(test_y_labels, test_pred_labels, target_names=list(class_name))
    f_train.write('Test_acc is: {:.4f}\nClassfication report:\n{}\n'.format(acc, test_cla_report))
    recall_value = metrics.recall_score(test_y_labels, test_pred_labels, average='micro')
    precision_value = metrics.precision_score(test_y_labels, test_pred_labels, average='micro')
    f1_score_value = metrics.f1_score(test_y_labels, test_pred_labels, average='micro')
    recall_value_2 = metrics.recall_score(test_y_labels, test_pred_labels, average='macro')
    precision_value_2 = metrics.precision_score(test_y_labels, test_pred_labels, average='macro')
    f1_score_value_2 = metrics.f1_score(test_y_labels, test_pred_labels, average='macro')
    f_train.write('Micro\nprecision is: {:.4f}\nRecall is: {:.4f}\nF1_score is: {:.4f}\n'.format(precision_value, recall_value,
                                                                                       f1_score_value))
    f_train.write('Macro\nprecision is: {:.4f}\nRecall is: {:.4f}\nF1_score is: {:.4f}\n'.format(precision_value_2,
                                                                                           recall_value_2,
                                                                                           f1_score_value_2))
    del model, test_model, X_train, X_test, Y_train, Y_test, acc, test_cla_report, recall_value, \
        precision_value, f1_score_value, recall_value_2, precision_value_2, f1_score_value_2
    K.clear_session()
    gc.collect()
    j += 1
    break

s_train_time = time.time()
train_time = format_time(s_train_time - f_train_time)
f_train.write("5folds training time is {}\n".format(train_time))
f_train.close()

# independent test data performance
for n_fold in range(5):
    f_indep = open(train_info_record + sig + '_indep_bestmodel' + str(n_fold + 1) + '.txt', 'a')
    f_indep.write("\n\n{} bestmodel\n".format(n_fold + 1))
    # f.write("\n\n{} bestmodel\n".format(n_fold+1))
    indep_data = np.load(data_dir + args.file + "iid_indep_seq.npz")["data"]
    indep_labels = np.load(data_dir + args.file + "_iid_indep_seq.npz")["labels"]
    X_indep = np.array(map(seq2onehot, indep_data))
    t_model = load_model(train_info_record + sig + "bestmodel_fold" + str(n_fold + 1))
    indep_pred = t_model.predict(X_indep, batch_size=args.nbt, verbose=0)
    indep_pred_a = indep_pred.tolist()
    indep_pred_labels = [i.index(max(i)) for i in indep_pred_a]
    indep_y_labels = indep_labels.tolist()
    indep_cm = confusion_matrix(indep_y_labels, indep_pred_labels)
    acc = metrics.accuracy_score(indep_y_labels, indep_pred_labels)
    test_cla_report = classification_report(indep_y_labels, indep_pred_labels, target_names=list(class_name))
    f_indep.write('indep_acc is: {:.4f}\nClassfication report:\n{}\n'.format(acc, test_cla_report))
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

