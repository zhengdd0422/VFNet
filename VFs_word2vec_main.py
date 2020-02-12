# -*- coding: utf-8 -*-
import logging
from keras import backend as K
from keras.models import load_model
from sklearn.metrics import classification_report
import tensorflow as tf
import random as rn
from sklearn.model_selection import train_test_split
from keras.utils import plot_model
from nltk import trigrams
import gensim
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from models import *
import time
from utils import *
import gc
from sklearn.metrics import confusion_matrix
from DataGenerator_all import DataGenerator
from keras.callbacks import ModelCheckpoint
from sklearn import metrics
import argparse
import os

parser = argparse.ArgumentParser(description='TTSS_classifier')
parser.add_argument('--encode', type=str, default='word2vec')  # onehot or word2vec or embed(1,2..)
parser.add_argument('--model', type=str, default='bi-gru')  # lstm
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--nbt', type=int, default=128)
parser.add_argument('--opt', type=str, default='adam')
parser.add_argument('--file', type=str, default='VFG-2706')
parser.add_argument('--signal', type=int, default='13')
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=0, inter_op_parallelism_threads=0, allow_soft_placement=True)
config_gpu = tf.ConfigProto(allow_soft_placement=True)
config_gpu.gpu_options.allow_growth = True
# session_conf.gpu_options.per_process_gpu_memory_fraction = 0.4
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

max_sequence_length = 2500  # 5000

# directory
data_dir = os.getcwd() + "/data/"
all_data = np.load(data_dir + str(args.file) + "_seq.npz", allow_pickle=True)["data"]
all_labels = np.load(data_dir + str(args.file) + "_seq.npz", allow_pickle=True)["labels"]
class_name_dir = data_dir + args.file + "_class_name"
class_name = load_class_name(class_name_dir)
word2vec_model = gensim.models.Word2Vec.load(os.getcwd() + 'word2vec_model_trembl_size_200_gensim')
# process data
texts = []
data = []
for line in all_data:
    tri_tokens = trigrams(line)
    temp_str = ""
    for item in tri_tokens:
        temp_str = temp_str + " " + item[0] + item[1] + item[2]
    texts.append(temp_str)
tokenizer = Tokenizer(num_words=10334)  # MAX_NB_WORDS = 10334
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index

# Including all trigrams in word_index
count = len(word_index)
for index, item in enumerate(word2vec_model.wv.vocab):
    if item.lower() not in word_index:
        count = count + 1
        word_index[item.lower()] = count

data = pad_sequences(sequences, maxlen=max_sequence_length)

embedding_matrix = np.zeros((10334 + 1, 200))  # why add 1 to 10335 rows??? not 10334????
for word, i in word_index.items():
    if i >= 10334:
        continue
    if word.upper() in word2vec_model.wv.vocab:
        embedding_vector = word2vec_model[word.upper()]
        embedding_matrix[i] = embedding_vector

X_train, X_test_a, Y_train, Y_test_a = train_test_split(data, all_labels, test_size=0.4, stratify=all_labels,
                                                        random_state=args.signal)  # train:test:val=6:2:2
X_test, X_val, Y_test, Y_val = train_test_split(X_test_a, Y_test_a, test_size=0.5, stratify=Y_test_a,
                                                random_state=args.signal)
del all_data, all_labels, X_test_a, Y_test_a, word2vec_model
gc.collect()

params = {'batch_size': args.nbt,
          'n_classes': len(class_name),
          'encode': args.encode,
          'shuffle': True}
# Generators
training_generator = DataGenerator(X_train, Y_train, **params)
val_generator = DataGenerator(X_val, Y_val, **params)

model = word2vec_neubi(10334 + 1, 200, max_sequence_length, matrix=embedding_matrix, n_class=len(class_name))
# parallel_model = multi_gpu_model(model, gpus=2)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
sig = "word2vec_seed" + str(args.signal)
train_info_record = data_dir + args.file + "_record/word2vec/"
if not os.path.exists(train_info_record):
    os.makedirs(train_info_record)
plot_model(model, to_file=train_info_record + sig + '_model.png', show_shapes=True, show_layer_names=False)
f = open(train_info_record + sig + '.txt', 'a')
VFs_model_dir = train_info_record + sig + '_bestmodel'

f_time = time.time()
checkpoint = ModelCheckpoint(filepath=VFs_model_dir, monitor='val_accuracy', save_best_only='True', mode='max',
                             save_weights_only=False)
history = model.fit_generator(generator=training_generator, validation_data=val_generator, epochs=args.epoch, verbose=2,
                              callbacks=[checkpoint])
loss_values = history.history['loss']  # one value per epoch
acc_values = history.history['accuracy']
val_loss_values = history.history['val_loss']
val_acc_values = history.history['val_accuracy']  # one value per epoch

epochs = range(1, len(history.history['accuracy']) + 1)
for i in range(0, len(history.history['accuracy'])):
    f.write('epoch:{}, train_loss:{:.4f}, train_acc:{:.4f}, val_loss:{:.4f}, val_acc:{:.4f}'.
            format(i + 1, loss_values[i], acc_values[i], val_loss_values[i], val_acc_values[i]) + '\n')
plot_acc_loss(train_info_record, sig, acc_values, val_acc_values, epochs, ll='acc')  # plot acc
plot_acc_loss(train_info_record, sig, loss_values, val_loss_values, epochs, ll='loss')  # plot loss

del history, loss_values, acc_values, val_loss_values, val_acc_values, X_train, X_val, Y_train, Y_val
del model
K.clear_session()
gc.collect()
s1_time = time.time()
train_times = format_time(s1_time - f_time)
f.write('training time is :{}\n'.format(train_times))

t_model = load_model(VFs_model_dir)
test_pred = t_model.predict(X_test, batch_size=args.nbt, verbose=0)
test_pred_a = test_pred.tolist()
test_pred_labels = [i.index(max(i)) for i in test_pred_a]
test_y_labels = Y_test.tolist()
test_cm = confusion_matrix(test_y_labels, test_pred_labels)
test_cla_report = classification_report(test_y_labels, test_pred_labels, target_names=list(class_name))
acc = metrics.accuracy_score(test_y_labels, test_pred_labels)
# np.save(train_info_record + sig + '_test_predicted_confusion_matrix', test_cm)
# plot_confusion_matrix(train_info_record, sig, test_cm, classes=list(class_name))
f.write('Test_acc is: {:.4f}\nClassfication report:\n{}\n'.format(acc, test_cla_report))
recall_value = metrics.recall_score(test_y_labels, test_pred_labels, average='micro')
precision_value = metrics.precision_score(test_y_labels, test_pred_labels, average='micro')
f1_score_value = metrics.f1_score(test_y_labels, test_pred_labels, average='micro')
recall_value_2 = metrics.recall_score(test_y_labels, test_pred_labels, average='macro')
precision_value_2 = metrics.precision_score(test_y_labels, test_pred_labels, average='macro')
f1_score_value_2 = metrics.f1_score(test_y_labels, test_pred_labels, average='macro')
f.write('Micro\nprecision is: {:.4f}\nRecall is: {:.4f}\nF1_score is: {:.4f}\n'.format(precision_value, recall_value,
                                                                                       f1_score_value))
f.write(
    'Macro\nprecision is: {:.4f}\nRecall is: {:.4f}\nF1_score is: {:.4f}\n'.format(precision_value_2, recall_value_2,
                                                                                   f1_score_value_2))

s_time = time.time()
test_times = format_time(s_time - s1_time)
f.write('Time is :{}\n'.format(test_times))
f.close()
