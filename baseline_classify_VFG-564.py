import os
import numpy as np
from sklearn import svm
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import argparse
import pickle
from utils import *
import logging
import random as rn
from keras import backend as K

parser = argparse.ArgumentParser(description='ml_features_classifier')
parser.add_argument('--feature', type=str, default='aac')   # aac, dpc, ctd, pseaac1, pseaac2, all
parser.add_argument('--classify', type=str, default='linearsvc')  # LR, DT, RF, svm, linearsvc
parser.add_argument('--file', type=str, default='VFG-564')
parser.add_argument('--signal', type=int, default=13)  # 13, 23, 33, 43, 53
parser.add_argument('--split_cutoff', type=float, default=0.4)  # 13, 23, 33, 43, 53
args = parser.parse_args()
np.random.seed(42)
rn.seed(12345)

data_dir = os.getcwd() + "/data/"
features_dir = data_dir + args.file + "_features/"
class_name_dir = data_dir + args.file + "_class_name"
class_name = load_class_name(class_name_dir)

record_dir = data_dir + args.file + "_record/baseline/"
if not os.path.exists(record_dir):
    os.makedirs(record_dir)

if args.feature == 'all':
    all_data = np.load(features_dir + "aac_ml.npz", allow_pickle=True)['data']
    all_labels = np.load(features_dir + "aac_ml.npz", allow_pickle=True)['labels']
    dpc_data = np.load(features_dir + "dpc_ml.npz", allow_pickle=True)['data']
    all_data = np.concatenate((all_data, dpc_data), axis=1)
    ctd_data = np.load(features_dir + "ctd_ml.npz", allow_pickle=True)['data']
    all_data = np.concatenate((all_data, ctd_data), axis=1)
    pseaac1_data = np.load(features_dir + "pseaac1_ml.npz", allow_pickle=True)['data']
    all_data = np.concatenate((all_data, pseaac1_data), axis=1)
    pseaac2_data = np.load(features_dir + "pseaac2_ml.npz", allow_pickle=True)['data']
    all_data = np.concatenate((all_data, pseaac2_data), axis=1)
else:
    all_data = np.load(features_dir + args.feature + "_ml.npz", allow_pickle=True)['data']
    all_labels = np.load(features_dir + args.feature + "_ml.npz", allow_pickle=True)['labels']

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
all_data = min_max_scaler.fit_transform(all_data)
all_labels = map(int, all_labels)
random_state = np.random.RandomState(0)

class_weights_list = compute_class_weight('balanced', np.unique(all_labels), all_labels)   # list
# convet class_weights from list to dictionary
class_weights = {}
for i, eachclass in enumerate(class_weights_list):
    class_weights[i] = class_weights_list[i]
#

X_train,  X_test, y_train, y_test, = train_test_split(all_data, all_labels, test_size=args.split_cutoff, stratify=all_labels, random_state=args.signal)
print('train number is {}\ntest number is {}\n'.format(len(X_train), len(X_test)))

model = None
if args.classify == 'svc':
    model = svm.SVC(random_state=random_state)
elif args.classify == "linearsvc":
    model = LinearSVC(random_state=random_state)
elif args.classify == 'RF':
    model = RandomForestClassifier(random_state=random_state, n_estimators=100)
elif args.classify == 'LR':
    model = LogisticRegression(random_state=random_state)
elif args.classify == 'DT':
    model = DecisionTreeClassifier(random_state=random_state)
model.fit(X_train, y_train)

save_dir = record_dir + args.feature + "_" + args.classify + "_split" + str(args.split_cutoff) + "_"
f = open(save_dir + str(args.signal) + '.txt', 'a')
f_save_best_model_dir = save_dir + str(args.signal) + '_bestmodel'
# pickle.dump(model, open(f_save_best_model_dir, 'ab'))  # save model

train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

y_pred = model.predict(X_test)
test_pred_labels = y_pred.tolist()
test_cla_report = classification_report(y_test, test_pred_labels, target_names=list(class_name))
f.write('split: {}\nTraining acc is {:.4f}\nTest_acc is: {:.4f}\nClassfication report:\n{}\n'.
        format(args.split_cutoff, train_acc, test_acc, test_cla_report))
recall_value = metrics.recall_score(y_test, test_pred_labels, average='micro')
precision_value = metrics.precision_score(y_test, test_pred_labels, average='micro')
f1_score_value = metrics.f1_score(y_test, test_pred_labels, average='micro')
recall_value_2 = metrics.recall_score(y_test, test_pred_labels, average='macro')
precision_value_2 = metrics.precision_score(y_test, test_pred_labels, average='macro')
f1_score_value_2 = metrics.f1_score(y_test, test_pred_labels, average='macro')
f.write('Micro\nprecision is: {:.4f}\nRecall is: {:.4f}\nF1_score is: {:.4f}\n'.format(precision_value,
                                                                                       recall_value,
                                                                                       f1_score_value))
f.write('Macro\nprecision is: {:.4f}\nRecall is: {:.4f}\nF1_score is: {:.4f}\n'.format(precision_value_2,
                                                                                       recall_value_2, f1_score_value_2))

# print(args.split_cutoff)
print(f1_score_value_2)
f.close()
print("finished\n")
