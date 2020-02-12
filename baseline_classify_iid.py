import os
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import argparse
import gc
from utils import *
from sklearn.model_selection import StratifiedKFold

parser = argparse.ArgumentParser(description='ml_features_classifier')
parser.add_argument('--feature', type=str, default='dpc')   # aac, dpc, ctd, pseaac1, pseaac2, all
parser.add_argument('--classify', type=str, default='linearsvc')  # LR, DT, RF,linearsvc,  svc,
parser.add_argument('--file', type=str, default='VFG-2706-iid')
parser.add_argument('--signal', type=int, default=13)  # 13, 23, 33, 43, 53
args = parser.parse_args()

data_dir = os.getcwd() + "/data/"
features_dir = data_dir + args.file + "_features/"
class_name_dir = data_dir + args.file + "_train_class_name"
class_name = load_class_name(class_name_dir)

record_dir = data_dir + args.file + "_record/baseline/" + args.feature + "_" + args.classify + "/"
sig = args.feature + "_" + args.classify
if not os.path.exists(record_dir):
    os.makedirs(record_dir)

if args.feature == 'all':
    train_data = np.load(features_dir + "train_aac_ml.npz", allow_pickle=True)['data']
    train_label = np.load(features_dir + "train_aac_ml.npz", allow_pickle=True)['labels']

    dpc_data = np.load(features_dir + "train_dpc_ml.npz", allow_pickle=True)['data']
    train_data = np.concatenate((train_data, dpc_data), axis=1)

    ctd_data = np.load(features_dir + "train_ctd_ml.npz", allow_pickle=True)['data']
    train_data = np.concatenate((train_data, ctd_data), axis=1)

    pseaac1_data = np.load(features_dir + "train_pseaac1_ml.npz", allow_pickle=True)['data']
    train_data = np.concatenate((train_data, pseaac1_data), axis=1)

    pseaac2_data = np.load(features_dir + "train_pseaac2_ml.npz", allow_pickle=True)['data']
    train_data = np.concatenate((train_data, pseaac2_data), axis=1)
else:
    # feature_data
    train_data = np.load(features_dir + "train_" + args.feature + "_ml.npz", allow_pickle=True)['data']
    train_label = np.load(features_dir + "train_" + args.feature + "_ml.npz", allow_pickle=True)['labels']

train_label = map(int, train_label)
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_data)
train_data = scaler.transform(train_data)

# indep
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
indep_feature_data = scaler.transform(indep_feature_data)

random_state = np.random.RandomState(0)

f_train = open(record_dir + sig + '_train.txt', 'a')
j = 0
cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=43)
for train, test in cv_outer.split(train_data, train_label):
    print("Fold number: ", j)
    f_train.write("Fold number: {}\n".format(j))
    X_train, X_test = train_data[train], train_data[test]
    Y_train, Y_test = np.array(train_label)[train], np.array(train_label)[test]

    model = None
    if args.classify == 'svc':
        model = svm.SVC(random_state=random_state, decision_function_shape='ovr')  # rbf

    elif args.classify == "linearsvc":
        model = LinearSVC(random_state=random_state)
    elif args.classify == 'RF':
        model = RandomForestClassifier(random_state=random_state, n_estimators=100)
    elif args.classify == 'LR':
        model = LogisticRegression(random_state=random_state)
    elif args.classify == 'DT':
        model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, Y_train)
    f_save_best_model_dir = record_dir + sig + '_bestmodel' + str(j + 1)
    # pickle.dump(model, open(f_save_best_model_dir, 'ab'))  # save model

    train_acc = model.score(X_train, Y_train)
    test_acc = model.score(X_test, Y_test)

    y_pred = model.predict(X_test)
    f_train.write('Best model, Training acc is {:.4f}\nval_acc is: {:.4f}\n'.format(train_acc, test_acc))

    recall_value = metrics.recall_score(Y_test, y_pred, average='micro')
    precision_value = metrics.precision_score(Y_test, y_pred, average='micro')
    f1_score_value = metrics.f1_score(Y_test, y_pred, average='micro')
    recall_value_2 = metrics.recall_score(Y_test, y_pred, average='macro')
    precision_value_2 = metrics.precision_score(Y_test, y_pred, average='macro')
    f1_score_value_2 = metrics.f1_score(Y_test, y_pred, average='macro')
    f_train.write('Micro\nprecision is: {:.4f}\nRecall is: {:.4f}\nF1_score is: {:.4f}\n'.format(precision_value,
                                                                                           recall_value,
                                                                                           f1_score_value))
    f_train.write('Macro\nprecision is: {:.4f}\nRecall is: {:.4f}\nF1_score is: {:.4f}\n'.format(precision_value_2,
                                                                                           recall_value_2,
                                                                f1_score_value_2))

    # independent test
    f_indep = open(record_dir + sig + '_indep_bestmodel' + str(j + 1) + '.txt', 'a')
    indep_pred = model.predict(indep_feature_data)
    indep_pred_labels = indep_pred.tolist()
    indep_cm = confusion_matrix(indep_feature_label, indep_pred_labels)
    indep_acc = metrics.accuracy_score(indep_feature_label, indep_pred_labels)
    indep_cla_report = classification_report(indep_feature_label, indep_pred_labels, target_names=list(class_name))
    f_indep.write('indep_acc is: {:.4f}\nClassfication report:\n{}\n'.format(indep_acc, indep_cla_report))
    # f_indep.write('Test_acc is: {:.4f}\n'.format(indep_acc))
    recall_value = metrics.recall_score(indep_feature_label, indep_pred_labels, average='micro')
    precision_value = metrics.precision_score(indep_feature_label, indep_pred_labels, average='micro')
    f1_score_value = metrics.f1_score(indep_feature_label, indep_pred_labels, average='micro')
    recall_value_2 = metrics.recall_score(indep_feature_label, indep_pred_labels, average='macro')
    precision_value_2 = metrics.precision_score(indep_feature_label, indep_pred_labels, average='macro')
    f1_score_value_2 = metrics.f1_score(indep_feature_label, indep_pred_labels, average='macro')
    f_indep.write(
        'Micro\nprecision is: {:.4f}\nRecall is: {:.4f}\nF1_score is: {:.4f}\n'.format(precision_value, recall_value,
                                                                                       f1_score_value))
    f_indep.write('Macro\nprecision is: {:.4f}\nRecall is: {:.4f}\nF1_score is: {:.4f}\n'.format(precision_value_2,
                                                                                                 recall_value_2,
                                                                                                 f1_score_value_2))
    j += 1
    f_indep.close()
    del model
    gc.collect()

f_train.close()
print("finish\n")
