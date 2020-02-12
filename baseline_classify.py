import os
import numpy as np
from sklearn import svm, datasets
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
import time
from utils import *

parser = argparse.ArgumentParser(description='ml_features_classifier')
parser.add_argument('--feature', type=str, default='dpc')   # aac, dpc, ctd, pseaac1, pseaac2, all
parser.add_argument('--classify', type=str, default='LR')  # LR, DT, RF, svm, linearsvc
parser.add_argument('--data', type=str, default='VFs')  # VFs or iris for test
parser.add_argument('--file', type=str, default='VFG-2706')  # VFG-2706 or VFG-740 or VFG-2706-1066
parser.add_argument('--signal', type=int, default=13)  # 13, 23, 33, 43, 53
args = parser.parse_args()

data_dir = os.getcwd() + "/data/"
features_dir = data_dir + args.file + "_features/"
class_name_dir = os.getcwd() + "/data/" + args.file + "_class_name"
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
data_minmax = min_max_scaler.fit_transform(all_data)

save_dir = record_dir + args.feature + "_" + args.classify + "/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
random_state = np.random.RandomState(0)
"""
class_weights_list = compute_class_weight('balanced', np.unique(all_labels), all_labels)   # list
# convet class_weights from list to dictionary
class_weights = {}
for i, eachclass in enumerate(class_weights_list):
    class_weights[i] = class_weights_list[i]
"""

X_train, X_test_a, y_train, Y_test_a = train_test_split(data_minmax, all_labels, test_size=0.4,
                                                        stratify=all_labels,
                                                        random_state=args.signal)
X_test, X_val, y_test, y_val = train_test_split(X_test_a, Y_test_a, test_size=0.5, stratify=Y_test_a,
                                                random_state=args.signal)
print('train number is {}\ntest number is {}\n val number is {}\n'.format(len(X_train), len(X_test), len(X_val)))

if args.grid == '0':
    f_time = time.time()
    model = None
    if args.classify == 'svc':
        # tuned_parameters = [{'kernel': ['rbf'], 'C': [0.1, 1, 10, 100, 1000], 'gamma':[1e-4, 1e-3, 1e-2, 1e-1]}]
        tuned_parameters = {'C': [0.1, 1, 10, 100, 1000]}
        # scores = ['precision', 'recall', 'f1']
        model = GridSearchCV(estimator=SVC(gamma='auto', decision_function_shape='ovr'), param_grid=tuned_parameters, cv=5, n_jobs=2)
    elif args.classify == 'linearsvc':
        tuned_parameters = {'C': [0.1, 1, 10, 100, 1000]}
        model = GridSearchCV(estimator=LinearSVC(random_state=random_state), param_grid=tuned_parameters, cv=5, n_jobs=2)
    elif args.classify == 'RF':
        tuned_parameters = {'n_estimators': [10, 50, 100, 200, 300]}
        model = GridSearchCV(estimator=RandomForestClassifier(random_state=random_state),
                             param_grid=tuned_parameters, cv=5, n_jobs=2)
    model.fit(X_train, y_train)
    if args.all == '1':
        f = open(save_dir + "all_" + args.classify + args.signal + '_grid_search.txt', 'a')
        f_save_best_model_dir = save_dir + "all_" + args.classify + args.signal + '_grid_search_bestmodel'
    else:
        f = open(save_dir + args.feature + "_" + args.classify + args.signal + '_grid_search.txt', 'a')
        f_save_best_model_dir = save_dir + args.feature + "_" + args.classify + args.signal + '_grid_search_bestmodel'

    f.write('Best parameters set found on development set:\n{}\n'.format(model.best_params_))

    means = model.cv_results_['mean_test_score']
    stds = model.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, model.cv_results_['params']):
        f.write('{:.4f} (+/-{:.4f}) for {}\n'.format(mean, std * 2, params))

    best_model = model.best_estimator_
    pickle.dump(best_model, open(f_save_best_model_dir, 'a'))  # save best model
    # load_model = pickle.load(open(f_save_best_model_dir, 'r'))
    f.write('Grid scores on development set:\n')
    train_acc = best_model.score(X_train, y_train)
    test_acc = best_model.score(X_test, y_test)
    y_pred = best_model.predict(X_test)
    test_pred_labels = y_pred.tolist()
    test_cla_report = classification_report(y_test, test_pred_labels, target_names=list(class_name))
    f.write('Best model, Training acc is {:.4f}\nTest_acc is: {:.4f}\nClassfication report:\n{}\n'.
            format(train_acc, test_acc, test_cla_report))
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
                                                                                           recall_value_2,
                                                                                           f1_score_value_2))
    s_time = time.time()
    times = format_time(s_time - f_time)
    f.write('Time is :{}\n'.format(times))
    f.close()
    print("finished\n")

elif args.grid == '1':
    f_time = time.time()
    model = None
    if args.classify == 'svc':
        model = svm.SVC(probability=True, random_state=random_state, decision_function_shape='ovr')  # rbf
        # model = OneVsRestClassifier(svm.SVC(probability=True, random_state=random_state))  # rbf
    elif args.classify == "linearsvc":
        model = LinearSVC(random_state=random_state)
    elif args.classify == 'RF':
        model = RandomForestClassifier(random_state=random_state, n_estimators=200)
    elif args.classify == 'LR':
        model = LogisticRegression(random_state=random_state)
    elif args.classify == 'DT':
        model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)

    if args.all == '1':
        f = open(save_dir + "all_" + args.classify + args.signal + '.txt', 'a')
        f_save_best_model_dir = save_dir + "all_" + args.classify + args.signal + '_bestmodel'
    else:
        f = open(save_dir + args.feature + "_" + args.classify + args.signal + '.txt', 'a')
        f_save_best_model_dir = save_dir + args.feature + "_" + args.classify + args.signal + '_bestmodel'

    pickle.dump(model, open(f_save_best_model_dir, 'ab'))  # save model

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    y_pred = model.predict(X_test)
    y_pred = model.predict_proba(X_test)
    test_pred_labels = y_pred.tolist()
    test_cla_report = classification_report(y_test, test_pred_labels, target_names=list(class_name))
    f.write('Best model, Training acc is {:.4f}\nTest_acc is: {:.4f}\nClassfication report:\n{}\n'.
            format(train_acc, test_acc, test_cla_report))
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
                                                                                           recall_value_2,
                                                                                           f1_score_value_2))
    s_time = time.time()
    times = format_time(s_time - f_time)
    f.write('Time is :{}\n'.format(times))
    f.close()
    print("finished\n")