import numpy as np
import collections
import json
import math
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import collections


def plot_acc_loss(train_info_record, sig, train_values, val_values, epochs, ll=False):
    plt.clf()
    plt.plot(epochs, train_values, 'bo', label='Training')
    plt.plot(epochs, val_values, 'r', label='Validation')
    if ll:
        name = train_info_record + sig + '_' + str(ll)
        title = sig + '_' + 'training and validation ' + str(ll)
    else:
        name = train_info_record + sig
        title = sig + '_' + 'training and validation'
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(str(ll))
    plt.legend()
    plt.savefig(str(name) + '.png')
    plt.clf()


def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.

    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)
    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def change_lr(lr, epoch):
    changed_lr = lr
    # global change_lr
    if epoch < 100:
        changed_lr = 0.1
    elif 100 <= epoch < 200:
        changed_lr = 0.01
    elif epoch >= 200:
        changed_lr = 0.001
    return changed_lr
"""
def change_lr_re(lr, epoch):
    change_lr = lr
    #global change_lr
    if epoch < 100 or (epoch < 150 and epoch>=100) or  epoch >=150:
        change_lr /=10
"""


def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.6
    epochs_drop = 3.0
    lrate = initial_lrate * math.pow(drop,
                                     math.floor((1 + epoch) / epochs_drop))
    return lrate

"""
lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]
"""


def data_shuffle_fixed(x, y):
    # select 1
    #    x=np.array(x)
    #    y=np.array(y)
    #    r = np.random.permutation(len(y))
    #    x = x[r]
    #    y = y[r]
    # select 2
    np.random.seed(2019)
    np.random.shuffle(x)
    np.random.seed(2019)
    np.random.shuffle(y)
    return x, y


def data_shuffle_nonfixed(x, y):
    x_num, _ = x.shape
    index = np.arange(x_num)
    np.random.shuffle(index)
    return x[index], y[index]


def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy) + 1])
    yy1[np.arange(len(yy)), yy] = 1
    return yy1


"""
Y_train = to_onehot(map(lambda x: mods.index(lbl[x][0]), train_idx))
Y_test = to_onehot(map(lambda x: mods.index(lbl[x][0]), test_idx))
"""


"""
layer_outputs = [layer.output for layer in model.layers]
activation_model = Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(X_train[10].reshape(1, 28, 28, 1))

"""


def display_activation(activations, col_size, row_size, act_index):
    activation = activations[act_index]
    activation_index = 0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size * 2.5, col_size * 1.5))
    for row in range(0, row_size):
        for col in range(0, col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
            activation_index += 1

"""
Desplaying above image after layer 2 .
layer 1 is input layer .
display_activation(activations, 8, 8, 1)
Displaying output of layer 4
display_activation(activations, 8, 8, 3)
Displaying output of layer 8
display_activation(activations, 8, 8, 7)
"""

"""
# predict results
results = model.predict(test)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

submissions=pd.DataFrame({"ImageId": list(range(1,len(results)+1)),
                         "Label": results})
submissions.to_csv("re2-submission.csv", index=False, header=True)
"""


def balance_minibatch_generator_from_data(data, label, minibatch_length, class_number, features=False):

    """balance minibatch datas from file npz"""
    if features:
        data = data[:, :, 0]
    data_sorted, label_sorted = sort_two_array(data, label)  # sort to build minibatch datasets
    label_counter = counter2dic(label_sorted)
    # num_each_sample = int(round(minibatch_length/class_number))   # how many samples each label to be extracted
    num_each_sample = minibatch_length // class_number

    while True:
        minibatch_data = []
        minibatch_label = []
        total_sofar = 0
        for ids in label_counter.keys():
            number_new = total_sofar + label_counter[ids]
            ids_data = data_sorted[total_sofar:number_new]
            for j in range(num_each_sample):  # extract num_each_sample sequences to become
                indices = np.random.permutation(label_counter[ids])
                minibatch_data.append(ids_data[indices[0]])
                minibatch_label.append(ids)
                # minibatch_label = minibatch_label + [ids.astype(np.int8)]
                # minibatch_label.append(ids.astype(np.int8).tolist())
            total_sofar = number_new
        minibatch_label = to_categorical(minibatch_label, num_classes=class_number)

        if features:
            minibatch_data = minibatch_data[:, :, np.newaxis]
        yield np.array(minibatch_data), np.array(minibatch_label)


def balance_minibatch_from_dict(datas, minibatch_length):
    """balance minibatch datas from file dict, onehot_label_split"""

    label_number = len(datas.keys())
    num_samples = minibatch_length/label_number
    minibatch_data = []
    minibatch_label = []
    for ids in datas.keys():
        for j in range(num_samples):
            indices = np.random.permutation(len(datas[ids]))
            minibatch_data.append(datas[ids][indices[0]])
            minibatch_label = minibatch_label + [ids]

    yield minibatch_data, minibatch_label


# up sample
"""from collections import Counter
Counter(labels)

labels = to_categorical(labels, num_classes=len(class_name))
data, labels = shuffle(data, labels,  random_state=22)
train_x_a, X_test, train_y_a, Y_test = train_test_split(data, labels, test_size=0.1, random_state=23, stratify=labels)
X_train, X_val, Y_train, Y_val = train_test_split(train_x_a, train_y_a, test_size=0.1, random_state=43, stratify=train_y_a)
del data, labels, train_x_a, train_y_a

from sklearn.datasets import make_classification
X, y = make_classification(n_samples=3000, n_features=2, n_informative=2,
                           n_redundant=0, n_repeated=0, n_classes=3,
                           n_clusters_per_class=1,
                           weights=[0.1, 0.05, 0.85],
                           class_sep=0.8, random_state=2018)
Counter(y)
ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_sample(X, y)
sorted(Counter(y_resampled).items())

from imblearn.under_sampling import RandomUnderSampler
# under_sample
rus = RandomUnderSampler(random_state=0)
X_resampled, y_resampled = rus.fit_sample(X, y)

sorted(Counter(y_resampled).items())
"""


def sort_two_array(data, label):
    """sort label with data at the same order"""
    sorted_indices = np.argsort(label)
    sorted_data = data[sorted_indices]
    sorted_label = label[sorted_indices]
    return sorted_data, sorted_label


def counter2dic(datas):
    """count numbers of each labels, return a dic"""
    c_l = {}
    counter_labels = collections.Counter(datas)
    for key in counter_labels:
        c_l[key] = counter_labels[key]
    return c_l


def counter2ndarry(datas):
    """count numbers of each labels, return a ndarry"""
    c_l = np.bincount(datas)
    return c_l


def count_eachlabel_class_weight(y_train):
    class_weight = 'balanced'
    class_number = np.unique(y_train)
    weight = compute_class_weight(class_weight, class_number, y_train)
    label_class_weight = {}
    for label_ids in class_number:
        label_class_weight[label_ids] = weight[label_ids]
    return label_class_weight


def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a


def save_train_loss_log(history, info_record, sig):
    f = open(info_record + sig + '.txt', 'a')
    loss_values = history.history['loss']  # one value per epoch
    val_loss_values = history.history['val_loss']
    acc_values = history.history['acc']
    val_acc_values = history.history['val_acc']  # one value per epoch
    epochs = range(1, len(history.history['acc']) + 1)
    plot_acc_loss(info_record, sig, acc_values, val_acc_values, epochs, ll='acc')  # plot acc
    plot_acc_loss(info_record, sig, loss_values, val_loss_values, epochs, ll='loss')  # plot loss
    for i in range(0, len(history.history['acc'])):
        f.write('epoch:{}, train_loss:{:.4f}, train_acc:{:.2f}, val_loss:{:.4f}, val_acc:{:.2f}'.format(i + 1,
            loss_values[i], acc_values[i], val_loss_values[i], val_acc_values[i]) + '\n')
    f.close()


def cosine_rampdown(epoch_current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983
    rampdown_length: cutoff of lr==0
    SGDR: Stochastic Gradient Descent with Warm Restarts
    """
    assert 0 <= epoch_current <= rampdown_length
    return float(.5 * (np.cos(np.pi * epoch_current / rampdown_length) + 1))


def min_max_mormalization_1(x):
    """[0 1]"""
    return [(float(i) - min(x)) / float(max(x) - min(x)) for i in x]


def min_max_mormalization_2(x):
    """[-1 1]"""
    return [(float(i) - np.mean(x)) / float(max(x) - min(x)) for i in x]


def zz_score_normalization(x):
    """mean=0,std=1"""
    return (x - np.mean(x)) / np.std(x, ddof=1)


def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu)/sigma


def save_dict_as_json2(data_dict):
    save_dict = {
        'version': "1.0",
        'results': data_dict,
        'explain': {
            'used': True,
            'details': "this is for dict2json_save",
        }
    }
    json.dumps(save_dict, indent=4)


def load_class_name(filename):
    all_name_list = []
    with open(filename, 'r') as f_class_name:
        for each_name in f_class_name.readlines():
            all_name_list.append(each_name.strip())
    return all_name_list


def reduce_dimension(each_list):
    each_list = each_list.reshape(len(each_list[0]))
    return each_list


def convert_float(each_list):   # for list
    each_list_float = list(map(float, each_list))
    return each_list_float


def create_class_weight1(labels_dict, mu=0.15):
    """
    # random labels_dict
    example: labels_dict = {0: 2813, 1: 78, 2: 2814, 3: 78, 4: 7914, 5: 248, 6: 7914, 7: 248}
    create_class_weight(labels_dict)
    """

    total = np.sum(labels_dict.values())
    keys = labels_dict.keys()
    class_weight = dict()

    for key in keys:
        score = math.log(mu*total/float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0

    return class_weight


def create_class_weight(y):
    counter = collections.Counter(y)
    majority = max(counter.values())
    return {cls: float(majority/count) for cls, count in counter.items()}