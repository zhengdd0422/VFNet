import numpy as np
import keras
import encodes
from keras.models import load_model
from keras.models import Model
from sklearn import preprocessing


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, datas, labels, batch_size=128, n_classes=10, encode='onehot', shuffle=True):
        """Initialization"""
        self.batch_size = batch_size
        self.datas = datas
        self.labels = labels
        self.encode = encode
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.datas))
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(len(self.datas) / float(self.batch_size)))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        batch_indexs = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_datas = [self.datas[k] for k in batch_indexs]
        batch_label = [self.labels[k] for k in batch_indexs]
        # Generate data
        x, y = self.data_generation(batch_datas, batch_label)
        return x, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def data_generation(self, batch_datas, batch_label):
        """Generates data containing batch_size samples"""
        x = ''
        if self.encode == 'onehot':
            x = encodes.onehot_encode(batch_datas)
        elif self.encode == 'pair2onehot':
            x = encodes.pair_onehot_encode(batch_datas)
        elif self.encode == 'embed':
            x = encodes.embed_encode(batch_datas)
        elif self.encode == 'word2vec':
            x = batch_datas
        # x = np.array(x).astype('float32')
        y = keras.utils.to_categorical(batch_label, num_classes=self.n_classes)
        # x = np.array(x).astype('float32')
        # y = np.array(y).astype('float32')
        return np.array(x), np.array(y)


class DataGenerator2inputs(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, datas, labels, batch_size=128, n_classes=10,  encode='onehot', feature_data=None, feature_label=None, shuffle=True):
        """Initialization"""
        self.batch_size = batch_size
        self.datas = datas
        self.labels = labels
        self.feature_data = feature_data
        self.feature_label = feature_label
        self.encode = encode
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.datas))
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(len(self.datas) / float(self.batch_size)))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        batch_indexs = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_datas = [self.datas[k] for k in batch_indexs]
        batch_label = [self.labels[k] for k in batch_indexs]
        x, y = self.data_generation(batch_datas, batch_label, feature=False)

        batch_datas_feature = [self.feature_data[k] for k in batch_indexs]
        batch_label_feature = [self.feature_label[k] for k in batch_indexs]
        x_feature, y_feature = self.data_generation(batch_datas_feature, batch_label_feature, feature=True)
        return [x, x_feature], y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def data_generation(self, batch_datas, batch_label, feature=False):
        """Generates data containing batch_size samples"""
        x = ''
        if feature:
            x = batch_datas
        else:
            if self.encode == 'onehot':
                x = encodes.onehot_encode(batch_datas)
            elif self.encode == 'embed':
                x = encodes.embed_encode(batch_datas)
            elif self.encode == 'word2vec':
                x = batch_datas
        y = keras.utils.to_categorical(batch_label, num_classes=self.n_classes)
        return np.array(x), np.array(y)


class DataGenerator2features4merge(keras.utils.Sequence):
    """Generates data for merge cnn_features and features, and then use mpl"""

    def __init__(self, datas, labels, batch_size=128, n_classes=10,  encode='onehot',  data_signal='1', feature_data=None, shuffle=True):
        """Initialization"""
        self.batch_size = batch_size
        self.datas = datas
        self.labels = labels
        self.feature_data = feature_data
        self.encode = encode
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.data_signal = data_signal
        self.indexes = np.arange(len(self.datas))
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(len(self.datas) / float(self.batch_size)))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        batch_indexs = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_datas = [self.datas[k] for k in batch_indexs]
        batch_datas_feature = [self.feature_data[k] for k in batch_indexs]
        batch_label = [self.labels[k] for k in batch_indexs]
        x, y = self.data_generation(batch_datas, batch_datas_feature, batch_label)
        # x_feature = x_feature.reshape((x_feature.shape[0], x_feature.shape[1], 1))
        if self.data_signal == "1":
            return x, y
        else:
            return x

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def data_generation(self, batch_datas, batch_datas_feature, batch_label):
        layer_name = 'flatten5'
        trained_cnn_model = load_model("/home/zhengdd/data_2t/Pycharm/data/VFs/VFDB_new2019/record/"
                                       "onehotcnn_ly2_bn_adam_seed21_class_weight/"
                                       "onehotcnn_ly2_bn_adam_seed21_class_weight_bestmodel")
        intermediate_layer_model = Model(inputs=trained_cnn_model.input,
                                         outputs=trained_cnn_model.get_layer(layer_name).output)
        cnn_feature = intermediate_layer_model.predict(batch_datas)
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        min_max_scaler.fit(cnn_feature)
        cnn_feature_minmax = min_max_scaler.transform(cnn_feature)
        feature_minmax = min_max_scaler.transform(batch_datas_feature)

        x = np.concatenate((cnn_feature_minmax, feature_minmax), axis=1)
        y = keras.utils.to_categorical(batch_label, num_classes=self.n_classes)
        return np.array(x), np.array(y)
