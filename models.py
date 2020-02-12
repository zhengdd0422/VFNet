from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Embedding, Activation, Input, concatenate, dot
from keras.layers import Conv1D, MaxPooling1D, LSTM, GRU
from keras.models import Sequential
from keras import regularizers
from keras.constraints import max_norm
from keras import initializers
from keras.layers.wrappers import Bidirectional
from keras.layers.normalization import BatchNormalization


def vfnet(maxlength, input_dim, class_num):
    # kernel_initia default is glorot_uniform(Xavier)
    # kernel_initia = "random_uniform"
    kernel_initia = initializers.random_normal(mean=0.0, stddev=0.05, seed=None)  # default
    # kernel_initia = initializers.random_uniform(seed=None)
    # kernel_initia = initializers.random_normal(mean=0.0, stddev=0.01, seed=None)
    # kernel_initia = initializers.truncated_normal(mean=0.0, stddev=0.05, seed=None)
    # kernel_initia = initializers.lecun_normal(seed=None)
    # kernel_initia = initializers.lecun_uniform(seed=None)
    # kernel_initia = initializers.he_normal(seed=None)
    # kernel_initia = initializers.he_uniform(seed=None)
    drop = 0.5  # 0.5
    n_filter = 128
    kernel = 7
    pool_size = 5
    model = Sequential()
    model.add(Conv1D(filters=n_filter, kernel_size=kernel, input_shape=(maxlength, input_dim,), name='conv1',
                     kernel_initializer=kernel_initia))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=pool_size, name='maxpool2'))
    model.add(Conv1D(filters=n_filter, kernel_size=kernel, name='conv3', kernel_initializer=kernel_initia))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=pool_size, name='maxpool4'))
    model.add(Dropout(drop, name='dropout1'))
    model.add(Flatten(name='flatten5'))
    model.add(Dense(512, name='fl1', ))
    model.add(Dropout(drop, name='dropout2'))
    # model.add(Dense(512, name='fl2', ))
    # model.add(Dropout(drop, name='dropout3'))
    model.add(Dense(class_num, activation='softmax', name='prediction'))
    print(model.summary())
    return model


def vfneth(maxlength, input_dim, maxlength_2, class_num):
    kernel_initia = initializers.random_normal(mean=0.0, stddev=0.05, seed=None)

    # input1
    input1 = Input(shape=(maxlength, input_dim,), name='input1')
    x1 = Conv1D(filters=128, kernel_size=7, name='conv1',
                kernel_initializer=kernel_initia)(input1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling1D(pool_size=5, name='maxpool1')(x1)
    x1 = Conv1D(filters=128, kernel_size=7, name='conv2',
                kernel_initializer=kernel_initia)(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling1D(pool_size=5, name='maxpool2')(x1)
    x1 = Dropout(0.5, name='dropout1')(x1)
    x1 = Flatten()(x1)
    x1 = Dense(512, name='fl1')(x1)
    x1 = Dropout(0.5, name='dropout2')(x1)

    # input2
    input2 = Input(shape=(maxlength_2,), name='input2')
    x = concatenate([x1, input2], axis=1)
    # x = dot([x1, input2], normalize=True)
    # x = BatchNormalization()(x)
    # x = Flatten()(x)
    x = Dense(512, name='fl')(x)
    x = Dropout(0.5, name='dropout3')(x)
    x = Dense(class_num, activation='softmax', name='prediction')(x)
    model = Model(inputs=[input1, input2], outputs=x)
    print(model.summary())
    return model


def embed_amps(embed_vector, max_length, n_class=1):
    model = Sequential()
    model.add(Embedding(20, embed_vector, input_length=max_length, name='ly1'))
    model.add(Conv1D(filters=64, kernel_size=16, padding="same", activation='relu', name='ly2'))
    model.add(MaxPooling1D(pool_size=5, name='ly3'))
    model.add(LSTM(100, use_bias=True, dropout=0.1, return_sequences=False, name='ly4'))  # merge_mode='ave'
    model.add(Dense(n_class, activation='softmax', name="prediction"))  # dense is full connection
    print(model.summary())
    return model


def word2vec_neubi(embed_input, embed_out, max_length, matrix=False, n_class=1):
    model = Sequential()
    e = Embedding(embed_input, embed_out, weights=[matrix], input_length=max_length, trainable=False)
    model.add(e)
    model.add(Bidirectional(GRU(32, dropout=0.3, recurrent_dropout=0.1, return_sequences=True)))
    model.add(Bidirectional(GRU(64, dropout=0.3, recurrent_dropout=0.1)))
    model.add(Dense(n_class, activation='softmax', name="prediction"))
    print(model.summary())
    return model
