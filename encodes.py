import numpy as np
from keras.preprocessing.sequence import pad_sequences


def embed_encode(datas):
    x = []
    maxlen = 2500
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa2int = dict((c, i+1) for i, c in enumerate(amino_acids))
    for d in datas:
        x.append([aa2int[aa] for aa in d])
    x = pad_sequences(x, maxlen=maxlen)
    return x


def onehot_encode(datas):
    x = map(seq2onehot, datas)
    return x


def seq2onehot(x):
    """return one sequence"""
    maxlen = 2500
    dim = 20
    aa2num = []
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    aa2int = dict((ami_acid, id_x+1) for id_x, ami_acid in enumerate(amino_acids))
    for aa in x:
        aa2num.append(aa2int[aa])

    # methods2 more fast
    zero_matrix = np.eye(dim, dtype=np.int8)[np.array(aa2num) - 1]
    zero_matrix = np.pad(zero_matrix, ((0, maxlen - len(aa2num)), (0, 0)), 'constant')
    return zero_matrix


def onehot_aac(datas, aac):
    maxlen = 2500
    dim = 20
    new_data = []
    for i, each_seq in enumerate(datas):  # aac[i]
        aa2num = []
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        aa2int = dict((ami_acid, id_x + 1) for id_x, ami_acid in enumerate(amino_acids))
        for aa in each_seq:
            aa2num.append(aa2int[aa])

        aa2num = aa2num + [0] * (maxlen - len(aa2num))
        zero_matrix = np.zeros(shape=(maxlen, dim), dtype=np.float32)
        for aa, numb in enumerate(aa2num):
            if numb != 0:
                zero_matrix[aa][numb-1] = aac[i][aa]
        new_data.append(zero_matrix)

    return new_data


def onehot_aac_one_seq(datas, aac):
    maxlen = 2500
    dim = 20
    aa2num = []
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    aa2int = dict((ami_acid, id_x + 1) for id_x, ami_acid in enumerate(amino_acids))
    for aa in datas:
        aa2num.append(aa2int[aa])

    aa2num = aa2num + [0] * (maxlen - len(aa2num))
    zero_matrix = np.zeros(shape=(maxlen, dim))
    for aa, numb in enumerate(aa2num):
        if numb != 0:
            zero_matrix[aa][numb - 1] = aac[aa]

    return zero_matrix

