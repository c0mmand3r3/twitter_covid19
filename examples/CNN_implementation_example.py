"""
 - Author : Anish Basnet
 - Email : anishbasnetworld@gmail.com
 - Date : Sunday, July 19, 2021
"""
import os
import pandas as pd
import numpy as np
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.python.keras.utils.np_utils import to_categorical

from tweeter_covid19.utils.pickleutils import read_pickle_data

from sklearn import preprocessing

N_SETS = 10


def cnn_model():
    sequence_input = Input(shape=(300, 1))
    x = Conv1D(128, 5, activation='relu')(sequence_input)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)  # global max pooling
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(3, activation='softmax')(x)
    return Model(sequence_input, preds)


def cnn_model_for_17dim():
    sequence_input = Input(shape=(17, 1))
    x = Conv1D(8, 2, activation='relu')(sequence_input)
    x = MaxPooling1D(2)(x)
    x = Conv1D(8, 2, activation='relu')(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(8, 2, activation='relu')(x)
    x = MaxPooling1D(2)(x)  # global max pooling
    x = Flatten()(x)
    x = Dense(8, activation='relu')(x)
    preds = Dense(3, activation='softmax')(x)
    return Model(sequence_input, preds)


if __name__ == '__main__':
    model = cnn_model()
    print(model.summary())
    model_2 = cnn_model_for_17dim()
    print(model_2.summary())
    # read_main_path = os.path.join('data', 'fold_train_test_collector')
    #
    # for n_set in range(N_SETS):
    #     print("---------------- Loading Set {} data sets -------------------".format(n_set + 1))
    #     read_joiner_path = os.path.join(read_main_path, 'set_' + str(n_set + 1))
    #
    #     train_x = read_pickle_data(os.path.join(read_joiner_path, 'train_x.pkl'))
    #     train_y = read_pickle_data(os.path.join(read_joiner_path, 'train_y.pkl'))
    #     test_x = read_pickle_data(os.path.join(read_joiner_path, 'test_x.pkl'))
    #     test_y = read_pickle_data(os.path.join(read_joiner_path, 'test_y.pkl'))
    #     print("---------------- Loading Set {} completed -------------------".format(n_set + 1))
    #
    #     print(np.shape(train_x), np.shape(train_y))
    #     print(np.shape(test_x), np.shape(test_y))
    #
    #     le = preprocessing.LabelEncoder()
    #     le.fit(train_y)
    #
    #
    #
    #     train_y = le.transform(train_y)
    #     test_y = le.transform(test_y)
    #
    #     train_y = to_categorical(train_y)
    #     test_y = to_categorical(test_y)
    #
    #     train_x = np.array(train_x).reshape((np.shape(train_x)[0], np.shape(train_x)[1], np.shape(train_x)[2], 1))
    #     test_x = np.array(test_x).reshape((np.shape(test_x)[0], np.shape(test_x)[1], np.shape(test_x)[2], 1))
    #
    #     print(np.shape(train_x), np.shape(train_y))
    #     print(np.shape(test_x), np.shape(test_y))

    #
    # model.compile(loss='categorical_crossentropy',
    #               optimizer='rmsprop',
    #               metrics=['acc'])
    #
    # # happy learning!
    # model.fit(train_x, train_y, validation_data=(test_x, test_y),
    #           epochs=2, batch_size=128)
