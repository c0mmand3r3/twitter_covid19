"""
 - Author : Anish Basnet
 - Email : anishbasnetworld@gmail.com
 - Date : Sunday, July 19, 2021
"""
import os

import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout
from tensorflow.python.keras.utils.np_utils import to_categorical

from tweeter_covid19.utils import standard_normalize, mkdir
from tweeter_covid19.utils.pickleutils import read_pickle_data, write_pickle_data


def cnn2d_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(12, 300, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    return model


def cnn2d_model_for_17features():
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(12, 17, 1)))
    model.add(Conv2D(8, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    return model


N_SETS = 10

if __name__ == '__main__':
    writer_main_path = os.path.join('data', 'fold_train_test_collector_with_normalization')

    read_main_path = os.path.join('data', 'fold_train_test_collector')

    for n_set in range(N_SETS):
        print("---------------- Loading Set {} data sets -------------------".format(n_set + 1))
        read_joiner_path = os.path.join(read_main_path, 'set_' + str(n_set + 1))

        train_x = read_pickle_data(os.path.join(read_joiner_path, 'train_x.pkl'))
        train_y = read_pickle_data(os.path.join(read_joiner_path, 'train_y.pkl'))
        test_x = read_pickle_data(os.path.join(read_joiner_path, 'test_x.pkl'))
        test_y = read_pickle_data(os.path.join(read_joiner_path, 'test_y.pkl'))
        print("---------------- Loading Set {} completed -------------------".format(n_set + 1))

        writer_joiner_path = os.path.join('data', 'fold_train_test_collector_with_normalization', 'set_' + str(n_set + 1))
        mkdir(writer_joiner_path)

        print(np.shape(train_x), np.shape(train_y))

        train_flatten = []

        for index in range(np.shape(train_x)[0]):
            for row_vector in range(np.shape(train_x)[1]):
                train_flatten.append(train_x[index][row_vector])
            if index % 1000 == 0:
                print("{} - Succesfully executed".format(index))

        scale_model = StandardScaler()
        scale_model.fit(train_flatten)

        train_vectors = []
        for vector in train_x:
            train_vectors.append(scale_model.transform(vector))


        test_vectors = []
        for vector in test_x:

            test_vectors.append(scale_model.transform(vector))

        write_pickle_data(os.path.join(writer_joiner_path, 'train_x.pkl'), train_vectors)
        write_pickle_data(os.path.join(writer_joiner_path, 'train_y.pkl'), train_y)
        write_pickle_data(os.path.join(writer_joiner_path, 'test_x.pkl'), test_vectors)
        write_pickle_data(os.path.join(writer_joiner_path, 'test_y.pkl'), test_y)
        print("Sucessfully written for SET -- {} fold. ".format(n_set+1))