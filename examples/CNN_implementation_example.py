"""
 - Author : Anish Basnet
 - Email : anishbasnetworld@gmail.com
 - Date : Sunday, July 19, 2021
"""
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.python.keras import Sequential, Input
from tensorflow.python.keras.layers import Flatten, Dense, Dropout, Conv1D
from tensorflow.python.keras.utils.np_utils import to_categorical

from tweeter_covid19.utils.pickleutils import read_pickle_data


def mkdir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except NotADirectoryError:
            pass


def cnn2d_model_for_17features():

    dropout_rate = 0.2
    input_layer = Input((17, 1), name="Input17d_Conv")
    x = Conv1D(filters=32, kernel_size=3, activation='relu', name='layer_17d_1')(input_layer)
    x = Conv1D(filters=16, kernel_size=3, activation='relu', name='layer_17d_2')(x)
    x = Flatten(name='flatten_17d_1')(x)
    x = Dropout(dropout_rate, name='dropout_17d_1')(x)
    x = Dense(128, activation='relu', name='layer_17d_3')(x)
    x = Dropout(dropout_rate, name='dropout_17d_2')(x)
    x = Dense(64, activation='relu', name='layer_17d_4')(x)
    x = Dense(3, activation='softmax', name='layer_17d_5')(x)
    model = keras.models.Model(inputs=input_layer, outputs=x)
    opt = keras.optimizers.RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def cnn2d_model_for_300features():
    dropout_rate = 0.2
    input_layer = Input((300, 1), name="Input300d_Conv")
    x = Conv1D(filters=32, kernel_size=3, activation='relu', name='layer_300d_1')(input_layer)
    x = Conv1D(filters=16, kernel_size=3, activation='relu', name='layer_300d_2')(x)
    x = Flatten(name='flatten_300d_1')(x)
    x = Dropout(dropout_rate, name='dropout_300d_1')(x)
    x = Dense(128, activation='relu', name='layer_300d_3')(x)
    x = Dropout(dropout_rate, name='dropout_300d_2')(x)
    x = Dense(64, activation='relu', name='layer_300d_4')(x)
    x = Dense(3, activation='softmax', name='layer_300d_5')(x)
    model = keras.models.Model(inputs=input_layer, outputs=x)
    opt = keras.optimizers.RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


def cnn2d_model_for_3features():
    dropout_rate = 0.2
    input_layer = Input((3, 1), name="Input3d_Conv")
    x = Conv1D(filters=8, kernel_size=2, activation='relu', name='layer_3d_1')(input_layer)
    x = Conv1D(filters=8, kernel_size=2, activation='relu', name='layer_3d_2')(x)
    x = Flatten(name='flatten_3d_1')(x)
    x = Dropout(dropout_rate, name='dropout_3d_1')(x)
    x = Dense(6, activation='relu', name='layer_3d_3')(x)
    x = Dropout(dropout_rate, name='dropout_3d_2')(x)
    x = Dense(4, activation='relu', name='layer_3d_4')(x)
    x = Dense(3, activation='softmax', name='layer_3d_5')(x)
    model = keras.models.Model(inputs=input_layer, outputs=x)
    opt = keras.optimizers.RMSprop(lr=0.00001, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model



N_SETS = 10

if __name__ == '__main__':

    read_main_path = os.path.join('data', 'fold_train_test_dataset_overall_vectors_3dim')
    model_write_path = os.path.join('data', 'model_save')

    data_dict = dict()
    accuracy = []

    for n_set in range(N_SETS):
        print("---------------- Loading Set {} data sets -------------------".format(n_set + 1))
        read_joiner_path = os.path.join(read_main_path, 'set_' + str(n_set + 1))
        writer_joiner_path = os.path.join(model_write_path, 'set_' + str(n_set + 1))

        mkdir(writer_joiner_path)

        train_x = read_pickle_data(os.path.join(read_joiner_path, 'train_x.pkl'))
        train_y = read_pickle_data(os.path.join(read_joiner_path, 'train_y.pkl'))
        test_x = read_pickle_data(os.path.join(read_joiner_path, 'test_x.pkl'))
        test_y = read_pickle_data(os.path.join(read_joiner_path, 'test_y.pkl'))
        print("---------------- Loading Set {} completed -------------------".format(n_set + 1))

        print(np.shape(train_x), np.shape(train_y))
        print(np.shape(test_x), np.shape(test_y))

        scale_model = StandardScaler()
        scale_model.fit(train_x)

        train_x = scale_model.transform(train_x)
        test_x = scale_model.transform(test_x)

        print(np.shape(train_x), np.shape(train_y))
        print(np.shape(test_x), np.shape(test_y))

        le = preprocessing.LabelEncoder()
        le.fit(train_y)

        train_y = le.transform(train_y)
        test_y = le.transform(test_y)

        train_y = to_categorical(train_y)
        test_y = to_categorical(test_y)

        train_x = np.array(train_x).reshape((np.shape(train_x)[0], np.shape(train_x)[1], 1))
        test_x = np.array(test_x).reshape((np.shape(test_x)[0], np.shape(test_x)[1], 1))

        print(np.shape(train_x), np.shape(train_y))
        print(np.shape(test_x), np.shape(test_y))
        model = cnn2d_model_for_3features()
        print(model.summary())
        exit(0)

        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.RMSprop(),
                      metrics=['acc'])

        print('---------batch size : {}----------------'.format(64))
        history = model.fit(train_x, train_y, validation_data=(test_x, test_y),
                            epochs=1, batch_size=64)

        print("Evaluate on test data")
        results = model.evaluate(test_x, test_y, batch_size=64)
        model.save(os.path.join(writer_joiner_path, str(n_set + 1) + '_17dim.h5'))
        print("test loss, test acc:", results)
        data_dict[str(n_set + 1)] = [results[1] * 100]
        accuracy.append(results[1] * 100)
    data_dict['avg'] = np.mean(accuracy)
    print("Average values is : ", np.mean(accuracy))
    df = pd.DataFrame(data_dict)
    df.to_csv('result.csv')
    #     predict = model.predict(test_x)
    #     predict_ = np.zeros_like(predict)
    #     predict_[np.arange(len(predict)), predict.argmax(1)] = 1
    #
    #     print(classification_report(test_y, predict_))
    #
    #
    #     from matplotlib import pyplot
    #
    #     # learning curves of model accuracy
    #     pyplot.plot(history.history['acc'], label='train')
    #     pyplot.plot(history.history['val_acc'], label='test')
    #     pyplot.legend()
    #     pyplot.show()
    #     exit(0)
