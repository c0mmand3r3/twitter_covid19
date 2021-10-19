"""
 - Author : Anish Basnet
 - Email : anishbasnetworld@gmail.com
 - Date : Monday, August 9, 2021
"""
import os

import tensorflow as tf
from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.layers import  maximum
from tensorflow.python.keras.saving.save import load_model

from tweeter_covid19.utils import mkdir


def get_data_split(data):
    x = []
    y = []
    for key, value in data.items():
        target = list(value.keys())[0]
        x.append(value[target])
        y.append(target)
    return x, y


def read_pickle_data(path=None):
    import pickle
    with open(path, 'rb') as fid:
        data = pickle.load(fid)
        fid.close()
        return data


from tensorflow.python.keras import Model, Sequential
import pandas as pd

from sklearn.preprocessing import StandardScaler
import numpy as np

N_SETS = 10

if __name__ == '__main__':
    model_path = os.path.join('data', 'model_save')

    read_main_path_3D = os.path.join('data', 'fold_train_test_dataset_overall_vectors_3dim')
    read_main_path_17D = os.path.join('data', 'fold_train_test_dataset_overall_vectors')
    read_main_path_300D = os.path.join('data', 'fold_train_test_dataset_overall_vectors_for_300dim')

    model_write_path = os.path.join('data', 'model_save_ensemble')

    data_dict = dict()
    accuracy = []

    for n_set in range(N_SETS):
        print("---------------- Loading Set {} data sets -------------------".format(n_set + 1))
        model_path_joiner = os.path.join(model_path, 'set_' + str(n_set + 1))

        read_joiner_path_3D = os.path.join(read_main_path_3D, 'set_' + str(n_set + 1))
        read_joiner_path_17D = os.path.join(read_main_path_17D, 'set_' + str(n_set + 1))
        read_joiner_path_300D = os.path.join(read_main_path_300D, 'set_' + str(n_set + 1))

        writer_joiner_path = os.path.join(model_write_path, 'set_'+str(n_set+1))

        mkdir(writer_joiner_path)

        train_x_3D = read_pickle_data(os.path.join(read_joiner_path_3D, 'train_x.pkl'))
        train_y_3D = read_pickle_data(os.path.join(read_joiner_path_3D, 'train_y.pkl'))
        test_x_3D = read_pickle_data(os.path.join(read_joiner_path_3D, 'test_x.pkl'))
        test_y_3D = read_pickle_data(os.path.join(read_joiner_path_3D, 'test_y.pkl'))


        train_x_17D = read_pickle_data(os.path.join(read_joiner_path_17D, 'train_x.pkl'))
        train_y_17D = read_pickle_data(os.path.join(read_joiner_path_17D, 'train_y.pkl'))
        test_x_17D = read_pickle_data(os.path.join(read_joiner_path_17D, 'test_x.pkl'))
        test_y_17D = read_pickle_data(os.path.join(read_joiner_path_17D, 'test_y.pkl'))

        train_x_300D = read_pickle_data(os.path.join(read_joiner_path_300D, 'train_x.pkl'))
        train_y_300D = read_pickle_data(os.path.join(read_joiner_path_300D, 'train_y.pkl'))
        test_x_300D = read_pickle_data(os.path.join(read_joiner_path_300D, 'test_x.pkl'))
        test_y_300D = read_pickle_data(os.path.join(read_joiner_path_300D, 'test_y.pkl'))

        print("---------------- Loading Set {} completed -------------------".format(n_set + 1))

        print(np.shape(train_x_3D), np.shape(train_y_3D))
        print(np.shape(test_x_3D), np.shape(test_y_3D))
        print(np.shape(train_x_17D), np.shape(train_y_17D))
        print(np.shape(test_x_17D), np.shape(test_y_17D))
        print(np.shape(train_x_300D), np.shape(train_y_300D))
        print(np.shape(test_x_300D), np.shape(test_y_300D))

        scale_model = StandardScaler()
        scale_model.fit(train_x_3D)

        train_x_3D = scale_model.transform(train_x_3D)
        test_x_3D = scale_model.transform(test_x_3D)

        print(np.shape(train_x_3D), np.shape(train_y_3D))
        print(np.shape(test_x_3D), np.shape(test_y_3D))

        scale_model_17D = StandardScaler()
        scale_model_17D.fit(train_x_17D)

        train_x_17D = scale_model_17D.transform(train_x_17D)
        test_x_17D = scale_model_17D.transform(test_x_17D)

        print(np.shape(train_x_17D), np.shape(train_y_17D))
        print(np.shape(test_x_17D), np.shape(test_y_17D))

        scale_model_300D = StandardScaler()
        scale_model_300D.fit(train_x_300D)

        train_x_300D = scale_model_300D.transform(train_x_300D)
        test_x_300D = scale_model_300D.transform(test_x_300D)

        print(np.shape(train_x_300D), np.shape(train_y_300D))
        print(np.shape(test_x_300D), np.shape(test_y_300D))

        le = preprocessing.LabelEncoder()
        le.fit(train_y_3D)

        train_y_3D = le.transform(train_y_3D)
        test_y_3D = le.transform(test_y_3D)

        train_y_3D = to_categorical(train_y_3D)
        test_y_3D = to_categorical(test_y_3D)

        train_x_3D = np.array(train_x_3D).reshape((np.shape(train_x_3D)[0], np.shape(train_x_3D)[1], 1))
        test_x_3D = np.array(test_x_3D).reshape((np.shape(test_x_3D)[0], np.shape(test_x_3D)[1], 1))

        print(np.shape(train_x_3D), np.shape(train_y_3D))
        print(np.shape(test_x_3D), np.shape(test_y_3D))


        le_17 = preprocessing.LabelEncoder()
        le_17.fit(train_y_17D)

        train_y_17D = le_17.transform(train_y_17D)
        test_y_17D = le_17.transform(test_y_17D)

        train_y_17D = to_categorical(train_y_17D)
        test_y_17D = to_categorical(test_y_17D)

        train_x_17D = np.array(train_x_17D).reshape((np.shape(train_x_17D)[0], np.shape(train_x_17D)[1], 1))
        test_x_17D = np.array(test_x_17D).reshape((np.shape(test_x_17D)[0], np.shape(test_x_17D)[1], 1))

        print(np.shape(train_x_17D), np.shape(train_y_17D))
        print(np.shape(test_x_17D), np.shape(test_y_17D))

        le_300D = preprocessing.LabelEncoder()
        le_300D.fit(train_y_300D)

        train_y_300D = le_300D.transform(train_y_300D)
        test_y_300D = le_300D.transform(test_y_300D)

        train_y_300D = to_categorical(train_y_300D)
        test_y_300D = to_categorical(test_y_300D)

        train_x_300D = np.array(train_x_300D).reshape((np.shape(train_x_300D)[0], np.shape(train_x_300D)[1], 1))
        test_x_300D = np.array(test_x_300D).reshape((np.shape(test_x_300D)[0], np.shape(test_x_300D)[1], 1))

        print(np.shape(train_x_300D), np.shape(train_y_300D))
        print(np.shape(test_x_300D), np.shape(test_y_300D))

        # model_17D = Sequential()
        # model_17D.add(Conv1D(filters=32, kernel_size=3, input_shape=(17, 1), activation='relu'))
        # model_17D.add(Conv1D(filters=16, kernel_size=3, activation='relu'))
        # model_17D.add(Flatten())
        # model_17D.add(Dropout(0.2))
        # model_17D.add(Dense(128, input_dim=(17, 1), activation='relu'))
        # model_17D.add(Dropout(0.2))
        # model_17D.add(Dense(64, activation='relu'))
        # model_17D.add(Dense(3, activation='softmax'))
        #
        # dropout_rate = 0.2
        # model_3D = Sequential()
        # model_3D.add(Conv1D(filters=8, kernel_size=2, input_shape=(3, 1), activation='relu'))
        # model_3D.add(Conv1D(filters=8, kernel_size=2, activation='relu'))
        # model_3D.add(Flatten())
        # model_3D.add(Dropout(dropout_rate))
        # model_3D.add(Dense(6, input_dim=(3, 1), activation='relu'))
        # model_3D.add(Dropout(dropout_rate))
        # model_3D.add(Dense(4, activation='relu'))
        # model_3D.add(Dense(3, activation='softmax'))
        #
        # model_300D = Sequential()
        # model_300D.add(Conv1D(filters=32, kernel_size=3, input_shape=(300, 1), activation='relu'))
        # model_300D.add(Conv1D(filters=16, kernel_size=3, activation='relu'))
        # model_300D.add(Flatten())
        # model_300D.add(Dropout(0.2))
        # model_300D.add(Dense(128, input_dim=(300, 1), activation='relu'))
        # model_300D.add(Dropout(0.2))
        # model_300D.add(Dense(64, activation='relu'))
        # model_300D.add(Dense(3, activation='softmax'))

        model_x = load_model(os.path.join('data', 'model_save', 'set_1', '1_3dim_rmsprop.h5'))
        model_x._name = "embedding_3_1"
        for index, layer in enumerate(model_x.layers):
            model_x._name = 'ensemble_1'+str(index+1)+ model_x.name
        model_y = load_model(os.path.join('data', 'model_save', 'set_1', '1_17dim_rmsprop.h5'))
        model_y._name = "embedding_17_1"

        for index, layer in enumerate(model_y.layers):
            model_y._name = 'ensemble_2'+str(index+1)+ model_y.name
        model_z = load_model(os.path.join('data', 'model_save', 'set_1', '1_300dim_rmsprop.h5'))
        model_z._name = "embedding_300_1"

        for index, layer in enumerate(model_z.layers):
            model_z._name = 'ensemble_3'+str(index+1)+ model_z.name

        input_1 = model_x.inputs
        input_2 = model_y.inputs
        input_3 = model_z.inputs

        print(input_1)
        print(input_2)
        print(input_3)

        input_1.name = "embedding_3"
        input_2.name = "embedding_17"
        input_3.name = "embedding_300"

        input_1._type_spec._name = "embedding_3"
        input_2._type_spec._name = "embedding_17"
        input_3._type_spec._name = "embedding_300"

        print(input_1)
        print(input_2)
        print(input_3)

        model_ = maximum([model_z.output, model_y.output, model_x.output])
        modelEns = Model(inputs=[input_3, input_2, input_1], outputs=model_, name='ensemble')

        modelEns.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=['acc'])
        # train the model
        print("[INFO] training model...")
        modelEns.fit(
            x=[train_x_300D, train_x_17D, train_x_3D], y=train_y_3D,
            validation_data=([test_x_300D, test_x_17D, test_x_3D], test_y_3D),
            epochs=2, batch_size=32)
        print("Evaluate on test data")
        results = modelEns.evaluate([test_x_300D, test_x_17D, test_x_3D], test_y_3D, batch_size=32)
        print(results)
        exit(0)
        data_dict[str(n_set + 1)] = [results[1] * 100]
        accuracy.append(results[1] * 100)
    data_dict['avg'] = np.mean(accuracy)
    print("Average values is : ", np.mean(accuracy))
    df = pd.DataFrame(data_dict)
    df.to_csv('result.csv')