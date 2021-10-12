"""
 - Author : Anish Basnet
 - Email : anishbasnetworld@gmail.com
 - Date : Aug 23, 2020, 07:37 AM
"""
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
import os

from examples.classification_example import get_data_split
from tweeter_covid19 import read_pickle_data
from tweeter_covid19.utils import list_files, get_file_name


def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs
        :param history: Training history of model
        :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()


def build_model(input_shape, output_dim):
    """Generates RNN-LSTM model
    :param input_shape (tuple): Shape of input set
    :return model: RNN-LSTM model
    """

    # build network topology
    model = keras.Sequential()

    # 2 LSTM layers
    model.add(keras.layers.LSTM(300, input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.LSTM(150))

    # dense layer
    model.add(keras.layers.Dense(150, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(output_dim, activation='softmax'))

    return model


def read_data(files):
    X, Y = [], []
    for file in files:
        X.append(read_pickle_data(file)[get_file_name(file, 2)])
        Y.append(get_file_name(file, 2))
    return X, Y

N_SETS = 10

if __name__ == '__main__':
    accuracy = np.array([])
    for fold in range(N_SETS):

        train_x = read_pickle_data(os.path.join('data', 'lstm_based_histogram', 'set_' + str(fold + 1), 'train_x.pkl'))
        train_y = read_pickle_data(os.path.join('data', 'lstm_based_histogram', 'set_' + str(fold + 1), 'train_y.pkl'))
        test_x = read_pickle_data(os.path.join('data', 'lstm_based_histogram', 'set_' + str(fold + 1), 'test_x.pkl'))
        test_y = read_pickle_data(os.path.join('data', 'lstm_based_histogram', 'set_' + str(fold + 1), 'test_y.pkl'))

        output_dim = len(list(set(train_y)))

        number_generate = list(set(test_y))
        get_index = lambda key: number_generate.index(key)
        train_y = [get_index(x) for x in train_y]
        test_y = [get_index(x) for x in test_y]
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        test_x = np.array(test_x)
        test_y = np.array(test_y)

        print(np.shape(train_x), np.shape(train_y))
        print(np.shape(test_x), np.shape(test_y))

        model = build_model((np.shape(train_x)[1], np.shape(train_x)[2]), output_dim)

        # compile model
        optimiser = keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=optimiser,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.summary()

        # train model
        history = model.fit(train_x, train_y, validation_data=(test_x, test_y), batch_size=32, epochs=15)

        # plot accuracy/error for training and validation
        plot_history(history)

        # evaluate model on test set
        test_loss, test_acc = model.evaluate(test_x, test_y, verbose=2)
        accuracy = np.append(accuracy, test_acc)
        print('\nTest accuracy:', test_acc)
    print('Final Mean Accuracy = {}'.format(np.mean(accuracy)))
