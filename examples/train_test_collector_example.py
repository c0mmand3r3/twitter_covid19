"""
 - Author : Anish Basnet
 - Email : anishbasnetworld@gmail.com
 - Date : Sunday, July 18, 2021
"""
import os
import pandas as pd
import numpy as np

from tweeter_covid19.utils.pickleutils import read_pickle_data, write_pickle_data
from tweeter_covid19.utils import mkdir

N_SETS = 10

if __name__ == '__main__':

    writer_main_path = os.path.join('data', 'fold_train_test_collector')

    read_main_path = os.path.join('data', 'fold_train_test_dataset_vectors')

    for n_set in range(N_SETS):
        read_joiner_path = os.path.join(read_main_path, 'set_' + str(n_set + 1))
        writer_joiner_path = os.path.join('data', 'fold_train_test_collector', 'set_' + str(n_set + 1))
        mkdir(writer_joiner_path)

        train_data = pd.read_csv(os.path.join(read_joiner_path, 'train.csv'))

        test_data = pd.read_csv(os.path.join(read_joiner_path, 'test.csv'))

        train_X = []
        train_Y = []

        test_X = []
        test_Y = []

        print(' -------- Initializing loading training data ---------')
        for train_index, t_label in enumerate(train_data['Label']):
            vector = read_pickle_data(os.path.join(read_joiner_path, 'train',
                                                   str(train_data['File_Key'][train_index]) + '.pkl'))
            train_X.append(vector)
            train_Y.append(float(t_label))
        print(' -------- Successfully Completed loading training data ---------')

        print(' -------- Initializing loading testing data ---------')
        for test_index, tt_label in enumerate(test_data['Label']):
            vector = read_pickle_data(os.path.join(read_joiner_path, 'test',
                                                   str(train_data['File_Key'][test_index]) + '.pkl'))
            test_X.append(vector)
            test_Y.append(float(tt_label))
        print(' -------- Successfully Completed loading testing data ---------')

        print(np.shape(train_X))
        print(np.shape(train_Y))
        print(np.shape(test_X))
        print(np.shape(test_Y))

        write_pickle_data(os.path.join(writer_joiner_path, 'train_x.pkl'), train_X)
        write_pickle_data(os.path.join(writer_joiner_path, 'train_y.pkl'), train_Y)
        write_pickle_data(os.path.join(writer_joiner_path, 'test_x.pkl'), test_X)
        write_pickle_data(os.path.join(writer_joiner_path, 'test_y.pkl'), test_Y)
        print('{} - Successfully created'.format(n_set + 1))
