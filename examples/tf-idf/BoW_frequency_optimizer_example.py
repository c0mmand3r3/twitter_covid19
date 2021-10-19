"""
 - Author : Anish Basnet
 - Email : anishbasentworld@gmail.com
 - Date : Friday, May 8, 2020
"""
import os
import re

import numpy as np
import pandas as pd

from tweeter_covid19 import read_pickle_data, write_pickle_data
from tweeter_covid19.tfidf.bow_optimizer import bow_filter

from tweeter_covid19.utils import mkdir
from tweeter_covid19.utils import get_all_directory, read_data_from_file, list_files

N_SETS = 10
if __name__ == '__main__':
    for fold in range(N_SETS):
        BoW_path = os.path.join('data', 'tfidf_processing', 'tf-idf', 'BoW', 'set_' + str(fold + 1),
                                'bag_of_word.pkl')

        additional_stop_word_path = os.path.join('resources', 'dictionary', 'additional_stop_words.txt')

        train_path = os.path.join('data', 'fold_train_test_dataset_vectors', 'set_' + str(fold + 1), 'train.csv')

        test_path = os.path.join('data', 'fold_train_test_dataset_vectors', 'set_' + str(fold + 1), 'test.csv')

        write_path = os.path.join('data', 'processing', 'tfidf_vector_test', 'set_' + str(fold + 1))
        mkdir(write_path)

        BoW = read_pickle_data(BoW_path)
        stop_words = read_data_from_file(additional_stop_word_path)
        filtered_BoW = bow_filter(stop_words, BoW)[0:100]
        print(filtered_BoW)
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        print(train_df.shape)

        # first preprocess IDF count - single vector of 1 x 100
        print('Processing ')
        train_Y = np.array([])
        test_Y = np.array([])

        # exit(0)
        # tf_vector_train = np.zeros(shape=(train_df['Tweet'].shape[0], 100), dtype=float)
        # tf_vector_test = np.zeros(shape=(test_df['Tweet'].shape[0], 100), dtype=float)

        tf_vector_train = []
        tf_vector_test = []

        idf_vector = np.zeros(shape=(100,), dtype=int)
        for t_index, tweet in enumerate(train_df['Tweet']):
            if len(tweet.split(' ')) > 2:
                train_Y = np.append(train_Y, train_df['Label'][t_index])
                vec = np.zeros(shape=(100,), dtype=float)
                for f_index, (bow_word, freq) in enumerate(filtered_BoW):
                    found_items = re.findall(bow_word, tweet)
                    vec[f_index] = len(found_items) / len(tweet.split(' '))
                    if len(found_items) != 0:
                        idf_vector[f_index] += 1
                tf_vector_train.append(vec)
            # exit(0)

        for t_index, tweet in enumerate(test_df['Tweet']):
            if len(tweet.split(' ')) > 2:
                test_Y = np.append(test_Y, test_df['Label'][t_index])
                vec = np.zeros(shape=(100,), dtype=float)
                for f_index, (bow_word, freq) in enumerate(filtered_BoW):
                    found_items = re.findall(bow_word, tweet)
                    vec[f_index] = len(found_items) / len(tweet.split(' '))
                tf_vector_test.append(vec)

        idf_vector = train_df.shape[0] / idf_vector

        idf_vector[idf_vector == np.inf] = 0
        idf_vector = np.log10(idf_vector + 1)

        train_X = tf_vector_train * idf_vector

        test_X = tf_vector_test * idf_vector


        write_pickle_data(os.path.join(write_path, 'train_x.pkl'), train_X)
        write_pickle_data(os.path.join(write_path, 'train_y.pkl'), train_Y)
        write_pickle_data(os.path.join(write_path, 'test_x.pkl'), test_X)
        write_pickle_data(os.path.join(write_path, 'test_y.pkl'), test_Y)
        print('{} - Successfully created'.format(fold + 1))
