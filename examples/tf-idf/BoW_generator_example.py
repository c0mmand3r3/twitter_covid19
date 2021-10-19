"""
 - Author : Anish Basnet
 - Email : anishbasnetworld@gmail.com
 - Date : May 8, 2020
"""
import os

import numpy as np

from tweeter_covid19 import read_pickle_data, write_pickle_data
from tweeter_covid19.utils import mkdir
N_SETS = 10
if __name__ == '__main__':
    for fold in range(N_SETS):

        write_path = os.path.join('data', 'processing', 'tf-idf', 'BoW', 'set_' + str(fold + 1))
        mkdir(write_path)
        # corpus_path = os.path.join('data', 'processing', 'corpus', 'fusion_news', 'sets', 'set_1',
        #                                                            'corpus_stemmer.txt')

        model_path = os.path.join('data', 'distance_based', 'processing', 'label_based_frequency',
                                  'set_' + str(fold + 1), 'optimizer.pkl')
        write_path = os.path.join(write_path, 'bag_of_word.pkl')

        freq_model = read_pickle_data(model_path)
        words = []
        for key in freq_model.word_with_freq:
            words.append((key, np.sum(freq_model.word_with_freq[key])))
        word_with_freq = sorted(words, key=lambda x: -x[1])
        write_pickle_data(write_path, word_with_freq)
        print("Successfully Executed!")
