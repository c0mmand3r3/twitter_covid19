"""
 - Author : Anish Basnet, Ashish Mainali
 - Email : anishbasnetworld@gmail.com, mainaliashish@outlook.com
 - Date : Saturday, July 17, 2021
"""
import os
from collections import Counter

import pandas as pd

from tweeter_covid19.utils import mkdir, write_pickle_data

SETS = 10


def convert_sentences_to_list(string):
    li = list(string.split(","))
    return li


if __name__ == '__main__':
    data_path = os.path.join('data', 'fold_dataset')
    word_freq_path = os.path.join('data', 'word_with_frequency_folds')

    for n_set in range(SETS):
        joiner_path = os.path.join(word_freq_path, 'set_' + str(n_set + 1))
        mkdir(joiner_path)

        df = pd.read_csv(os.path.join(data_path, 'set_' + str(n_set + 1), 'train.csv'))

        freq_dict = dict()

        mergedlist = []

        for line in df['Tokanize_tweet']:
            tokens = convert_sentences_to_list(line)
            for token in tokens:
                mergedlist.append(token)

        res = Counter(mergedlist).most_common()
        for (token, freq) in res:
            freq_dict[token] = freq

        write_pickle_data(os.path.join(joiner_path, 'train_wwf.pkl'), freq_dict)
        print('Set - {} Successfully completed!'.format(n_set+1))
