"""
 - Author : Anish Basnet
 - Email : anishbasnetworld@gmail.com
 - Date : Saturday, July 17, 2021
"""
import os
import pandas as pd
import numpy as np

from tweeter_covid19.utils import mkdir
from tweeter_covid19.utils.pickleutils import read_pickle_data, write_pickle_data

SETS = 10

TOKENS_N_NUMBERS = 12
VECTOR_LEN = 17

if __name__ == '__main__':
    write_path = os.path.join('data', 'fold_train_test_dataset_vectors')

    data_path = os.path.join('data', 'fold_dataset')

    optimizer_path = os.path.join('C:\\Users\\Anish\\news_classification\\data\\vectors\\training_nodes_vectors\\'
                                  'nepali_linguistic\\sets\\set_1\\direct_training_vectors_final.pkl')

    optimizer_log_path = os.path.join('C:\\Users\\Anish\\news_classification\\data\\vectors\\training_nodes_vectors\\'
                                      'nepali_linguistic\\sets\\set_1\\logger.pkl')

    word_with_freq_fold = os.path.join('data', 'word_with_frequency_folds')

    tokens = read_pickle_data(optimizer_log_path)

    vectors_optimizer = read_pickle_data(optimizer_path)

    for n_set in range(SETS):

        data_dict = {
            'File_Key': [],
            'Label': [],
            'Datetime': [],
            'Tweet': [],
            'Tokenize_tweet': [],
        }

        write_joiner_path = os.path.join(write_path, 'set_' + str(n_set + 1))
        read_joiner_path = os.path.join(data_path, "set_" + str(n_set + 1), "train.csv")
        word_with_freq = read_pickle_data(os.path.join(word_with_freq_fold,
                                                       'set_' + str(n_set + 1), 'train_wwf.pkl'))
        mkdir(write_joiner_path)
        vector_writer_joiner_path = os.path.join(write_joiner_path, 'train')
        mkdir(vector_writer_joiner_path)
        data = pd.read_csv(read_joiner_path)

        for index, tweet in enumerate(data['Tweet']):
            tweet_vectors = np.zeros(shape=(TOKENS_N_NUMBERS, 17), dtype=float)
            checker = 0
            for tweet_token in tweet.split(' '):
                if checker < TOKENS_N_NUMBERS:
                    if tweet_token in tokens:
                        try:
                            tweet_vectors[checker] = vectors_optimizer[tweet_token]
                        except Exception:
                            pass
                        checker += 1
            data_dict['File_Key'].append(index)
            data_dict['Label'].append(data['Label'][index])
            data_dict['Datetime'].append(data['Datetime'][index])
            data_dict['Tweet'].append(data['Tweet'][index])
            data_dict['Tokenize_tweet'].append(data['Tokenize_tweet'][index])
            write_pickle_data(os.path.join(vector_writer_joiner_path, str(index)+'.pkl'), tweet_vectors)
            if index % 1000 == 0:
                print('{}/{} -- Successfully Created! ---- Set : {}'.format(index, len(data['Tweet']), n_set+1))
        df = pd.DataFrame(data_dict)
        df.to_csv(os.path.join(write_joiner_path, 'train.csv'))
        print("{} - Set CSV file Successfully created")
