"""
 - Author : Anish Basnet
 - Email : anishbasnetworld@gmail.com
 - Date : April 23, 2020
"""
import pandas as pd
import os

from tweeter_covid19.utils import mkdir, write_text_file_linewise

N_SETS = 10


def process_data(data_values):
    return [tok for token in data_values['Tweet'] for tok in token.split(' ')]


if __name__ == '__main__':
    data_paths = os.path.join('data', 'fold_dataset')

    write_path = os.path.join('data', 'distance_based', 'processing', 'label_based_corpus')

    for fold in range(N_SETS):
        reader_path_joiner = os.path.join(data_paths, 'set_' + str(fold + 1), 'train.csv')
        write_path_joiner = os.path.join(write_path, 'set_' + str(fold + 1))

        data = pd.read_csv(reader_path_joiner)

        positive_label_data = data.query('Label == 1')
        negative_label_data = data.query('Label == -1')
        neutral_label_data = data.query('Label == 0')

        mkdir(os.path.join(write_path_joiner, '1'))
        mkdir(os.path.join(write_path_joiner, '-1'))
        mkdir(os.path.join(write_path_joiner, '0'))

        positive_tokens = process_data(positive_label_data)
        negative_tokens = process_data(negative_label_data)
        neutral_tokens = process_data(neutral_label_data)

        write_text_file_linewise(os.path.join(write_path_joiner, '1', '1.txt'), positive_tokens,
                                 encoding='utf-8')
        write_text_file_linewise(os.path.join(write_path_joiner, '-1', '-1.txt'), negative_tokens,
                                 encoding='utf-8')
        write_text_file_linewise(os.path.join(write_path_joiner, '0', '0.txt'), neutral_tokens, encoding='utf-8')
        print("{} - Successfully executed . ".format(fold + 1))

