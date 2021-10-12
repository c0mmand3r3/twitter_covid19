"""
 - Author : Anish Basnet
 - Email : anishbasneworld@gmail.com
 - Date : Aug 24, 2020
"""
import os

import numpy as np

from news_classification.utils import list_files, read_data_from_file, get_file_name, write_pickle_data, mkdir, \
    read_pickle_data

if __name__ == '__main__':

    LSTM_SERIAL_N = 50

    data_path = os.path.join('data', 'processing', 'stemmer', 'nepali_news_dataset_20_categories_large', 'sets',
                             'set_2',
                             'dataset_stemmer_with_rasuwa_dirga_filter', 'train')

    optimizer_path = os.path.join('data', 'vectors', 'distance_vector_optimizer',
                                  'nepali_news_dataset_20_categories_large', 'sets', 'set_2', 'vectors')

    optimizer_log_path = os.path.join('data', 'vectors', 'distance_vector_optimizer',
                                      'nepali_news_dataset_20_categories_large', 'sets', 'set_2', 'optimizer_log.pkl')

    write_path = os.path.join('data', 'vectors', 'lstm_based_histogram_distance_based_vector', 'nepali_news_dataset_20_categories_large',
                              'sets', 'set_2',
                              'train')

    optimizer_model = read_pickle_data(optimizer_log_path)

    files = list_files(data_path)

    MODEL_VECTOR_LENGTH = len(read_pickle_data(list_files(optimizer_path)[0]))

    for index, file in enumerate(files):
        mkdir(os.path.join(write_path, get_file_name(file, 2)))
        sub_vector = dict()
        tokens = read_data_from_file(file)
        doc_vectors = np.zeros((LSTM_SERIAL_N, MODEL_VECTOR_LENGTH), dtype=float)
        if len(tokens) >= LSTM_SERIAL_N:
            for token_index, token in enumerate(tokens[0:LSTM_SERIAL_N]):
                if token in optimizer_model:
                    doc_vectors[token_index] = read_pickle_data(os.path.join(optimizer_path,
                                                                             str(optimizer_model.index(
                                                                                 token)) + '.pkl'))
        else:
            for token_index, token in enumerate(tokens):
                if token in optimizer_model:
                    doc_vectors[token_index] = read_pickle_data(os.path.join(optimizer_path,
                                                                             str(optimizer_model.index(
                                                                                 token)) + '.pkl'))
        sub_vector[get_file_name(file, 2)] = doc_vectors
        print("{} Successfully written! Remaining : {}/{} .".format(get_file_name(file, 2), index + 1, len(files)))
        write_pickle_data(os.path.join(write_path, get_file_name(file, 2), get_file_name(file, 1)), sub_vector)
