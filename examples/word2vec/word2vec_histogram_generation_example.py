"""
 - Author : Anish Basnet
 - Email : anishbasnetworld@gmail.com
 - Date : Sunday, Aug 23, 2020
"""
import os
import numpy as np
from gensim.models import Word2Vec

from news_classification import read_pickle_data
from news_classification.utils import list_files, read_data_from_file, get_file_name, write_pickle_data
from collections import OrderedDict

if __name__ == '__main__':
    backup_path = os.path.join('data', 'vectors', 'word2vec_based_histogram',
                               'nepali_news_dataset_20_categories_large', 'sets', 'set_1', 'backup.bak')
    log_backup_path = os.path.join('data', 'vectors', 'word2vec_based_histogram',
                                   'nepali_news_dataset_20_categories_large', 'sets', 'set_1', 'log_backup.bak')
    document_path = os.path.join('data', 'processing', 'stemmer', 'nepali_news_dataset_20_categories_large', 'sets',
                                 'set_1',
                                 'dataset_stemmer_with_rasuwa_dirga_filter',
                                 'train')

    histogram_write_path = os.path.join('data', 'vectors', 'word2vec_based_histogram',
                                        'nepali_news_dataset_20_categories_large', 'sets', 'set_1',
                                        'avg_distance_based_train_300_dim.pkl')

    log_path = os.path.join('data', 'vectors', 'word2vec_based_histogram',
                            'nepali_news_dataset_20_categories_large', 'sets', 'set_1',
                            'log_avg_distance_based_train_300_dim.pkl')

    word2vec_model_path = os.path.join('data', 'vectors', 'word2vec_embedding_vectors', 'nepali_news_dataset_20_categories_large', 'sets',
                                       'set_1', 'word2vec_embedding_model.bin')

    model = Word2Vec.load(word2vec_model_path)

    vocabulary = list(model.wv.vocab)

    additional_stop_word_path = os.path.join('resources', 'dictionary', 'additional_stop_words.txt')

    stop_words = read_data_from_file(additional_stop_word_path)

    files = list_files(document_path)

    log = read_pickle_data(log_path)

    if log is None:
        log = []

    vector = OrderedDict()
    if os.path.isfile(backup_path):
        vector = read_pickle_data(backup_path)
    if os.path.isfile(log_backup_path):
        log = read_pickle_data(log_backup_path)
    for index, file in enumerate(files):
        if file in log:
            print("{} histogram already created! Remaining : {}/{} .".format(
                get_file_name(file, 2), index + 1, len(files)))
        else:
            sub_vector = OrderedDict()
            tokens = read_data_from_file(file)
            doc_vectors = []
            for token in tokens:
                if not (token in stop_words):
                    if token in vocabulary:
                        doc_vectors.append(model[token])
            if len(np.shape(doc_vectors)) != 2:
                pool_vector = np.zeros((model.vector_size,), dtype=float)
            else:
                pool_vector = np.mean(doc_vectors, axis=0)
            sub_vector[get_file_name(file, 2)] = pool_vector
            vector[file] = sub_vector
            log.append(file)
            if index % 1000 == 0:
                # for backup the pickle file to prevent data loss
                if os.path.exists(histogram_write_path):
                    write_dir = get_file_name(histogram_write_path, 1, directory_only=True)
                    filename = get_file_name(histogram_write_path, 1)
                    backup_name = 'backup.bak'
                    if os.path.exists(os.path.join(write_dir, backup_name)):
                        os.remove(os.path.join(write_dir, backup_name))
                    os.rename(histogram_write_path, os.path.join(write_dir, backup_name))

                    # for backup the pickle file to prevent data loss
                    if os.path.exists(log_path):
                        write_dir = get_file_name(log_path, 1, directory_only=True)
                        filename = get_file_name(log_path, 1)
                        backup_name = 'log_backup.bak'
                        if os.path.exists(os.path.join(write_dir, backup_name)):
                            os.remove(os.path.join(write_dir, backup_name))
                        os.rename(log_path, os.path.join(write_dir, backup_name))

                write_pickle_data(histogram_write_path, vector)
                write_pickle_data(log_path, log)
            print("{} Successfully written! Remaining : {}/{} | "
                  "vector length : {} .".format(get_file_name(file, 2), index + 1, len(files), len(vector)))
    write_pickle_data(histogram_write_path, vector)
    write_pickle_data(log_path, log)
