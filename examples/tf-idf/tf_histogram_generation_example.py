"""
 - Author : Anish Basnet
 - Email : anishbasentworld@gmail.com
 - Date : Saturday, May 9, 2020
"""
import os
from collections import OrderedDict

from news_classification.tfidf import get_tf_idf_vector, write_pickle_data
from news_classification.utils import read_pickle_data, list_files, get_file_name

if __name__ == '__main__':
    write_path = os.path.join('data', 'processing', 'tf-idf', 'vectors', '16NepaliNews', 'sets', 'set_1',
                              'tf_train.pkl')

    files_path = os.path.join('data', 'processing', 'stemmer', '16NepaliNews', 'sets', 'set_1',
                              'dataset_stemmer_with_rasuwa_dirga_filter',
                              'train')

    frequency_optimizer_path = os.path.join('data', 'processing', 'tf-idf', 'bow_optimizer',
                                            '16NepaliNews', 'sets', 'set_1',
                                            'train_bow_frequency_optimizer.pkl')
    frequency_optimizer = read_pickle_data(frequency_optimizer_path)

    files = list_files(files_path)

    vector = OrderedDict()
    for index, file in enumerate(files):
        sub_vector = OrderedDict()

        vectors = frequency_optimizer[file]
        sub_vector[get_file_name(file, 2)] = vectors
        vector[file] = sub_vector
        print("{} Successfully written! Remaining : {}/{} | "
              "vector length : {} .".format(get_file_name(file, 2), index + 1, len(files), len(vector)))

    write_pickle_data(write_path, vector)
