"""
 - Author : Anish Basnet
 - Email : anishbasentworld@gmail.com
 - Date : Saturday, May 9, 2020
"""
import os
from collections import OrderedDict

from news_classification.tfidf import write_pickle_data, convert_vector
from news_classification.utils import read_pickle_data, list_files, get_file_name

if __name__ == '__main__':
    write_path = os.path.join('data', 'processing', 'tf-idf', 'vectors', 'fusion_news', 'sets', 'set_1',
                              'boolean_train.pkl')

    files_path = os.path.join('data', 'processing', 'tokens', 'fusion_news', 'sets', 'set_1',
                              'dataset_stemmer_with_rasuwa_dirga_filter', 'train')

    frequency_optimizer_path = os.path.join('data', 'processing', 'tf-idf', 'bow_optimizer',
                                            'fusion_news', 'sets', 'set_1', 'train_bow_frequency_optimizer.pkl')
    frequency_optimizer = read_pickle_data(frequency_optimizer_path)

    files = list_files(files_path)

    vector = OrderedDict()
    for index, file in enumerate(files):
        sub_vector = OrderedDict()

        vectors = frequency_optimizer[file]

        sub_vector[get_file_name(file, 2)] = convert_vector(vectors)
        vector[file] = sub_vector
        print("{} Successfully written! Remaining : {}/{} | "
              "vector length : {} .".format(get_file_name(file, 2), index + 1, len(files), len(vector)))

    write_pickle_data(write_path, vector)
