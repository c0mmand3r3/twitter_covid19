"""
 - Author : Anish Basnet
 - Email : anishbasentworld@gmail.com
 - Date : Friday, May 8, 2020
"""
import os
from collections import OrderedDict

from tweeter_covid19.tfidf.histogram_generator import get_tf_idf_vector
from tweeter_covid19.utils import read_pickle_data, list_files, get_file_name

if __name__ == '__main__':
    write_path = os.path.join('data', 'processing', 'tf-idf', 'vectors', 'fusion_news', 'sets', 'set_1',
                              'tf_idf_test.pkl')

    files_path = os.path.join('data', 'processing', 'tokens', 'fusion_news', 'sets', 'set_1',
                              'dataset_stemmer_with_rasuwa_dirga_filter',
                              'test')

    frequency_optimizer_path = os.path.join('data', 'processing', 'tf-idf', 'bow_optimizer',
                                            'fusion_news', 'sets', 'set_1',
                                            'test_bow_frequency_optimizer.pkl')

    idf_optimizer_path = os.path.join('data', 'processing', 'tf-idf', 'bow_optimizer',
                                      'fusion_news', 'sets', 'set_1', 'train_idf_optimizer.pkl')

    frequency_optimizer = read_pickle_data(frequency_optimizer_path)

    idf_optimizer = read_pickle_data(idf_optimizer_path)

    files = list_files(files_path)

    vector = OrderedDict()
    for index, file in enumerate(files):
        sub_vector = OrderedDict()

        vectors = get_tf_idf_vector(frequency_optimizer[file], idf_optimizer)
        sub_vector[get_file_name(file, 2)] = vectors
        vector[file] = sub_vector
        print("{} Successfully written! Remaining : {}/{} | "
              "vector length : {} .".format(get_file_name(file, 2), index + 1, len(files), len(vector)))

    write_pickle_data(write_path, vector)


