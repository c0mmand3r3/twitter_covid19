"""
 - Author : Anish Basnet
 - Email : anishbasentworld@gmail.com
 - Date : Wednesday, May 13, 2020
"""
import os
from collections import OrderedDict

from news_classification.tfidf import get_tf_icf_vector, get_icf, write_pickle_data
from news_classification.utils import list_files, read_pickle_data, get_file_name, read_data_from_file, \
    get_all_directory

if __name__ == '__main__':
    write_path = os.path.join('data', 'processing', 'tf-icf', 'vectors', 'fusion_news', 'sets', 'set_1',
                              'tf_icf_test.pkl')

    files_path = os.path.join('data', 'processing', 'tokens', 'fusion_news', 'sets', 'set_1',
                              'dataset_stemmer_with_rasuwa_dirga_filter',
                              'test')

    classes = get_all_directory(files_path)

    frequency_optimizer_path = os.path.join('data', 'processing',
                                            'tf-idf', 'bow_optimizer', 'fusion_news', 'sets', 'set_1',
                                            'test_bow_frequency_optimizer.pkl')

    model_path = os.path.join('data', 'processing', 'label_based_frequency', 'fusion_news', 'sets', 'set_1',
                              'optimizer.pkl')
    bow_path = os.path.join('data', 'processing', 'neighbour_concept_third_level_filter_if_neighbour_freq_greater',
                            'fusion_news', 'sets', 'set_1',
                            'filter_bank.txt')

    frequency_optimizer = read_pickle_data(frequency_optimizer_path)
    model = read_pickle_data(model_path)
    bow = [(token, 0) for token in read_data_from_file(bow_path)]

    files = list_files(files_path)

    vector = OrderedDict()
    icf_count = get_icf(bow, model.word_with_freq)
    for index, file in enumerate(files):
        sub_vector = OrderedDict()

        sub_vector[get_file_name(file, 2)] = get_tf_icf_vector(frequency_optimizer[file], icf_count,
                                                               total_classes=len(classes))
        vector[file] = sub_vector
        print("{} Successfully written! Remaining : {}/{} | "
              "vector length : {} .".format(get_file_name(file, 2), index + 1, len(files), len(vector)))
    write_pickle_data(write_path, vector)
