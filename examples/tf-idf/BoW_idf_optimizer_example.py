"""
 - Author : Anish Basnet
 - Email : anishbasentworld@gmail.com
 - Date : Friday, May 8, 2020
"""
import os

from news_classification import read_pickle_data
from news_classification.tfidf import Bow_IDF_Optimizer, bow_filter, token_filter
from news_classification.utils import get_all_directory, read_data_from_file, list_files

if __name__ == '__main__':
    BoW_path = os.path.join('data', 'processing', 'tf-idf', 'BoW', 'fusion_news', 'sets', 'set_1',
                            'bag_of_word.pkl')

    additional_stop_word_path = os.path.join('resources', 'dictionary', 'additional_stop_words.txt')
    write_path = os.path.join('data', 'processing', 'tf-idf', 'bow_optimizer', 'fusion_news',
                              'sets', 'set_1', 'train_idf_optimizer.pkl')

    files_path = os.path.join('data', 'processing', 'tokens', 'fusion_news', 'sets', 'set_1',
                              'dataset_stemmer_with_rasuwa_dirga_filter',
                              'train')

    filter_bank_path = os.path.join('data', 'processing',
                                    'neighbour_concept_third_level_filter_if_neighbour_freq_greater'
                                    , 'fusion_news', 'sets', 'set_1', 'filter_bank.txt')

    filter_bank = read_data_from_file(filter_bank_path)

    labels = get_all_directory(files_path)

    BoW = read_pickle_data(BoW_path)
    stop_words = read_data_from_file(additional_stop_word_path)

    filtered_BoW = bow_filter(stop_words, BoW)[0:len(filter_bank)]

    optimizer_model = Bow_IDF_Optimizer(filtered_BoW, labels)

    for label in labels:
        files = list_files(os.path.join(files_path, label))
        for file in files:
            file_tokens = read_data_from_file(file)
            filtered_file_tokens = token_filter(stop_words, file_tokens)
            optimizer_model.fit(file, filtered_file_tokens)
        print("{}  - Completed".format(label))
    optimizer_model.calculate_idf_frequency(save_path=write_path)
