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
    BoW_path = os.path.join('data', 'processing', 'neighbour_concept_third_level_filter_if_neighbour_freq_greater',
                            '24NepaliNews',
                            'filter_bank.txt')

    additional_stop_word_path = os.path.join('resources', 'dictionary', 'additional_stop_words.txt')
    write_path = os.path.join('data', 'processing', 'tf-idf', 'filter_bank_optimizer',
                              '24NepaliNews', 'train_idf_optimizer.pkl')

    files_path = os.path.join('data', 'processing', 'stemmer', '24NepaliNews',
                              'dataset_stemmer_with_rasuwa_dirga_filter', '24NepaliNews',
                              'train')

    labels = get_all_directory(files_path)

    BoW = [(token, 0) for token in read_data_from_file(BoW_path)]

    stop_words = read_data_from_file(additional_stop_word_path)

    filtered_BoW = bow_filter(stop_words, BoW)

    optimizer_model = Bow_IDF_Optimizer(filtered_BoW, labels)

    for label in labels:
        files = list_files(os.path.join(files_path, label))
        for file in files:
            file_tokens = read_data_from_file(file)
            filtered_file_tokens = token_filter(stop_words, file_tokens)
            optimizer_model.fit(file, filtered_file_tokens)
        print("{}  - Completed".format(label))
    optimizer_model.calculate_idf_frequency(save_path=write_path)
