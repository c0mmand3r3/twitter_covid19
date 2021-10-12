"""
 - Author : Anish Basnet
 - Email : anishbasnetworld@gmail.com
 - Date : Friday, May 8, 2020
"""
import numpy as np

from tweeter_covid19.utils import write_pickle_data


def bow_filter(_stop_words, _BoW):
    bow_tokens = []
    for token, freq in _BoW:
        bow_tokens.append((token, freq))
    return bow_tokens


def token_filter(_stop_words, _tokens):
    filter_tokens = []
    for token in _tokens:
        if not (token in _stop_words):
            filter_tokens.append(token)
    return filter_tokens


class Bow_Frequency_Optimizer:
    def __init__(self, bag_of_words, labels):
        self.bag_of_words = bag_of_words
        self.labels = labels
        self.file_with_tokens = dict()
        self.file_with_bow_frequency = dict()

    def fit(self, file, tokens):
        self.file_with_tokens[file] = tokens

    def calculate_bow_frequency(self, save_path, verbose=True):
        for _index, file in enumerate(self.file_with_tokens):
            frequency = np.zeros(len(self.bag_of_words), dtype=int)
            for index, (token, freq) in enumerate(self.bag_of_words):
                for bow_token in self.file_with_tokens[file]:
                    if token == bow_token:
                        frequency[index] += 1
            self.file_with_bow_frequency[file] = frequency
            if verbose:
                print("Frequency Calculation Successfully executed! - {}/{}".format(_index,
                                                                                    len(self.file_with_tokens)))
        write_pickle_data(save_path, self.file_with_bow_frequency)


class Bow_IDF_Optimizer:
    def __init__(self, bag_of_words, labels):
        self.bag_of_words = bag_of_words
        self.labels = labels
        self.file_with_tokens = dict()
        self.file_with_idf_frequency = dict()

    def fit(self, file, tokens):
        self.file_with_tokens[file] = tokens

    def calculate_idf_frequency(self, save_path):
        """
        This is only for training file.
        :param save_path:
        :param verbose:
        :return:
        """
        total_frequency = np.zeros(len(self.bag_of_words), dtype=int)
        for index, (token, freq) in enumerate(self.bag_of_words):
            for search_index, search_file in enumerate(self.file_with_tokens):
                for search_token in self.file_with_tokens[search_file]:
                    if token == search_token:
                        total_frequency[index] += 1
                        break
        write_pickle_data(save_path, (len(self.file_with_tokens), total_frequency))
