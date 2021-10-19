"""
 - Author : Anish Basnet
 - Email : anishbasnetworld@gmail.com
 - Date : Friday, May 8, 2020
"""
import numpy as np


def get_tf_idf_vector(frequency_vector, idf_optimizer):
    vector = np.zeros(len(frequency_vector), dtype=float)
    if len(frequency_vector) == len(idf_optimizer[1]):
        for index, freq in enumerate(frequency_vector):
            vector[index] = freq * np.log10(idf_optimizer[0] / (idf_optimizer[1][index] + 1))
        return vector
    else:
        return None


def get_tf_icf_vector(frequency_vector, icf_count, total_classes):
    vector = np.zeros(len(frequency_vector), dtype=float)
    for index, freq in enumerate(frequency_vector):
        vector[index] = freq * np.log10(1+(total_classes/icf_count[index]))
    return vector


def get_icf(bow, word_with_freq):
    word_with_category_count = []
    for word, count in bow:
        total_count = 0
        for value in word_with_freq[word]:
            if value != 0:
                total_count += 1
        word_with_category_count.append(total_count)
    return word_with_category_count


def convert_vector(vector):
    bool_vector = np.zeros(len(vector), dtype=int)
    for index, value in enumerate(vector):
        if value != 0:
            bool_vector[index] = 1
    return bool_vector
