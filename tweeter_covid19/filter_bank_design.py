"""
 - Author : Anish Basnet
 - Email : anishbasnetworld@gmail.com
 - Date : April 29, 2020
"""
import numpy as np

from tweeter_covid19.utils import flatten


def generate_filter_bank(corpus, tokens):
    edges = []
    for token in tokens:
        for sentence in corpus:
            if token in corpus[sentence]:
                edges.append(corpus[sentence])
    edges = flatten(edges)
    return edges


def get_label_vectors(embedding_vectors, labels):
    vectors = dict()
    for key in labels:
        vector = []
        for label in labels[key]:
            if label in embedding_vectors:
                vector.append(embedding_vectors[label])
            else:
                print(label)
                print("Sorry! {} - Label value error occurs . Terminating Execution!".format(key))
                exit(0)
        if len(vector) == 1:
            vectors[key] = vector[0]
        else:
            node_vector = np.mean(vector, axis=0)
            vectors[key] = node_vector
    return vectors


def get_neighbour_keys(keys, corpus):
    neighbour_keys = []
    for key in keys:
        for sentence in corpus:
            words = sentence.split(',')
            total_words = len(words)
            if key in words:
                word_index = [i for i in range(len(words)) if words[i] in key]
                for index in word_index:
                    if total_words != 1:
                        if index == 0:
                            neighbour_keys.append(words[1])
                        elif index == total_words - 1:
                            neighbour_keys.append(words[total_words - 2])
                        else:
                            neighbour_keys.append(words[index + 1])
                            neighbour_keys.append(words[index - 1])
    return list(set(neighbour_keys))


def frequency_based_neighbour_elimination(freq_model, keys, neighbours):
    collective_neighbours = []
    key_freq = 0
    for key in keys:
        if key in freq_model.word_with_freq:
            key_freq += np.sum(freq_model.word_with_freq[key])
            for neighbour in list(set(neighbours)):
                if neighbour in freq_model.word_with_freq:
                    # if neighbours frequency is greater than label frequency then select that neighbour for filter bank
                    if key_freq <= np.sum(freq_model.word_with_freq[neighbour]):
                        collective_neighbours.append(neighbour)
    return collective_neighbours
