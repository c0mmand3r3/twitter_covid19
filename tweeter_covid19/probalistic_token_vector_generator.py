"""
 - Author : Anish Basnet
 - Email : anishbasnetworld@gmail.com
 - date : April 23, 2020
"""
import numpy as np
from numpy import linalg


class Token2vector:
    def __init__(self, model):
        self.model = model
        self.labels = None
        self.label_dim = None
        self.label_with_token = dict()
        self.feature_vector = None

    def fit(self):
        self.labels = self.model.model_17D.labels
        self.label_dim = len(self.labels)

    def generate_vector(self, edges):
        node_vector = []
        for edge in edges:
            if len(edges) >= 100000:
                return None
            vector = np.zeros(self.label_dim, dtype=float)
            if edge in self.model.word_with_freq:
                frequency = self.model.word_with_freq[edge]
                for index, label in enumerate(self.labels):
                    vector[index] = frequency[index]/len(self.model.model_17D.tokens[index])
                node_vector.append(vector / linalg.norm(vector) + 0.00000008)
            else:
                node_vector.append(vector)
        try:
            _node_vector = np.sum(node_vector, axis=0)
        except MemoryError:
            _node_vector = None
        return _node_vector

    def generate_direct_vectors(self, node):
        vector = np.zeros(self.label_dim, dtype=float)
        if node in self.model.word_with_freq:
            frequency = self.model.word_with_freq[node]
            for index, label in enumerate(self.labels):
                vector[index] = frequency[index] / len(self.model.model_17D.tokens[index])
        return vector
