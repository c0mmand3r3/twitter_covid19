"""
 - Author : Anish Basnet
 - Email : anishbasneworld@gmail.com
 - Date : Aug 24, 2020
"""
import os
import pandas as pd
import numpy as np
from gensim.models import Word2Vec

from tweeter_covid19.utils import list_files, read_data_from_file, get_file_name, write_pickle_data, mkdir

N_SETS = 10
if __name__ == '__main__':
    for fold in range(N_SETS):
        LSTM_SERIAL_N = 10

        MODEL_VECTOR_LENGTH = 300

        data_path = os.path.join('data', 'fold_train_test_dataset_vectors', 'set_' + str(fold + 1),
                                 'test.csv')

        model_path = os.path.join('data', 'word2vec_tweeter', 'set_' + str(fold + 1), 'word2vec_embedding_model.bin')

        write_path = os.path.join('data', 'lstm_based_histogram', 'set_' + str(fold + 1))
        mkdir(write_path)

        model = Word2Vec.load(model_path)

        vocabulary = list(model.wv.key_to_index.keys())
        files = pd.read_csv(data_path)['Tweet']

        train_x_vec = []
        train_y_vec = pd.read_csv(data_path)['Label']
        for index, file in enumerate(files):
            tokens = file.split(' ')

            doc_vectors = np.zeros((LSTM_SERIAL_N, MODEL_VECTOR_LENGTH), dtype=float)
            if len(tokens) >= LSTM_SERIAL_N:
                for token_index, token in enumerate(tokens[0:LSTM_SERIAL_N]):
                    if token in vocabulary:
                        doc_vectors[token_index] = model.wv[token]
            else:
                for token_index, token in enumerate(tokens):
                    if token in vocabulary:
                        doc_vectors[token_index] = model.wv[token]
            train_x_vec.append(doc_vectors)

        write_pickle_data(os.path.join(write_path, 'test_x.pkl'), train_x_vec)
        write_pickle_data(os.path.join(write_path, 'test_y.pkl'), train_y_vec)
        print(np.shape(train_x_vec), ' ', np.shape(train_y_vec))
        print("{} - Successfully Completed".format(fold+1))