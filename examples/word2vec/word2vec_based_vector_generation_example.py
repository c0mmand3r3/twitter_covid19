"""
 - Author : Anish Basnet
 - Email : anishbasnetworld@gmail.com
 - Date : Aug 23, 2020, 07:37 AM
"""
import os
import pandas as pd

from gensim.models import Word2Vec

from tweeter_covid19.utils import read_pickle_data,mkdir
N_SETS = 10
if __name__ == '__main__':
    for fold in range(N_SETS):

        model_write_path = os.path.join('data', 'word2vec_tweeter', 'set_' + str(fold + 1))
        mkdir(model_write_path)
        model_write_path = os.path.join(model_write_path, 'word2vec_embedding_model.bin')
        corpus_path = os.path.join('data', 'fold_train_test_dataset_vectors',  'set_' + str(fold + 1),
                                   'train.csv')

        corpus = pd.read_csv(corpus_path)
        sentences = [tweet.split(' ') for tweet in corpus['Tweet']]
        model = Word2Vec(sentences, vector_size=300, min_count=1)
        print(model)

        model.save(model_write_path)
        print("Vector model successfully generated")
