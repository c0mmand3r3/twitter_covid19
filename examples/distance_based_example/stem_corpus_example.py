"""
 - Author : Anish Basnet
 - Email : anishbasnetworld@gmail.com
 - Date : April 13, 2020
"""
import os
from tweeter_covid19.corpus_generator import Doc2corpus
from tweeter_covid19.utils import read_file

N_SETS = 10

if __name__ == '__main__':
    corpus_path = os.path.join('data', 'distance_based', 'processing', 'corpus')
    for fold in range(N_SETS):
        sent_tokens = read_file(os.path.join(corpus_path, 'set_' + str(fold+1), 'corpus.txt'))
        model = Doc2corpus()
        model.fit(sent_tokens)
        model.save_model(os.path.join('data', 'distance_based', 'processing', 'corpus', 'set_' + str(fold+1),
                                      'corpus_stemmer.pkl'))
