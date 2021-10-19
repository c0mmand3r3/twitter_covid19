"""
 - author : Anish Basnet
 - email : anishbasnetworld@gmail.com
 - date : April 5, 2020
"""
import os
import pandas as pd

from tweeter_covid19.utils import list_files, read_data_from_file, detect_encoding, mkdir
from tweeter_covid19.corpus_generator import Doc2corpus

N_SETS = 10

if __name__ == '__main__':
    path = os.path.join('data', 'fold_dataset')
    corpus_save_path = os.path.join('data', 'distance_based', 'processing', 'corpus')

    for fold in range(N_SETS):
        reader_path = os.path.join(path, 'set_' + str(fold + 1), 'train.csv')

        joiner_path = os.path.join(corpus_save_path, 'set_' + str(fold + 1))
        data = pd.read_csv(reader_path)

        mkdir(joiner_path)
        model = Doc2corpus()

        for index, doc_data in enumerate(data['Tweet']):
            model.create_corpus(doc_data)
            if index % 1000 == 0:
                print('{} fold -  - tweet processing on going processing : {} Remaining : {}/{} . '.
                      format(str(fold+1), str(index), str(index), str(len(data['Tweet']))))
        model.save_corpus(path=os.path.join(joiner_path, 'corpus.txt'))
