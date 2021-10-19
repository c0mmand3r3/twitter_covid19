"""
 -- author : Anish Basnet
 -- email : anishbasnetworld@gmail.com
 -- date : 11/14/2019
"""

import os
import numpy as np

from tweeter_covid19 import read_pickle_data
from tweeter_covid19.classification import Classification
from tweeter_covid19.utils.modelutils import optimize_model

import os
import pickle

import numpy as np
from sklearn import preprocessing
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import StandardScaler



def read_pickle_data(path=None):
    if path is None or not os.path.isfile(path):
        return None
    with open(path, 'rb') as fid:
        data = pickle.load(fid)
        fid.close()
        return data


SETS = 10

def get_data_split(data):
    x = []
    y = []
    for key, value in data.items():
        target = list(value.keys())[0]
        x.append(value[target])
        y.append(target)
    return x, y


if __name__ == '__main__':
    # train_path = os.path.join('data', 'processing', 'tf-icf', 'vectors', 'fusion_news', 'sets', 'set_1',
    #                           'tf_icf_train.pkl')
    # test_path = os.path.join('data', 'processing', 'tf-icf', 'vectors', 'fusion_news', 'sets', 'set_1',
    #                          'tf_icf_test.pkl')
    #
    # train_data = read_pickle_data(train_path)
    # test_data = read_pickle_data(test_path)
    #
    # train_x, train_y = get_data_split(train_data)
    # test_x, test_y = get_data_split(test_data)
    #
    # print(np.shape(train_x), np.shape(train_y))
    # print(np.shape(test_x), np.shape(test_y))
    #
    # model = Classification()
    # accuracy = optimize_model(model, data=(train_x, train_y, test_x, test_y),
    #                           limit=(1, 100, 10), gamma=1e-05, normalize=True, verbose=True)
    # maximum = max(accuracy, key=lambda item: item[1])
    # print("At C={}, the maximum accuracy is {}.".format(maximum[0], maximum[1]))



    if __name__ == '__main__':
        read_main_path = os.path.join('data', 'tfidf_processing', 'tfidf_vector')
        read_path = os.path.join('data', 'model_save_update')

        data_structure = {
            'sets': [],
            'f1_score': [],
            'precision_score': [],
            'recall_score': [],
            'accuracy_score': [],
        }
        for n_set in range(SETS):
            read_joiner_path = os.path.join(read_main_path, 'set_' + str(n_set + 1))
            model = Classification()
            train_x = read_pickle_data(os.path.join(read_joiner_path, 'train_x.pkl'))
            train_y = read_pickle_data(os.path.join(read_joiner_path, 'train_y.pkl'))
            print(train_y)
            exit(0)
            test_x = read_pickle_data(os.path.join(read_joiner_path, 'test_x.pkl'))
            test_y = read_pickle_data(os.path.join(read_joiner_path, 'test_y.pkl'))
            print("---------------- Loading Set {} completed -------------------".format(n_set + 1))

            print(np.shape(train_x), np.shape(train_y))
            print(np.shape(test_x), np.shape(test_y))
            # exit(0)
        #     model.fit_parameter(c=61, kernel='rbf', gamma=1e-04)
        #     predict = model.fit_predict(train_x, train_y, test_x, test_y, normalize=True)
        #     # maximum = max(accuracy, key=lambda item: item[1])
        #     # print("At C={}, the maximum accuracy is {}.".format(maximum[0], maximum[1]))
        #
        #     # predict = model.predict(test_x)
        #     data_structure['sets'].append('set_' + str(n_set + 1))
        #     data_structure['f1_score'].append(f1_score(test_y, predict, average='weighted'))
        #     data_structure['precision_score'].append(precision_score(test_y, predict, average='weighted'))
        #     data_structure['recall_score'].append(recall_score(test_y, predict, average='weighted'))
        #     data_structure['accuracy_score'].append(accuracy_score(test_y, predict))
        #     print("successfully completed!")
        #
        # data_structure['sets'].append('average')
        # data_structure['f1_score'].append(np.mean(data_structure['f1_score']))
        # data_structure['precision_score'].append(np.mean(data_structure['precision_score']))
        # data_structure['recall_score'].append(np.mean(data_structure['recall_score']))
        # data_structure['accuracy_score'].append(np.mean(data_structure['accuracy_score']))
        # import pandas as pd
        #
        # df = pd.DataFrame(data_structure)
        # df.to_csv('data//result.csv')
