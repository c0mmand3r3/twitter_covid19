import os
import pickle

import numpy as np
from sklearn import preprocessing
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.saving.save import load_model
from tensorflow.python.keras.utils.np_utils import to_categorical


def read_pickle_data(path=None):
    if path is None or not os.path.isfile(path):
        return None
    with open(path, 'rb') as fid:
        data = pickle.load(fid)
        fid.close()
        return data


SETS = 10

if __name__ == '__main__':
    read_main_path = os.path.join('data', 'fold_train_test_dataset_overall_vectors_for_300dim')
    read_path = os.path.join('data', 'model_save_update')

    data_structure = {
        'sets' : [],
        'f1_score': [],
        'precision_score': [],
        'recall_score': [],
        'accuracy_score': [],
    }
    for n_set in range(SETS):
        read_joiner_path = os.path.join(read_main_path, 'set_' + str(n_set + 1))
        # model = load_model(os.path.join(read_path, 'set_' + str(n_set + 1), str(n_set + 1) + '_300dim_rmsprop.h5'))

        train_x = read_pickle_data(os.path.join(read_joiner_path, 'train_x.pkl'))
        train_y = read_pickle_data(os.path.join(read_joiner_path, 'train_y.pkl'))
        test_x = read_pickle_data(os.path.join(read_joiner_path, 'test_x.pkl'))
        test_y = read_pickle_data(os.path.join(read_joiner_path, 'test_y.pkl'))
        print("---------------- Loading Set {} completed -------------------".format(n_set + 1))

        print(np.shape(train_x), np.shape(train_y))
        print(np.shape(test_x), np.shape(test_y))
        exit(0)
        scale_model = StandardScaler()
        scale_model.fit(train_x)

        train_x = scale_model.transform(train_x)
        test_x = scale_model.transform(test_x)

        print(np.shape(train_x), np.shape(train_y))
        print(np.shape(test_x), np.shape(test_y))

        le = preprocessing.LabelEncoder()
        le.fit(train_y)

        train_y = le.transform(train_y)
        test_y = le.transform(test_y)

        train_y = to_categorical(train_y)
        test_y = to_categorical(test_y)

        train_x = np.array(train_x).reshape((np.shape(train_x)[0], np.shape(train_x)[1], 1))
        test_x = np.array(test_x).reshape((np.shape(test_x)[0], np.shape(test_x)[1], 1))

        print(np.shape(train_x), np.shape(train_y))
        print(np.shape(test_x), np.shape(test_y))

        results = model.evaluate(test_x, test_y, batch_size=64)

        predict = model.predict(test_x)
        predict_ = np.zeros_like(predict)
        predict_[np.arange(len(predict)), predict.argmax(1)] = 1
        data_structure['sets'].append('set_'+str(n_set + 1))
        data_structure['f1_score'].append(f1_score(test_y, predict_, average='weighted'))
        data_structure['precision_score'].append(precision_score(test_y, predict_, average='weighted'))
        data_structure['recall_score'].append(recall_score(test_y, predict_, average='weighted'))
        data_structure['accuracy_score'].append(accuracy_score(test_y, predict_))

    data_structure['sets'].append('average')
    data_structure['f1_score'].append(np.mean(data_structure['f1_score']))
    data_structure['precision_score'].append(np.mean(data_structure['precision_score']))
    data_structure['recall_score'].append(np.mean(data_structure['recall_score']))
    data_structure['accuracy_score'].append(np.mean(data_structure['accuracy_score']))
    import pandas as pd
    df = pd.DataFrame(data_structure)
    df.to_csv('data//result.csv')
