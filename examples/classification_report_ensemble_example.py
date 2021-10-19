"""
 - Author : Anish Basnet
 - Email : anishbasnetworld@gmail.com
 - Date : Monday, August 9, 2021
"""
import os

from sklearn import preprocessing
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tensorflow.python.keras.saving.save import load_model
from tensorflow.python.keras.utils.np_utils import to_categorical


def mkdir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except NotADirectoryError:
            pass


def get_data_split(data):
    x = []
    y = []
    for key, value in data.items():
        target = list(value.keys())[0]
        x.append(value[target])
        y.append(target)
    return x, y


def read_pickle_data(path=None):
    import pickle
    with open(path, 'rb') as fid:
        data = pickle.load(fid)
        fid.close()
        return data


from sklearn.preprocessing import StandardScaler
import numpy as np

N_SETS = 10

if __name__ == '__main__':
    model_path = os.path.join('data', 'model_save_update')

    read_main_path_3D = os.path.join('data', 'fold_train_test_dataset_overall_vectors_3dim')
    read_main_path_17D = os.path.join('data', 'fold_train_test_dataset_overall_vectors')
    # read_main_path_300D = os.path.join('data', 'fold_train_test_dataset_overall_vectors_3dim')
    read_main_path_300D = os.path.join('data', 'fold_train_test_dataset_overall_vectors_for_300dim')

    model_write_path = os.path.join('data', 'model_save_ensemble_update')

    data_dict = dict()
    accuracy = []

    f_scores = []
    p_scores = []
    a_scores = []
    r_scores = []
    for n_set in range(N_SETS):
        print("---------------- Loading Set {} data sets -------------------".format(n_set + 1))
        model_path_joiner = os.path.join(model_path, 'set_' + str(n_set + 1))

        read_joiner_path_3D = os.path.join(read_main_path_3D, 'set_' + str(n_set + 1))
        read_joiner_path_17D = os.path.join(read_main_path_17D, 'set_' + str(n_set + 1))
        read_joiner_path_300D = os.path.join(read_main_path_300D, 'set_' + str(n_set + 1))

        writer_joiner_path = os.path.join(model_write_path, 'set_' + str(n_set + 1))

        mkdir(writer_joiner_path)

        train_x_3D = read_pickle_data(os.path.join(read_joiner_path_3D, 'train_x.pkl'))
        train_y_3D = read_pickle_data(os.path.join(read_joiner_path_3D, 'train_y.pkl'))
        test_x_3D = read_pickle_data(os.path.join(read_joiner_path_3D, 'test_x.pkl'))
        test_y_3D = read_pickle_data(os.path.join(read_joiner_path_3D, 'test_y.pkl'))

        train_x_17D = read_pickle_data(os.path.join(read_joiner_path_17D, 'train_x.pkl'))
        train_y_17D = read_pickle_data(os.path.join(read_joiner_path_17D, 'train_y.pkl'))
        test_x_17D = read_pickle_data(os.path.join(read_joiner_path_17D, 'test_x.pkl'))
        test_y_17D = read_pickle_data(os.path.join(read_joiner_path_17D, 'test_y.pkl'))

        train_x_300D = read_pickle_data(os.path.join(read_joiner_path_300D, 'train_x.pkl'))
        train_y_300D = read_pickle_data(os.path.join(read_joiner_path_300D, 'train_y.pkl'))
        test_x_300D = read_pickle_data(os.path.join(read_joiner_path_300D, 'test_x.pkl'))
        test_y_300D = read_pickle_data(os.path.join(read_joiner_path_300D, 'test_y.pkl'))

        print("---------------- Loading Set {} completed -------------------".format(n_set + 1))

        print(np.shape(train_x_3D), np.shape(train_y_3D))
        print(np.shape(test_x_3D), np.shape(test_y_3D))
        print(np.shape(train_x_17D), np.shape(train_y_17D))
        print(np.shape(test_x_17D), np.shape(test_y_17D))
        print(np.shape(train_x_300D), np.shape(train_y_300D))
        print(np.shape(test_x_300D), np.shape(test_y_300D))
        scale_model = StandardScaler()
        scale_model.fit(train_x_3D)

        train_x_3D = scale_model.transform(train_x_3D)
        test_x_3D = scale_model.transform(test_x_3D)

        print(np.shape(train_x_3D), np.shape(train_y_3D))
        print(np.shape(test_x_3D), np.shape(test_y_3D))

        scale_model_17D = StandardScaler()
        scale_model_17D.fit(train_x_17D)

        train_x_17D = scale_model_17D.transform(train_x_17D)
        test_x_17D = scale_model_17D.transform(test_x_17D)

        print(np.shape(train_x_17D), np.shape(train_y_17D))
        print(np.shape(test_x_17D), np.shape(test_y_17D))

        scale_model_300D = StandardScaler()
        scale_model_300D.fit(train_x_300D)

        train_x_300D = scale_model_300D.transform(train_x_300D)
        test_x_300D = scale_model_300D.transform(test_x_300D)

        print(np.shape(train_x_300D), np.shape(train_y_300D))
        print(np.shape(test_x_300D), np.shape(test_y_300D))

        le = preprocessing.LabelEncoder()
        le.fit(train_y_3D)

        train_y_3D = le.transform(train_y_3D)
        test_y_3D = le.transform(test_y_3D)

        train_y_3D = to_categorical(train_y_3D)
        test_y_3D = to_categorical(test_y_3D)

        train_x_3D = np.array(train_x_3D).reshape((np.shape(train_x_3D)[0], np.shape(train_x_3D)[1], 1))
        test_x_3D = np.array(test_x_3D).reshape((np.shape(test_x_3D)[0], np.shape(test_x_3D)[1], 1))

        print(np.shape(train_x_3D), np.shape(train_y_3D))
        print(np.shape(test_x_3D), np.shape(test_y_3D))

        le_17 = preprocessing.LabelEncoder()
        le_17.fit(train_y_17D)

        train_y_17D = le_17.transform(train_y_17D)
        test_y_17D = le_17.transform(test_y_17D)

        train_y_17D = to_categorical(train_y_17D)
        test_y_17D = to_categorical(test_y_17D)

        train_x_17D = np.array(train_x_17D).reshape((np.shape(train_x_17D)[0], np.shape(train_x_17D)[1], 1))
        test_x_17D = np.array(test_x_17D).reshape((np.shape(test_x_17D)[0], np.shape(test_x_17D)[1], 1))

        print(np.shape(train_x_17D), np.shape(train_y_17D))
        print(np.shape(test_x_17D), np.shape(test_y_17D))

        le_300D = preprocessing.LabelEncoder()
        le_300D.fit(train_y_300D)

        train_y_300D = le_300D.transform(train_y_300D)
        test_y_300D = le_300D.transform(test_y_300D)

        train_y_300D = to_categorical(train_y_300D)
        test_y_300D = to_categorical(test_y_300D)

        train_x_300D = np.array(train_x_300D).reshape((np.shape(train_x_300D)[0], np.shape(train_x_300D)[1], 1))
        test_x_300D = np.array(test_x_300D).reshape((np.shape(test_x_300D)[0], np.shape(test_x_300D)[1], 1))

        print(np.shape(train_x_300D), np.shape(train_y_300D))
        print(np.shape(test_x_300D), np.shape(test_y_300D))

        modelEns = load_model(
            os.path.join('data', 'model_save_ensemble_update', 'set_' + str(n_set + 1),
                         str(n_set + 1) + '_mean_ensemble.h5'))
        print(modelEns.summary())

        predict = modelEns.predict([test_x_300D, test_x_17D, test_x_3D])
        predict_ = np.zeros_like(predict)
        predict_[np.arange(len(predict)), predict.argmax(1)] = 1

        f_scores.append(f1_score(test_y_3D, predict_, average=None))
        p_scores.append(precision_score(test_y_3D, predict_, average=None))
        r_scores.append(recall_score(test_y_3D, predict_, average=None))
        a_scores.append(accuracy_score(test_y_3D, predict_))

    print('P-Score : ', np.mean(p_scores, axis=0))
    print('R-Score : ', np.mean(r_scores, axis=0))
    print('F-score : ', np.mean(f_scores, axis=0))
    print('Accuracy : ', np.mean(a_scores))
