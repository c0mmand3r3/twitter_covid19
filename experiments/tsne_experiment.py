"""
 -- author : Anish Basnet
 -- email : anishbasnetworld@gmail.com
 -- date : 11/14/2019
"""

import os

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE

from tweeter_covid19 import read_pickle_data


def get_data_split(data):
    x = []
    y = []
    for key, value in data.items():
        target = list(value.keys())[0]
        x.append(value[target])
        y.append(target)
    return x, y


if __name__ == '__main__':

    read_main_path = os.path.join('data', 'fold_train_test_collector_with_normalization')

    for n_set in range(10):
        print("---------------- Loading Set {} data sets -------------------".format(n_set + 1))
        read_joiner_path = os.path.join(read_main_path, 'set_' + str(n_set + 1))

        train_x = read_pickle_data(os.path.join(read_joiner_path, 'train_x.pkl'))
        train_y = read_pickle_data(os.path.join(read_joiner_path, 'train_y.pkl'))
        test_x = read_pickle_data(os.path.join(read_joiner_path, 'test_x.pkl'))
        test_y = read_pickle_data(os.path.join(read_joiner_path, 'test_y.pkl'))
        print("---------------- Loading Set {} completed -------------------".format(n_set + 1))

        print(np.shape(train_x), np.shape(train_y))
        print(np.shape(test_x), np.shape(test_y))

        train_x = np.average(train_x, axis=1)
        test_x = np.average(test_x, axis=1)

        print(np.shape(train_x), np.shape(train_y))
        print(np.shape(test_x), np.shape(test_y))
        palette_colors = len(set(train_y))

        tsne = TSNE(n_components=2, random_state=0)
        x_2d = tsne.fit_transform(test_x)

        tsne_df = pd.DataFrame({'X': x_2d[:, 0],
                                'Y': x_2d[:, 1],
                                'Classes': test_y})
        tsne_df.head()
        fig = sns.scatterplot(x=x_2d[:, 0], y=x_2d[:, 1],
                              hue=test_y,
                              palette=sns.color_palette("Set1", n_colors=palette_colors, desat=.5),
                              legend=False)
        fig.figure.savefig('a.png')
        exit(0)
