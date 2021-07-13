"""
 - Author : Anish Basnet
 - Email : anishbasnetworld@gmail.com
 - Date : Tuesday, July 13, 2021
"""

import os
import pandas as pd

from tweeter_covid19.utils import mkdir

TOTAL_SET = 10

if __name__ == '__main__':
    read_path = os.path.join('data', 'original', 'covid19_tweeter_final_dataset.csv')
    write_path = os.path.join('data', 'fold_dataset')

    data = pd.read_csv(read_path)

    positive_label_data = data.query('Label == 1')
    negative_label_data = data.query('Label == -1')
    neutral_label_data = data.query('Label == 0')

    for fold in range(TOTAL_SET):
        joiner_path = os.path.join(write_path, 'set_' + str(fold + 1))
        mkdir(joiner_path)

        # positive split
        pos_train_data = positive_label_data.sample(frac=0.7)
        post_test_data = positive_label_data.drop(pos_train_data.index)

        # negative split
        neg_train_data = negative_label_data.sample(frac=0.7)
        neg_test_data = negative_label_data.drop(neg_train_data.index)

        # neutral split
        neu_train_data = neutral_label_data.sample(frac=0.7)
        neu_test_data = neutral_label_data.drop(neu_train_data.index)

        train_data = [pos_train_data, neg_train_data, neu_train_data]
        test_data = [post_test_data, neg_test_data, neu_test_data]

        train_df = pd.concat(train_data)
        test_df = pd.concat(test_data)

        train_df.to_csv(os.path.join(joiner_path, 'train.csv'))
        test_df.to_csv(os.path.join(joiner_path, 'test.csv'))

        print('FOLD - {} // Successfully Created ! Train tweets - {} :: Test tweets - {} .'.
              format(fold + 1, train_df.shape[0], test_df.shape[0]))
