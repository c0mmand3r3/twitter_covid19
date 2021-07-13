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

    for fold in range(TOTAL_SET):
        joiner_path = os.path.join(write_path, 'set_' + str(fold + 1))
        mkdir(joiner_path)
        train_data = data.sample(frac=0.7)
        test_data = data.drop(train_data.index)

        train_data.to_csv(os.path.join(joiner_path, 'train.csv'))
        test_data.to_csv(os.path.join(joiner_path, 'test.csv'))

        print('FOLD - {} // Successfully Created ! Train tweets - {} :: Test tweets - {} .'.
              format(fold + 1, train_data.shape[0], test_data.shape[0]))
