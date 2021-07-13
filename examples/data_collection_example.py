"""
 - Author : Anish Basnet
 - Date : Tuesay, July 13, 2021
 This is to collect the different csv file.
"""
import math
import os

import numpy as np
import pandas as pd

from tweeter_covid19.utils import flatten

if __name__ == '__main__':
    first_path = os.path.join("data", "covid_1.csv")
    second_path = os.path.join("data", "covid_2.csv")

    write_path = os.path.join('data', 'covid19_tweeter_dataset.csv')

    first_dataframe = pd.read_csv(first_path)
    second_dataframe = pd.read_csv(second_path, error_bad_lines=False)

    new_dataset = {
        'Label': [],
        'Datetime': [],
        'Tweet': [],
        'Tokanize_tweet': [],
    }

    for row in first_dataframe.iterrows():
        if row[0] < 26000:
            new_dataset['Datetime'].append(row[1][1])
            new_dataset['Tweet'].append(row[1][2])
            new_dataset['Tokanize_tweet'].append(row[1][3])
            for n_index, val in enumerate(row[1][4:]):
                status = False
                if not pd.isna(val):
                    if val == 'positive':
                        new_dataset['Label'].append(1)
                    elif val == 'negative':
                        new_dataset['Label'].append(-1)
                    elif val == 'neutral':
                        new_dataset['Label'].append(0)
                    else:
                        new_dataset['Label'].append(val)
                    status = True
                    break
                if not status and n_index == 42:
                    print(row[0])

    new_dataset['Label'].append(list(second_dataframe['Label']))
    new_dataset['Datetime'].append(list(second_dataframe['Datetime']))
    new_dataset['Tweet'].append(list(second_dataframe['Tweets']))
    new_dataset['Tokanize_tweet'].append(list(second_dataframe['Non_stop_Tokanize_tweets']))

    new_dataset['Label'] = flatten(new_dataset['Label'])
    new_dataset['Datetime'] = flatten(new_dataset['Datetime'])
    new_dataset['Tweet'] = flatten(new_dataset['Tweet'])
    new_dataset['Tokanize_tweet'] = flatten(new_dataset['Tokanize_tweet'])

    write_df = pd.DataFrame(new_dataset)

    write_df.to_csv(write_path)
    print("Succefully collected Data")

