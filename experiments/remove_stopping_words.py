"""
- author: Ashish Mainali
- email: ashishmainalee@gmail.com
- date: 2021-01-13
"""

import os
import pandas as pd

from tweeter_covid19.utils import read_file

if __name__ == '__main__':
    stopping_words_path = os.path.join('resources', 'additional_stop_words.txt')
    read_path = os.path.join('data', 'twitter_datasets', 'raw_data', 'covid_tweets_no_duplicate.csv')
    write_path = os.path.join('data', 'twitter_datasets', 'raw_data', 'covid_tweets_no_stopping_words.csv')

    dataset = read_file(stopping_words_path)
    tweets = pd.read_csv(read_path)

    data_df = dict({
        'Datetime': [],
        'Tweets': [],
        'Non_stop_Tokanize_tweets': []
    })
    # print(tweets['Datetime'][0])
    print(data_df['Datetime'])
    # exit(0)
    for index, _tweets in enumerate(tweets['Tokanize_tweets']):
        clean_tweets = []
        for _index, token in enumerate(_tweets.split(',')):
            if not token in dataset:
                clean_tweets.append(token)
        data_df['Datetime'].append(tweets['Datetime'][index])
        data_df['Tweets'].append(tweets['Tweets'][index])
        data_df['Non_stop_Tokanize_tweets'].append(','.join(clean_tweets))

    my_df = pd.DataFrame(data_df)
    my_df.to_csv(write_path)

    print("completed")