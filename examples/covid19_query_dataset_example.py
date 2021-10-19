import pandas as pd
import os
from tweeter_covid19.utils.utils import flatten

if __name__ == '__main__':
    raw_data = pd.read_csv(os.path.join('data', 'original', 'covid19_tweeter_dataset.csv'))
    clean_data = pd.read_csv(os.path.join('data', 'original', 'covid19_tweets_refactor.csv'))

    raw_tweets = raw_data.query('Label == "1"')
    clean_tweets = clean_data.query('Label == 1')
    raw_tokens = [tweet.split(' ') for tweet in raw_tweets['Tweet']]

    clean_tweets_ = [tweet.split(' ') for tweet in clean_tweets['Tweet'] if len(tweet.split(' ')) > 2]

    # raw_tokens = [tweet.split(' ') for tweet in raw_tweets['Tweet']]
    # clean_tokens = [tweet.split(' ') for tweet in clean_tweets['Tweet'] if len(tweet.split(' ')) > 2]
    print('Total Number of raw tweets : ', len(raw_tweets))
    print('Clean Total number of tweets : ', len(clean_tweets_))
    print('Total Number of tokens in raw : ', len(flatten(raw_tokens)))
    print('Total Number of tokens in Clean : ', len(flatten(clean_tweets_)))
