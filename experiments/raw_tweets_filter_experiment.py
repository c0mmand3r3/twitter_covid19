"""
- author: Ashish Mainali
- email: ashishmainalee@gmail.com
- date: 2021-01-11
"""

import os
import pandas as pd
import re
import string


def filter_text(sentence):
    string.punctuation
    '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    sentence.strip(string.punctuation)
    cleaned_fullstop = re.sub(r'[।ः|०-९]', '', str(sentence))
    # clean_text = re.sub(r'[^\w\s]', '', str(cleaned_fullstop))
    return ' '.join(re.findall(r'[\u0900-\u097F]+', str(cleaned_fullstop), re.IGNORECASE))


if __name__ == '__main__':
    read_path = os.path.join('data', 'twitter_datasets', 'raw_data', 'covid_tweets_raw.csv')
    write_path = os.path.join('data', 'twitter_datasets', 'raw_data', 'covid_tweets_clean.csv')

    df = pd.read_csv(read_path)
    clean_tweets = dict({
        'Datetime': [],
        'Clean_text': [],
    })
    for index, content in enumerate(df['Text']):
        print(df['Datetime'][index], content)
        filter_content = filter_text(content)
        clean_tweets['Datetime'].append(df['Datetime'][index])
        clean_tweets['Clean_text'].append(filter_content)
    df_tweets = pd.DataFrame(clean_tweets)
    df_tweets.to_csv(write_path)
