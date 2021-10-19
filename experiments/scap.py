import os

import pandas as pd
if __name__ == '__main__':
    read_path = os.path.join('data', 'covid_tweets_clean.csv')

    df = pd.read_csv(read_path)
    print(len(set(df['Clean_text'])))