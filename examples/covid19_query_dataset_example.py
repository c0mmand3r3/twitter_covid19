import pandas as pd
import os

if __name__ == '__main__':
    data = pd.read_csv(os.path.join('data', 'covid19_tweeter_dataset.csv'))

    data_w = data.query('Label == "0" or Label == "1" or Label == "-1"')

    print(data_w)
    data_w.to_csv('data\\covid19_tweeter_final_dataset.csv')