"""
 - Author : Anish Basnet
 - Email : anishbasnetworld@gmail.com
 - Date : Tuesday, July 13, 2021
"""
import os
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

if __name__ == '__main__':
    read_path = os.path.join('data', 'original', 'covid19_tweeter_final_dataset.csv')

    data = pd.read_csv(read_path)

    positive_label_data = data.query('Label == 1')
    negative_label_data = data.query('Label == -1')
    neutral_label_data = data.query('Label == 0')

    text = ' '.join(list(neutral_label_data['Tweet']))
    file = open('data/a.txt', mode='w', encoding='utf-8')
    file.write(text)
    file.close()
    # print(text)
    # wordcloud = WordCloud(
    #     collocations=False,
    #     normalize_plurals=False,
    #     max_words=1000,
    #     font_path=os.path.join('data', 'lohit-Devanagari.ttf'),
    #     background_color='white',
    #     regexp=r"[\u0900-\u097F]+",
    #     max_font_size=50).generate(text)
    #
    # plt.figure(figsize=(8, 8), facecolor=None)
    # plt.imshow(wordcloud)
    # plt.axis("off")
    # plt.tight_layout(pad=0)
    # plt.savefig('data/wordcloud.png', dpi=300)
    # plt.show()
