"""
 -- author : Anish Basnet
 -- email : anishbasnetworld@gmail.com
 -- date : March 18, 2020
"""


import os

import pandas as pd

from tweeter_covid19.scrapping import Scrapping
if __name__ == '__main__':
    url = 'https://twitter.com/search?q=%23%E0%A4%95%E0%A5%8B%E0%A4%AD%E0%A4%BF%E0%A4%A1%E2%80%93%E0%A5%A7%E0%A5%AF&src=typed_query&f=live'
    write_path = os.path.join('data', 'covid19_twitter.csv')
    sub_scrap = Scrapping()
    sub_scrap.set_url(url)
    print(sub_scrap.get_page(pause_time=2, activate_scroll_down=True, running_hrs=4, tweet_gap_in_sec=60))
    contents = sub_scrap.sub_get_content()
    sub_scrap.close_driver()
    df = pd.DataFrame(contents)
    df.to_csv(write_path)
    print("sucessfully generated")
