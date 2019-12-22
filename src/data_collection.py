"""
Get the latstes real_time tweet data from twitter

python3 data_collection.py --url https://twitter.com/realDonaldTrump --real_time True

"""
import pandas as pd
import argparse
import os
from twitter_scraper import get_tweets

def get_username(url):
    """
    url:str
    
    url's format looks like "https://twitter.com/realDonaldTrump/xxx"
    
    """
    user_name = url.split("/")[3]
    return user_name

 
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='https://twitter.com/realDonaldTrump', type=str, required = True)
    parser.add_argument('--data_path', default="../dataset/raw_data", type=str)
    parser.add_argument('--real_time', default=False, type=bool)
    parser.add_argument('--num_pages', default=1, type=int)

    opt = parser.parse_args()
    if opt.real_time == False:
        twitter_url = opt.url
        user_name = get_username(twitter_url)
        list_of_tweets = get_tweets(user_name, pages=50000)
        columns = [
         'tweetId',
         'isRetweet',
         'time',
         'text',
         'replies',
         'retweets',
         'likes',
         'entries']
        data = {c:[] for c in columns}
        for tweet in list_of_tweets:
            for col in columns:
                data[col].append(tweet[col])

        df = pd.DataFrame(data)
        df.to_csv(os.path.join(opt.data_path, "{}.csv".format(user_name)), index = False)
    else:
        twitter_url = opt.url
        user_name = get_username(twitter_url)
        list_of_tweets = get_tweets(user_name, pages=opt.num_pages)
        columns = [
         'tweetId',
         'isRetweet',
         'time',
         'text',
         'replies',
         'retweets',
         'likes',
         'entries']
        data = {c:[] for c in columns}
        for tweet in list_of_tweets:
            for col in columns:
                data[col].append(tweet[col])

        df = pd.DataFrame(data)
        df.to_csv(os.path.join(opt.data_path, "{}_real_time.csv".format(user_name)), index = False)
