import pandas as pd

def check_dir(dir_):
    import os
    if not os.path.isdir(dir_):
        os.makedirs(dir_)

def time_parser(tweets):
    """
    convert to date format and extract hour
    """
    from datetime import datetime
    date_format = "%Y-%m-%dT%H:%M:%S" 
    tweets["time"]   = pd.to_datetime(tweets["time"],format = date_format)
    tweets["hour"]   = pd.DatetimeIndex(tweets["time"]).hour
    tweets["month"]  = pd.DatetimeIndex(tweets["time"]).month
    tweets["day"]    = pd.DatetimeIndex(tweets["time"]).day
    
    return tweets

def tweets_cleaning(df,column):
    """
    function to remove special characters , punctuations ,stop words, digits ,hyperlinks and case conversion
    """
    import  re
    from nltk.corpus import stopwords
    #stop_words = stopwords.words("english")
    #extract hashtags
    df["hashtag"]  = df[column].str.findall(r'#.*?(?=\s|$)')
    #extract twitter account references
    df["accounts"] = df[column].str.findall(r'@.*?(?=\s|$)')
    
    #remove hashtags and accounts from tweets
    df[column] = df[column].str.replace(r'#.*?(?=\s|$)'," ")
    #df[column] = df[column].str.replace(r'@.*?(?=\s|$)'," ")
    
    #convert to lower case
    df[column] = df[column].str.lower()
    #remove hyperlinks
    df[column] = df[column].apply(lambda x: re.match('(.*?)http.*?\s?(.*?)', x).group(1) if re.match('(.*?)http.*?\s?(.*?)', x) else x)
    #remove under scores
    df[column] = df[column].str.replace("_"," ")

    return df

def convert_words_list(df):
    """
    Convert words to lists

    """ 
    words = tweets_cleaning(df,0)
    words_list = words[words[0] != ""][0].tolist()
    return words_list

def scoring_tweets(data_frame,text_column,positive_words_list,negative_words_list) :
    #identifying +ve and -ve words in tweets
    data_frame["positive"] = data_frame[text_column].apply(lambda x:" ".join([i for i in x.split() 
                                                                              if i in (positive_words_list)]))
    data_frame["negative"] = data_frame[text_column].apply(lambda x:" ".join([i for i in x.split()
                                                                              if i in (negative_words_list)]))
    #scoring
    data_frame["positive_count"] = data_frame["positive"].str.split().str.len()
    data_frame["negative_count"] = data_frame["negative"].str.split().str.len()
    data_frame["score"]          = (data_frame["positive_count"] -
                                    data_frame["negative_count"])
    
    #create new feature sentiment :
    #+ve if score is +ve , #-ve if score is -ve , # neutral if score is 0
    def labeling(data_frame) :
        if data_frame["score"]   > 0  :
            return "positive"
        elif data_frame["score"] < 0  :
            return "negative"
        elif data_frame["score"] == 0 :
            return "neutral"
    data_frame["sentiment"] = data_frame.apply(lambda data_frame:labeling(data_frame),
                                               axis = 1)
        
    return data_frame

def time_parser(tweets):
    """
    convert to date format and extract hour
    """
    from datetime import datetime
    date_format = "%Y-%m-%dT%H:%M:%S" 
    tweets["time"]   = pd.to_datetime(tweets["time"],format = date_format)
    tweets["hour"]   = pd.DatetimeIndex(tweets["time"]).hour
    tweets["month"]  = pd.DatetimeIndex(tweets["time"]).month
    tweets["day"]    = pd.DatetimeIndex(tweets["time"]).day
    
    return tweets

def tweets_cleaning(df,column):
    """
    function to remove special characters , punctuations ,stop words, digits ,hyperlinks and case conversion
    """
    import  re
    from nltk.corpus import stopwords
    #stop_words = stopwords.words("english")
    #extract hashtags
    df["hashtag"]  = df[column].str.findall(r'#.*?(?=\s|$)')
    #extract twitter account references
    df["accounts"] = df[column].str.findall(r'@.*?(?=\s|$)')
    
    #remove hashtags and accounts from tweets
    df[column] = df[column].str.replace(r'#.*?(?=\s|$)'," ")
    #df[column] = df[column].str.replace(r'@.*?(?=\s|$)'," ")
    
    #convert to lower case
    df[column] = df[column].str.lower()
    #remove hyperlinks
    df[column] = df[column].apply(lambda x: re.match('(.*?)http.*?\s?(.*?)', x).group(1) if re.match('(.*?)http.*?\s?(.*?)', x) else x)

    #remove under scores
    df[column] = df[column].str.replace("_"," ")

    return df

def to_txt(df, file_path):
    f = open(file_path, "w")
    for ix, row in df.iterrows():
        text = row.text.strip()
        text = text.replace('\n',' ')
        text = text.replace('\r',' ')
        label = row.sentiment
        f.write(text)
        f.write('\n')
        f.write(label)
        f.write('\n')
    f.close()
    print ("writing txt finished")
