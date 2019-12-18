import os
import argparse
import pandas as pd
from utils import convert_words_list,time_parser,tweets_cleaning,scoring_tweets,check_dir,to_txt

dataset_files = {
    'twitter':{
        "train":"../dataset/twitter/twitter_train.txt",
        "test":"../dataset/twitter/twitter_test.txt"

    },
}

pd.options.display.max_colwidth = 5000
pd.options.display.max_rows = 100


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default="../dataset/raw_data", type=str)
    parser.add_argument('--output_path', default="../dataset/tagged_data", type=str)

    opt = parser.parse_args()

    data_dir = opt.input_path
    out = []
    for file_name in os.listdir(data_dir):
        user_name = file_name[:-4]
        file_path = os.path.join(data_dir,file_name)
        df = pd.read_csv(file_path)
        df["user_name"] = [user_name for i in range(len(df))]
        out.append(df)
        #print (user_name)
        #print (file_path)
    df = pd.concat(out, axis = 0)
    print ("num of raw data we crawled for model training", len(df))

    #------------------------
    # data processing
    #------------------------
    # parse time
    df = time_parser(df)
    # text cleaning
    df = tweets_cleaning(df,"text")

    #------------------------
    # data tagging
    #------------------------

    #positive words
    positive_words = pd.read_csv("../asset/positive-words.txt",
                                 header=None)
    #negative words
    negative_words = pd.read_csv("../asset/negative-words.txt",
                                 header=None,encoding='latin-1')

    positive_words_list = convert_words_list(positive_words)

    #remove word trump from positive word list
    positive_words_list = [i for i in positive_words_list if i not in "trump"]
    negative_words_list = convert_words_list(negative_words)
    df = scoring_tweets(df,"text",positive_words_list,negative_words_list)
    print ("Tagging Finished !")
    # save
    check_dir(opt.output_path)
    df.to_csv(os.path.join(opt.output_path, "{}.csv".format("tagging")), index = False)
    #------------------------
    # exp
    #------------------------
    df = pd.read_csv(os.path.join(opt.output_path, "{}.csv".format("tagging")))
    df.dropna(subset = ["text"], inplace = True)
    print ("num_data : ", len(df))
    df_train = df.sample(frac = 0.8)
    print ("num_training_data : ", len(df_train))
    df_test = df[~df.index.isin(df_train.index)]
    print ("num_testing_data : ", len(df_test))
    assert len(df_train)+len(df_test)==len(df), "it should be the same."

    to_txt(df_train,dataset_files["twitter"]["train"])
    to_txt(df_test,dataset_files["twitter"]["test"])


