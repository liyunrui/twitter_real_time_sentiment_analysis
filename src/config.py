import sys
sys.path.append("/Users/yunruili/twitter_real_time_sentiment_analysis/src")
from models.bert_ssc import BERT_SSC
import torch

class Option(object): 
    pass
class Configs:
    input_colses = {
        'bert_ssc': ['text_raw_bert_indices'],

    }
    # set your trained models here
    model_state_dict_paths = {
        'bert_ssc':'/Users/yunruili/twitter_post/state_dict/bert_ssc_twitter_val_acc0.8849',
    }
    opt = Option()
    opt.model_name = 'bert_ssc'
    # model path
    opt.state_dict_path =  model_state_dict_paths[opt.model_name]
    opt.max_seq_len = 140
    opt.pretrained_bert_name = "bert-base-uncased"
    opt.dropout = 0.1
    opt.bert_dim = 768
    opt.max_seq_len = 140
    opt.polarities_dim = 3
    opt.inputs_cols = input_colses[opt.model_name]
    opt.hops = 3
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_classes = {
        'bert_ssc':BERT_SSC,
    }
    opt.model_class = model_classes[opt.model_name]
    input_colses = {
        'bert_ssc': ['text_raw_bert_indices'],

    }
    opt.inputs_cols = input_colses[opt.model_name]

    polarity_dict ={
            2:"postive",
            1:"neutral",
            0:"negative"
    }