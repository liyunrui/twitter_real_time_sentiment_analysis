"""
python3 infer.py --input_path ../output/realDonaldTrump_processed_data.csv --output_path ../output/

"""
from pytorch_pretrained_bert import BertModel
from data_utils import Tokenizer4Bert
from models.bert_ssc import BERT_SSC
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import os
import argparse
import warnings
from lime_explainer import explainer
warnings.simplefilter("ignore", UserWarning)
from tqdm import tqdm 

class Inferer:
    """A simple inference example"""
    def __init__(self, opt):
        self.opt = opt
        if 'bert' in opt.model_name:
            self.tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
            bert = BertModel.from_pretrained(opt.pretrained_bert_name)
            self.model = opt.model_class(bert, opt).to(opt.device)
            print('loading model {0} ...'.format(opt.model_name))
            # remember removed map_location='cpu' when using on server w GPU
            if opt.device == "cuda":
                self.model.load_state_dict(torch.load(opt.state_dict_path))
            else:
                self.model.load_state_dict(torch.load(opt.state_dict_path, map_location='cpu'))

        # switch model to evaluation mode
        self.model.eval()
        torch.autograd.set_grad_enabled(False)

    def evaluate(self, raw_texts, input_cols):
        """
        Paras:
            raw_texts: list of string
            input_cols: list
         
        Return list of probabilities of predicted class from the BERT model, list of list of float.        
        """
        if input_cols == ['text_raw_bert_indices']:
            #-----------------------
            # bert_ssc
            #-----------------------
            # text-preprocessing
            texts = [self.text_preprocessing(raw_text) for raw_text in raw_texts]
            # tokenize
            text_bert_indices = []
            for text in texts:
                tokens = self.tokenizer.text_to_sequence("[CLS] " + text)
                text_bert_indices.append(tokens)
            # conver to tensor
            text_bert_indices = torch.tensor(text_bert_indices, dtype=torch.int64).to(self.opt.device)


            t_inputs = [text_bert_indices]
            t_outputs = self.model(t_inputs)

            t_probs = F.softmax(t_outputs, dim=-1).cpu().detach().numpy()
            return t_probs
        else:
            raise ValueError('Invalid input_cols')

    @staticmethod
    def text_preprocessing(text):
        #text = text.strip(" ")
        #text = text.replace("\n", " ")
        return text

if __name__ == '__main__':

    # # setting
    # class AttrDict(dict):
    #     def __init__(self, *args, **kwargs):
    #         super(AttrDict, self).__init__(*args, **kwargs)
    #         self.__dict__ = self

    # input_colses = {
    #     'bert_ssc': ['text_raw_bert_indices'],

    # }
    # # set your trained models here
    # model_state_dict_paths = {
    #     'bert_ssc':'/Users/yunruili/twitter_post/state_dict/bert_ssc_twitter_val_acc0.8849',
    # }

    # opt = AttrDict()
    # opt.model_name = 'bert_ssc'
    # opt["max_seq_len"] = 140
    # opt["pretrained_bert_name"] = "bert-base-uncased"
    # opt.dropout = 0.1
    # opt.bert_dim = 768
    # opt.state_dict_path = model_state_dict_paths[opt.model_name]
    # opt.max_seq_len = 140
    # opt.polarities_dim = 3
    # opt.inputs_cols = input_colses[opt.model_name]
    # opt.hops = 3
    # opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model_classes = {
    #     'bert_ssc':BERT_SSC,
    # }
    # opt.model_class = model_classes[opt.model_name]
    # input_colses = {
    #     'bert_ssc': ['text_raw_bert_indices'],

    # }
    # opt.inputs_cols = input_colses[opt.model_name]
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default="../dataset/raw_data", type=str)
    parser.add_argument('--output_path', default="../dataset/tagged_data", type=str)

    opt = parser.parse_args()

    data_dir = opt.input_path
    # load data
    df = pd.read_csv(data_dir)
    df.text = df.text.astype(str)
    from config import Configs
    # infer
    inf = Inferer(Configs.opt)
    
    # save 
    t_probs = inf.evaluate(
        df.text.tolist(),
        input_cols = Configs.opt.inputs_cols
        )
    print (t_probs)
    polarity_dict ={
            2:"postive",
            1:"neutral",
            0:"negative"
    }
    prediction = t_probs.argmax(axis=-1) 
    prediction = [polarity_dict[p] for p in prediction]
    print (prediction)
    df["pred"] = prediction
    user_name = data_dir.split("/")[-1][:-19]

    keywords = []
    for text in tqdm(df.text.tolist()):
        exp = explainer("BERT",
                        text=text,
                        num_samples=5,
                        num_classes=["negative","neutral","positive"]
                        )
        print (exp.local_exp.keys())
        keywords.append(exp.as_list(label = list(exp.local_exp.keys())[0]))
    df["keywords"] = keywords
    df.to_csv(os.path.join(opt.output_path, "{}.csv".format("{}_processed_data".format(user_name))), index = False)




