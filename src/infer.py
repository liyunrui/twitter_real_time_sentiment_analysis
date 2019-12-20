from pytorch_pretrained_bert import BertModel
from data_utils import Tokenizer4Bert
from models.bert_ssc import BERT_SSC
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

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
            self.model.load_state_dict(torch.load(opt.state_dict_path))

        # switch model to evaluation mode
        self.model.eval()
        torch.autograd.set_grad_enabled(False)

    def evaluate(self, raw_texts, input_cols):
        """
        paras:
            raw_texts: list of string
            input_cols: list
            aspects: list of string
        we need to add arguments called Aspects. [Now, we only support the below 4 aspects if using this argument]
            1.Ljud
            2.Komfort
            3.Product
            4.Batteri
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

# setting
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

input_colses = {
    'bert_ssc': ['text_raw_bert_indices'],

}
# set your trained models here
model_state_dict_paths = {
    'bert_ssc':'../state_dict/bert_ssc_twitter_val_acc0.8849',
}

opt = AttrDict()
opt.model_name = 'bert_ssc'
opt["max_seq_len"] = 140
opt["pretrained_bert_name"] = "bert-base-uncased"
opt.dropout = 0.1
opt.bert_dim = 768
opt.state_dict_path = model_state_dict_paths[opt.model_name]
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
# infer
inf = Inferer(opt)

# save 
t_probs = inf.evaluate(
    ["it's a good policy"],
    input_cols = opt.inputs_cols
    )
print (t_probs)
polarity_dict ={
        2:"postive",
        1:"neutral",
        0:"negative"
}
prediction = t_probs.argmax(axis=-1) 
print ([polarity_dict[p] for p in prediction])



