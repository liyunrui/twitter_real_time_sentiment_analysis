import torch
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert import BertModel
from torch.utils.data import Dataset
import pandas as pd



class Tokenizer4Bert:
    def __init__(self, max_seq_len, pretrained_bert_name):
        # Load pretrained model/tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return self.pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)

    def pad_and_truncate(self, sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
        import numpy as np
        x = (np.ones(maxlen) * value).astype(dtype)
        if truncating == 'pre':
            trunc = sequence[-maxlen:]
        else:
            trunc = sequence[:maxlen]
        trunc = np.asarray(trunc, dtype=dtype)
        if padding == 'post':
            x[:len(trunc)] = trunc
        else:
            x[-len(trunc):] = trunc
        return x

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class ABSADataset(Dataset):
    def __init__(self, fname, tokenizer):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()

        all_data = []
        for i in range(0, len(lines), 2):
            text = lines[i].strip()
            polarity = lines[i + 1].strip()
            # single-sentence classification
            text_raw_bert_indices = tokenizer.text_to_sequence("[CLS] " + text)
            # label
            polarity_dict ={
                2:"positive",
                1:"neutral",
                0:"negative"
            }
            polarity_dict_inv = {s:i for i,s in polarity_dict.items()}
            polarity_dict_inv
            #print (polarity)
            polarity = polarity_dict_inv[polarity]

            data = {
                'text_raw_bert_indices': text_raw_bert_indices,
                'polarity': polarity,
            }
            all_data.append(data)
            
        self.data = all_data

    def get_dataframe(self, tokenizer):
        """
        Conver dataset into DataFrame(Pandas)
        It's only support for bert-based model.
        """
        df = []
        columns_name = []
        for i in range(len(self.data)):
            tmp = []
            for k, v in self.data[i].items():
                try:
                    to_str = " ".join(tokenizer.tokenizer.convert_ids_to_tokens(v))
                    tmp.append(to_str)
                except:
                    if k == 'aspect_in_text':
                        # it's a 1-D tensor wtih shape of (2,), representing the start and end index of the aspect
                        v = v.numpy()  # 1-D tensor
                        #print (v.shape)
                    tmp.append(v)
                if i <= 0:
                    columns_name.append(k)
            df.append(tmp)
        df = pd.DataFrame(df,columns=columns_name)   
        return df

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)