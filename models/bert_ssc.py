# -*- coding: utf-8 -*-
# file: BERT_SSC.py
# author: yunruili <ray811030@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import torch
import torch.nn as nn


class BERT_SSC(nn.Module):
    """single sentence classification"""
    def __init__(self, bert, opt):
        super(BERT_SSC, self).__init__()
        # self.squeeze_embedding = SqueezeEmbedding()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)

    def forward(self, inputs):
        #text_bert_indices, bert_segments_ids = inputs[0], inputs[1]
        text_bert_indices = inputs[0]
        # text_bert_len = torch.sum(text_bert_indices != 0, dim=-1)
        # text_bert_indices = self.squeeze_embedding(text_bert_indices, text_bert_len)
        # bert_segments_ids = self.squeeze_embedding(bert_segments_ids, text_bert_len)
        _, pooled_output = self.bert(text_bert_indices, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.dense(pooled_output)
        return logits