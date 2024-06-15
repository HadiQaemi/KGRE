import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from .attention import Attention

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
        self.attention_block = Attention(768, 24, 0.5)
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, 18)

    def forward(self, input_ids, attn_mask, token_type_ids):
        outputs = self.bert_model(
            input_ids,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids
        )

        pooled_output = outputs.pooler_output
        last_hidden_state = outputs.last_hidden_state

        attention = self.attention_block(last_hidden_state, last_hidden_state, last_hidden_state)
        attention_mean = attention.mean(axis=1)
        cls = last_hidden_state[:, 0, :]

        output_dropout = self.dropout(attention_mean)
        output = self.linear(output_dropout)
        outputs = {
            'predicts': output,
            'cls_feats': cls,
            'mean_feats': attention_mean
        }
        return outputs