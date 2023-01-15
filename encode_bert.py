import torch
import torch.nn as nn
from transformers import BertTokenizer,BertModel
# import torch.nn.functional as F

class BERTSentenceEncoder(nn.Module):
    def __init__(self, config,ckptpath=None):
        nn.Module.__init__(self)
        if ckptpath != None:
            ckpt = torch.load(ckptpath)
            self.bert = BertModel.from_pretrained(config["pretrained_model"],state_dict=ckpt["bert-base"])
        else:
            self.bert = BertModel.from_pretrained(config["pretrained_model"])
        unfreeze_layers = ['layer.11', 'bert.pooler.', 'out.']
        print(unfreeze_layers)
        for name, param in self.bert.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break
        print("freeze finished")
        self.tokenizer = BertTokenizer.from_pretrained(config["pretrained_model"])
        self.output_size = 768
    def forward(self, inputs, mask):
        outputs = self.bert(inputs, attention_mask=mask)
        return outputs[1]

class BERTSentenceEncoder_new(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.bert = BertModel.from_pretrained(config["pretrained_model"])
        self.tokenizer = BertTokenizer.from_pretrained(config["pretrained_model"])
        self.output_size = 768
        self.fc = nn.Linear(self.output_size*2, self.output_size)
        self.ln = nn.LayerNorm([self.output_size])


    def forward(self, inputs, mask,headid,tailid,is_rel=False):
        outputs = self.bert(inputs, attention_mask=mask)
        if not is_rel:
            tensor_range = torch.arange(inputs.size()[0])  # inputs['word'].shape  [20, 128]
            h_state = outputs['last_hidden_state'][tensor_range, headid]  # h_state.shape [20, 768]
            t_state = outputs['last_hidden_state'][tensor_range, tailid]  # [20, 768]
            # return torch.cat((h_state,t_state),-1)
            return self.ln(self.fc(torch.cat((h_state,t_state),-1)))
        else:
            rel_loc=torch.mean(outputs['last_hidden_state'],1)
            # return torch.cat((outputs['pooler_output'],rel_loc),-1)
            return self.ln(self.fc(torch.cat((outputs['pooler_output'],rel_loc),-1)))

    def show_detail(self, inputs, mask,headid,tailid,is_rel=False):
        outputs = self.bert(inputs, attention_mask=mask)
        if not is_rel:
            tensor_range = torch.arange(inputs.size()[0])  # inputs['word'].shape  [20, 128]
            h_state = outputs['last_hidden_state'][tensor_range, headid]  # h_state.shape [20, 768]
            t_state = outputs['last_hidden_state'][tensor_range, tailid]  # [20, 768]
            return outputs,tensor_range,h_state,t_state



