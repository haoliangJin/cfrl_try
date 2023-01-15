import torch
import torch.nn as nn
import os
import json
import torch.nn.functional as F

class base_model(nn.Module):

    def __init__(self):
        super(base_model, self).__init__()
        self.zero_const = nn.Parameter(torch.Tensor([0]))
        self.zero_const.requires_grad = False
        self.pi_const = nn.Parameter(torch.Tensor([3.14159265358979323846]))
        self.pi_const.requires_grad = False

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(os.path.join(path)))
        self.eval()

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)

    def load_parameters(self, path):
        f = open(path, "r")
        parameters = json.loads(f.read())
        f.close()
        for i in parameters:
            parameters[i] = torch.Tensor(parameters[i])
        self.load_state_dict(parameters, strict = False)
        self.eval()

    def save_parameters(self, path):
        f = open(path, "w")
        f.write(json.dumps(self.get_parameters("list")))
        f.close()

    def get_parameters(self, mode = "numpy", param_dict = None):
        all_param_dict = self.state_dict()
        if param_dict == None:
            param_dict = all_param_dict.keys()
        res = {}
        for param in param_dict:
            if mode == "numpy":
                res[param] = all_param_dict[param].cpu().numpy()
            elif mode == "list":
                res[param] = all_param_dict[param].cpu().numpy().tolist()
            else:
                res[param] = all_param_dict[param]
        return res

    def set_parameters(self, parameters):
        for i in parameters:
            parameters[i] = torch.Tensor(parameters[i])
        self.load_state_dict(parameters, strict = False)
        self.eval()

class proto_softmax_layer_bert(base_model):

    def __distance__(self, rep, rel):
        rep_norm = rep / rep.norm(dim=1)[:, None]
        rel_norm = rel / rel.norm(dim=1)[:, None]
        res = torch.mm(rep_norm, rel_norm.transpose(0, 1))
        return res

    def __init__(self, sentence_encoder, num_class, id2rel, drop=0.1, config=None, rate=1.0):
        super(proto_softmax_layer_bert, self).__init__()

        self.config = config
        self.sentence_encoder = sentence_encoder
        self.num_class = num_class
        self.hidden_size = self.sentence_encoder.output_size
        # self.fc = nn.Linear(self.hidden_size*2, self.num_class, bias=False)
        self.fc = nn.Linear(self.hidden_size, self.num_class, bias=False)
        self.drop = nn.Dropout(drop)
        self.id2rel = id2rel
        self.rel2id = {}
        for id, rel in id2rel.items():
            self.rel2id[rel] = id
        self.ln = nn.LayerNorm([self.num_class])
        # add projector in contrastive learning
        self.con_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.con_fc2 = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.gelu = nn.GELU()

    def set_memorized_prototypes(self, protos):
        self.prototypes = protos.detach().to(self.config['device']) ##detach: release mem_emb from graph(important)

    def get_feature(self, sentences, mask, headid, tailid,is_rel=False):
        rep = self.sentence_encoder(sentences, mask, headid, tailid,is_rel)
        return rep.cpu().data.numpy()

    def get_mem_feature(self, rep):
        dis = self.mem_forward(rep)
        return dis.cpu().data.numpy()

    def cl_get_mem_feature(self, rep):
        cl_rep = self.contrastive_forward(rep)
        cl_prototypes= self.contrastive_forward(self.prototypes)
        return self.__distance__(cl_rep, cl_prototypes).cpu().data.numpy()

    def forward(self, sentences, mask, headid, tailid):
        rep = self.sentence_encoder(sentences, mask, headid, tailid)  # (B, H)
        repd = self.drop(rep)
        logits = self.fc(repd)
        #
        # logits = F.gelu(logits)
        # logits = self.ln(logits)

        return logits, rep

    def mem_forward(self, rep):
        dis_mem = self.__distance__(rep, self.prototypes)
        return dis_mem

    def contrastive_forward(self, rep):
        return self.con_fc2(self.gelu(self.con_fc1(rep)))

    def detail(self, sentences, mask, headid, tailid):
        outputs, tensor_range, h_state, t_state = self.sentence_encoder.show_detail(sentences, mask, headid, tailid)
        return outputs, tensor_range, h_state, t_state
