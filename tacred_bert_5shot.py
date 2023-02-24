import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
import torch.nn as nn
import sys
import json
import gc
from tqdm import tqdm
from sklearn.cluster import KMeans
from encode_bert import BERTSentenceEncoder,BERTSentenceEncoder_new
from dataprocess_bert import data_sampler_bert
from model_bert import proto_softmax_layer_bert
from dataprocess_tacred_bert import get_data_loader_bert
from transformers import BertTokenizer,BertModel,get_linear_schedule_with_warmup
from util_bert import set_seed,process_data,select_similar_data_new_bert,getnegfrombatch_bert,getnegfrombatch_bert_new,generate_neg_samples_bert,infonce_loss
from transformers import AdamW
import random
from torch.optim import Adam
import faiss

def eval_model(config, basemodel, test_set, mem_relations, use_projector = False):
    print("One eval")
    print("test data num is:\t",len(test_set))
    basemodel.eval()

    test_dataloader = get_data_loader_bert(config, test_set, shuffle=False, batch_size=24)
    allnum= 0.0
    correctnum = 0.0
    for step, (labels, neg_labels, sentences, firstent, firstentindex, secondent, secondentindex, headid, tailid, rawtext, lengths,
               typelabels,mask) in enumerate(test_dataloader):
        sentences = sentences.to(config['device'])
        mask = mask.to(config['device'])
        logits, rep = basemodel(sentences, mask,headid,tailid)
        if not use_projector:
            distances = basemodel.get_mem_feature(rep)
        else:
            distances = basemodel.cl_get_mem_feature(rep)
        short_logits = distances

        for index, logit in enumerate(logits):
            score = short_logits[index]
            allnum += 1.0
            golden_score = score[labels[index]]
            max_neg_score = -2147483647.0
            for i in neg_labels[index]:  # range(num_class):
                if (i != labels[index]) and (score[i] > max_neg_score):
                    max_neg_score = score[i]
            if golden_score > max_neg_score:
                correctnum += 1
    acc = correctnum / allnum
    print(acc)
    basemodel.train()
    return acc

def get_memory(config, model, proto_set, sp_token = 0, get_neg = False):
    memset = []
    resset = []
    rangeset = [0]
    for i in proto_set:
        memset += i
        rangeset.append(rangeset[-1] + len(i))
    data_loader = get_data_loader_bert(config, memset, False, False, 1)
    features = []
    all_neg_features = []
    for step, (labels, neg_labels, sentences, firstent, firstentindex, secondent, secondentindex, headid, tailid, rawtext, lengths,
               typelabels,mask) in enumerate(data_loader):

        if headid[0]!=-1:
            if get_neg:
                # generate neg sample
                allnum = len(labels)
                negnum = 1
                allneg, allmask, allheadid, alltailid = [], [], [], []
                for oneindex in range(allnum):
                    negres, maskres, headres, tailres = getnegfrombatch_bert_new(oneindex, firstent, firstentindex,
                                                                             secondent, secondentindex, headid, tailid,
                                                                             sentences, lengths, negnum, allnum,
                                                                             labels, neg_labels, config, sp_token)
                    for aa in negres:
                        allneg.append(torch.tensor(aa))
                    for aa in maskres:
                        allmask.append(torch.tensor(aa))
                    for aa in headres:
                        allheadid.append(torch.tensor(aa))
                    for aa in tailres:
                        alltailid.append(torch.tensor(aa))
                negtensor = torch.stack(allneg, 0)
                masktensor = torch.stack(allmask, 0)
                headtensor = torch.stack(allheadid, 0)
                tailtensor = torch.stack(alltailid, 0)

                sentences = torch.cat((sentences, negtensor), 0)
                mask = torch.cat((mask, masktensor), 0)
                headid = torch.cat((headid, headtensor), 0)
                tailid = torch.cat((tailid, tailtensor), 0)

                sentences = sentences.to(config['device'])
                mask = mask.to(config['device'])
                # feature = model.get_feature(sentences, mask, headid, tailid)
                all_feature = model.get_feature(sentences, mask, headid, tailid)
                feature = all_feature[ : allnum]
                neg_feature = torch.from_numpy(all_feature[allnum:])
                all_neg_features.append(neg_feature)
            else:
                sentences = sentences.to(config['device'])
                mask = mask.to(config['device'])
                feature = model.get_feature(sentences, mask, headid, tailid)
        else:
            sentences = sentences.to(config['device'])
            mask = mask.to(config['device'])
            feature = model.get_feature(sentences, mask, headid, tailid, is_rel=True)
        features.append(feature)
    features = np.concatenate(features)
    protos = []
    for i in range(len(proto_set)):
        # protos.append(torch.tensor(features[rangeset[i]:rangeset[i+1],:].mean(0, keepdims = True)))
        if rangeset[i] == rangeset[i + 1]-1:
            protos.append(torch.tensor(features[rangeset[i]:rangeset[i + 1], :].mean(0, keepdims=True)))
        else:
            protos.append(torch.tensor(features[rangeset[i]+1:rangeset[i + 1], :].mean(0, keepdims=True)) + torch.tensor(features[rangeset[i],:]))
    protos = torch.cat(protos, 0)
    if get_neg:
        all_neg_features = torch.stack(all_neg_features, 0)
    return protos, all_neg_features

def select_data(mem_set, proto_memory, config, model, divide_train_set, num_sel_data, current_relations, selecttype):
    rela_num = len(current_relations)
    for i in range(0, rela_num):
        thisrel = current_relations[i]
        if thisrel in mem_set.keys():
            mem_set[thisrel] = {'0': [], '1': {'h': [], 't': []}}
            proto_memory[thisrel].pop()
        else:
            mem_set[thisrel] = {'0': [], '1': {'h': [], 't': []}}
        thisdataset = divide_train_set[thisrel]
        data_loader = get_data_loader_bert(config, thisdataset, False, False)
        features = []
        for step, (labels, neg_labels, sentences, firstent, firstentindex, secondent, secondentindex, headid, tailid, rawtext,  lengths,
                   typelabels,mask) in enumerate(data_loader):
            sentences = sentences.to(config['device'])
            mask = mask.to(config['device'])
            feature = model.get_feature(sentences, mask,headid,tailid)
            features.append(feature)
        features = np.concatenate(features)
        num_clusters = min(num_sel_data, len(thisdataset))
        if selecttype == 0:
            kmeans = KMeans(n_clusters=num_clusters, random_state=0)
            distances = kmeans.fit_transform(features)
            for i in range(num_clusters):
                sel_index = np.argmin(distances[:, i])
                instance = thisdataset[sel_index]
                instance[11] = 3
                mem_set[thisrel]['0'].append(instance)  ####positive sample
                cluster_center = kmeans.cluster_centers_[i]
                proto_memory[thisrel].append(instance)
        elif selecttype == 1:
            samplenum = features.shape[0]
            veclength = features.shape[1]
            sumvec = np.zeros(veclength)
            for j in range(samplenum):
                sumvec += features[j]
            sumvec /= samplenum

            mindist = 100000000
            minindex = -100
            for j in range(samplenum):
                dist = np.sqrt(np.sum(np.square(features[j] - sumvec)))
                if dist < mindist:
                    minindex = j
                    mindist = dist
            instance = thisdataset[j]
            instance[11] = 3
            mem_set[thisrel]['0'].append(instance)
            proto_memory[thisrel].append(instance)
        else:
            print("error select type")
    if rela_num > 1:
        allnegres = {}
        for i in range(rela_num):
            thisnegres = {'h':[],'t':[]}
            currel = current_relations[i]
            thisrelposnum = len(mem_set[currel]['0'])
            for j in range(thisrelposnum):
                thisnegres['h'].append(mem_set[currel]['0'][j][3])
                thisnegres['t'].append(mem_set[currel]['0'][j][5])
            allnegres[currel] = thisnegres
        for i in range(rela_num):
            togetnegindex = (i + 1) % rela_num
            togetnegrelname = current_relations[togetnegindex]
            mem_set[current_relations[i]]['1']['h'].extend(allnegres[togetnegrelname]['h'])
            mem_set[current_relations[i]]['1']['t'].extend(allnegres[togetnegrelname]['t'])
    return mem_set

tempthre = 0.2


def train_contrastive_model(config, model, mem_set, traindata, epochs, current_proto, seen_relations, mem_neg_features, batch_size = 8, use_sample_self_contrastive = False, use_mem_self_contrastive = False, use_projector = False):
    mem_data=[]
    if len(mem_set) != 0:
        for key in mem_set.keys():
            mem_data.extend(mem_set[key]['0'])
    # print(len(mem_data))
    train_set = traindata
    # data_loader = get_data_loader_bert(config, train_set, batch_size=config['batch_size_per_step'])
    data_loader = get_data_loader_bert(config, train_set, batch_size = batch_size)
    # mem_data_loader = get_data_loader_bert(config, mem_data, batch_size = config['batch_size_per_step'], shuffle=False)
    # optimizer = AdamW(model.parameters(), lr = 2e-5, correct_bias=False)
    optimizer = AdamW(model.parameters(), lr = 2e-5)

    head_flag_list_all = []
    touseindex_list_all = []
    seen_relations_sorted = sorted(seen_relations)
    for epoch_i in range(epochs):
        batch_loss = []
        model.set_memorized_prototypes(current_proto)
        # calculate all mem_data rep first
        # all_mem_sent_emb = {}
        # # all_mem_mask={}
        # # all_mem_headid={}
        # # all_mem_tailid={}
        # for step, (
        #         mem_labels, _, mem_sentences, _, _, _, _, mem_headid, mem_tailid, _,
        #         _, mem_typelabels, mem_mask) in enumerate(mem_data_loader):
        #     mem_sentences = mem_sentences.to(config['device'])
        #     mem_mask = mem_mask.to(config['device'])
        #     logits, rep = model(mem_sentences, mem_mask, mem_headid, mem_tailid)
        #     for k, label in enumerate(mem_labels):
        #         all_mem_sent_emb[int(label)] = rep[k].detach()
        #         # all_mem_headid[int(label)]=headid[k]
        #         # all_mem_tailid[int(label)]=tailid[k]

        # calculate train_data rep
        for step, (labels, neg_labels, sentences, firstent, firstentindex, secondent, secondentindex, headid, tailid, rawtext,
        lengths, typelabels, mask) in enumerate(data_loader):
            model.zero_grad()
            optimizer.zero_grad()
            allnum=len(labels)

            # sample self contrastive
            # if use_sample_self_contrastive:
            #     neg_num = 2
            #     self_neg = []
            #     self_mask = []
            #     self_headid = []
            #     self_tailid = []
            #     for oneindex in range(allnum):
            #         negres, maskres, headres, tailres, head_flag_list, touseindex_list = generate_neg_samples_bert(oneindex,firstent,firstentindex,secondent,
            #                                                                       secondentindex,headid,tailid,sentences,lengths,neg_num,
            #                                                                       allnum,labels,neg_labels,config,epoch_i, head_flag_list_all, touseindex_list_all)
            #         if epoch_i == 0:
            #             head_flag_list_all.append(head_flag_list)
            #             touseindex_list_all.append(touseindex_list)
            #         for aa in negres:
            #             self_neg.append(torch.tensor(aa))
            #         for aa in maskres:
            #             self_mask.append(torch.tensor(aa))
            #         for aa in headres:
            #             self_headid.append(torch.tensor(aa))
            #         for aa in tailres:
            #             self_tailid.append(torch.tensor(aa))
            #     self_negtensor = torch.stack(self_neg, 0)
            #     self_masktensor = torch.stack(self_mask, 0)
            #     self_headtensor = torch.stack(self_headid, 0)
            #     self_tailtensor = torch.stack(self_tailid, 0)

            # if use_sample_self_contrastive:
            #     neg_num = 1
            #     true_neg_num = 2 * neg_num
            #     self_neg = []
            #     self_mask = []
            #     self_headid = []
            #     self_tailid = []
            #     for oneindex in range(allnum):
            #         negres, maskres, headres, tailres= getnegfrombatch_bert_new(oneindex,firstent,firstentindex,secondent,
            #                                                                       secondentindex,headid,tailid,sentences,lengths,neg_num,
            #                                                                       allnum,labels,neg_labels,config,sp_token)
            #         for aa in negres:
            #             self_neg.append(torch.tensor(aa))
            #         for aa in maskres:
            #             self_mask.append(torch.tensor(aa))
            #         for aa in headres:
            #             self_headid.append(torch.tensor(aa))
            #         for aa in tailres:
            #             self_tailid.append(torch.tensor(aa))
            #     self_negtensor = torch.stack(self_neg, 0)
            #     self_masktensor = torch.stack(self_mask, 0)
            #     self_headtensor = torch.stack(self_headid, 0)
            #     self_tailtensor = torch.stack(self_tailid, 0)

            # mem self contrastive
            if use_mem_self_contrastive:
                memindex = []
                numofmem = 0
                for index, onetype in enumerate(typelabels):
                    if onetype == 3:
                        numofmem += 1
                        memindex.append(index)
                getnegfromnum = 1
                allneg = []
                allmask = []
                allheadid = []
                alltailid = []
                if numofmem > 0:
                    for oneindex in memindex:
                        negres, maskres, headres, tailres = getnegfrombatch_bert_new(oneindex, firstent, firstentindex,
                                                                                 secondent, secondentindex, headid, tailid,
                                                                                 sentences, lengths, getnegfromnum, allnum,
                                                                                 labels, neg_labels, config,sp_token)
                        for aa in negres:
                            allneg.append(torch.tensor(aa))
                        for aa in maskres:
                            allmask.append(torch.tensor(aa))
                        for aa in headres:
                            allheadid.append(torch.tensor(aa))
                        for aa in tailres:
                            alltailid.append(torch.tensor(aa))

            pos_mem_sent_emb, neg_mem_sent_emb = [], []
            pos_mem_mask, neg_mem_mask = [], []
            pos_mem_headid, neg_mem_headid = [], []
            pos_mem_tailid, neg_mem_tailid = [], []
            # add projector in contrastive learning
            cl_current_proto = []
            if use_projector:
                # for label in labels:
                #     cl_current_proto[int(label)] = model.contrastive_forward(current_proto[int(label)].cuda())
                cl_current_proto = model.contrastive_forward(current_proto.cuda())

            for label in labels:
                if not use_projector:
                    pos_mem_sent_emb.append(current_proto[int(label)])
                else:
                    pos_mem_sent_emb.append(cl_current_proto[int(label)])
                for rel in seen_relations:
                    if rel != int(label):
                        if not use_projector:
                            neg_mem_sent_emb.append(current_proto[rel])
                        else:
                            neg_mem_sent_emb.append(cl_current_proto[rel])

                # pos_mem_mask.append(all_mem_mask[int(label)])
                # pos_mem_headid.append(all_mem_headid[int(label)])
                # pos_mem_tailid.append(all_mem_tailid[int(label)])
                # for key, value in all_mem_sent_emb.items():
                #     if label != key:
                #         neg_mem_sent_emb.append(value)
                # for key, value in all_mem_mask.items():
                #     if label != key:
                #         neg_mem_mask.append(value)
                # for key, value in all_mem_headid.items():
                #     if label != key:
                #         neg_mem_headid.append(value)
                # for key, value in all_mem_tailid.items():
                #     if label != key:
                #         neg_mem_tailid.append(value)

            pos_mem_sent_emb, neg_mem_sent_emb = torch.stack(pos_mem_sent_emb).to(config['device']), torch.stack(neg_mem_sent_emb).to(config['device'])
            # pos_mem_mask, neg_mem_mask = torch.stack(pos_mem_mask), torch.stack(neg_mem_mask)
            # pos_mem_headid, neg_mem_headid = torch.stack(pos_mem_headid), torch.stack(neg_mem_headid)
            # pos_mem_tailid, neg_mem_tailid = torch.stack(pos_mem_tailid), torch.stack(neg_mem_tailid)

            # sentences = sentences.to(config['device'])
            # mask = mask.to(config['device'])
            # headid = headid.to(config['device'])
            # tailid = tailid.to(config['device'])

            # cat self_neg
            # if use_sample_self_contrastive:
            #     sentences = torch.cat((sentences, self_negtensor), 0)
            #     mask = torch.cat((mask, self_masktensor), 0)
            #     headid = torch.cat((headid, self_headtensor), 0)
            #     tailid = torch.cat((tailid, self_tailtensor), 0)

            if use_mem_self_contrastive:
                if numofmem > 0:
                    negtensor = torch.stack(allneg, 0)
                    masktensor = torch.stack(allmask, 0)
                    headtensor = torch.stack(allheadid, 0)
                    tailtensor = torch.stack(alltailid, 0)

                    sentences = torch.cat((sentences, negtensor), 0)
                    mask = torch.cat((mask, masktensor), 0)
                    headid = torch.cat((headid, headtensor), 0)
                    tailid = torch.cat((tailid, tailtensor), 0)


            sentences = sentences.to(config['device'])
            mask = mask.to(config['device'])
            # labels = labels.to(config['device'])
            logits, rep = model(sentences, mask, headid, tailid)
            logits_drop, rep_drop = model(sentences, mask, headid, tailid)

            if use_projector:
                rep = model.contrastive_forward(rep)

            # mem_label_contrastive
            query = rep[ : allnum]
            mem_neg_emb = neg_mem_sent_emb.view(allnum, len(seen_relations)-1, -1)
            loss1 = infonce_loss(query, pos_mem_sent_emb, mem_neg_emb, temp = 0.12)
            loss = loss1
            # loss = 0.0
            # if use_sample_self_contrastive:
            #     self_neg_lambda = 0.0
            #     self_threshold = 0.9
            #     self_neg_emb = rep[allnum : allnum * (true_neg_num + 1)]
            #     self_neg_emb = self_neg_emb.view(allnum, true_neg_num, -1)
            #     # loss2 = infonce_loss(query, pos_mem_sent_emb, self_neg_emb, temp=0.12, type='self', threshold = self_threshold)
            #     loss2 = infonce_loss(query, rep_drop[ : allnum], self_neg_emb, temp=0.2)
            #     loss += self_neg_lambda * loss2

            # mem_neg
            if use_mem_self_contrastive:
                if numofmem > 0:
                    self_mem_neg_lambda = 1
                    # self_mem_neg_emb = rep[allnum * (neg_num+  1) :]
                    self_mem_neg_emb = rep[allnum :]
                    self_mem_neg_emb = self_mem_neg_emb.view(numofmem, 2 * getnegfromnum, -1)
                    loss3 = infonce_loss(query[memindex], pos_mem_sent_emb[memindex], self_mem_neg_emb, temp=0.12)
                    loss += self_mem_neg_lambda * loss3

            # drop_neg_contrastive
            label_set = set()
            true_drop_index = []
            if len(labels) > 1:
                for j, label in enumerate(labels):
                    if label not in label_set:
                        label_set.add(label)
                        true_drop_index.append(j)
                query_drop = rep_drop[true_drop_index]
                truenum = len(true_drop_index)
                drop_neg_emb = []
                for j in range(truenum):
                    temp = []
                    for k in range(truenum):
                        if k != j:
                            temp.append(query_drop[k])
                    temp = torch.stack(temp, 0)
                    drop_neg_emb.append(temp)
                drop_neg_emb = torch.stack(drop_neg_emb, 0)
                loss4 = infonce_loss(query[true_drop_index], query_drop, drop_neg_emb, temp = 0.04)
            else:
                loss4 = 0
            loss += 2 * loss4

            # mem_proto_neg

            # mem_proto_neg = []
            # for label in labels:
            #     index = seen_relations_sorted.index(label)
            #     mem_proto_neg.append(mem_neg_features[index].cuda())
            # mem_proto_neg = torch.stack(mem_proto_neg, 0)
            # loss5 = infonce_loss(query, pos_mem_sent_emb, mem_proto_neg, temp = 0.2, type = 'self', threshold = 0.9)
            # loss += loss5


            # all_query_drop = rep_drop[ : allnum]
            # loss5 = infonce_loss(query, all_query_drop, mem_neg_emb, temp = 0.08)
            # loss += loss5
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            optimizer.step()
            batch_loss.append(loss.item())
            # batch_avg_loss = batch_cum_loss / (step + 1)
        print('contrastive loss:%.4f' % (np.mean(batch_loss)))
    return model

def train_model_with_hard_neg(config, model, mem_set, traindata, epochs, current_proto, ifnegtive=0):
    # print(len(traindata))
    mem_data = []
    if len(mem_set) != 0:
        for key in mem_set.keys():
            mem_data.extend(mem_set[key]['0'])
    # print(len(mem_data))
    train_set = traindata + mem_data
    # print(len(train_set))
    data_loader = get_data_loader_bert(config, train_set, batch_size = config['batch_size_per_step'])
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config['learning_rate'], correct_bias=False)
    optimizer = AdamW(optimizer_grouped_parameters,lr=config['learning_rate'],
                      correct_bias=False)
    model.train()
    criterion = nn.CrossEntropyLoss()
    lossfn = nn.MultiMarginLoss(margin=0.2)
    for epoch_i in range(epochs):
        model.set_memorized_prototypes(current_proto)
        losses1 = []
        losses2 = []
        losses3 = []
        losses4 = []
        losses5 = []

        lossesfactor1 = 0.0
        lossesfactor2 = 1.0
        lossesfactor3 = 1.0
        lossesfactor4 = 1.0
        # lossesfactor5 = 0.1
        lossesfactor5 = 0.0
        for step, (labels, neg_labels, sentences, firstent, firstentindex, secondent, secondentindex, headid, tailid, rawtext, lengths,
                   typelabels,mask) in enumerate(tqdm(data_loader)):
            model.zero_grad()
            numofmem = 0
            numofnewtrain = 0
            allnum = 0
            memindex = []
            for index,onetype in enumerate(typelabels):
                if onetype == 1:
                    numofnewtrain += 1
                if onetype == 3:
                    numofmem += 1
                    memindex.append(index)
                allnum += 1
            getnegfromnum = 1
            allneg = []
            allmask = []
            allheadid=[]
            alltailid=[]
            if numofmem > 0:
                for oneindex in memindex:
                    negres,maskres,headres,tailres = getnegfrombatch_bert(oneindex,firstent,firstentindex,secondent,secondentindex,headid,tailid,sentences,lengths,getnegfromnum,allnum,labels,neg_labels,config)
                    for aa in negres:
                        allneg.append(torch.tensor(aa))
                    for aa in maskres:
                        allmask.append(torch.tensor(aa))
                    for aa in headres:
                        allheadid.append(torch.tensor(aa))
                    for aa in tailres:
                        alltailid.append(torch.tensor(aa))
            if numofmem > 0:
                negtensor = torch.stack(allneg,0)
                masktensor = torch.stack(allmask,0)

                headtensor=torch.stack(allheadid,0)
                tailtensor=torch.stack(alltailid,0)

                sentences1 = torch.cat((sentences,negtensor),0)
                mask1 = torch.cat((mask,masktensor),0)

                headid1 = torch.cat((headid,headtensor),0)
                tailid1 = torch.cat((tailid,tailtensor),0)

            sentences = sentences.to(config['device'])
            mask = mask.to(config['device'])
            labels = labels.to(config['device'])
            typelabels = typelabels.to(config['device'])  ####0:rel  1:pos(new train data)  2:neg  3:mem
            logits, rep = model(sentences, mask, headid, tailid)
            logits_proto = model.mem_forward(rep)

            # sentences1 = sentences1.to(config['device'])
            # mask1 = mask1.to(config['device'])
            # logits1, rep1 = model(sentences1, mask1, headid1, tailid1)
            # logits_proto1 = model.mem_forward(rep1)
            #
            # outputs, tensor_range, h_state, t_state= model.detail(sentences, mask, headid, tailid)
            # outputs1, tensor_range1, h_state1, t_state1 = model.detail(sentences1, mask1, headid1, tailid1)

            #
            # logitspos = logits[0:allnum,]
            # logits_proto_pos = logits_proto[0:allnum,]
            # if numofmem > 0:
            #     logits_proto_neg = logits_proto[allnum:,]
            #
            # logits = logitspos
            # logits_proto = logits_proto_pos

            loss1 = criterion(logits, labels)
            loss2 = criterion(logits_proto, labels)
            loss4 = lossfn(logits_proto, labels)
            loss3 = torch.tensor(0.0).to(config['device'])
            for index, logit in enumerate(logits):
                score = logits_proto[index]
                preindex = labels[index]
                maxscore = score[preindex]
                size = score.shape[0]
                secondmax = -100000
                for j in range(size):
                    if j != preindex and score[j] > secondmax:
                        secondmax = score[j]
                if secondmax - maxscore + tempthre > 0.0:
                    loss3 += (secondmax - maxscore + tempthre).to(config['device'])
            loss3 /= logits.shape[0]

            # start = 0
            # loss5 = torch.tensor(0.0).to(config['device'])
            # allusenum = 0
            # for index in memindex:
            #     onepos = logits_proto[index]
            #     posindex = labels[index]
            #     poslabelscore = onepos[posindex]
            #     negnum = getnegfromnum * 2
            #     negscore = torch.tensor(0.0).to(config['device'])
            #     for ii in range(start, start + negnum):
            #         oneneg = logits_proto_neg[ii]
            #         negscore = oneneg[posindex]
            #         if negscore - poslabelscore + 0.01 > 0.0 and negscore < poslabelscore:
            #             loss5 += (negscore - poslabelscore + 0.01)
            #             allusenum += 1
            #     start += negnum
            # if len(memindex) == 0:
            #     loss = loss1 * lossesfactor1 + loss2 * lossesfactor2 + loss3 * lossesfactor3 + loss4 * lossesfactor4
            # else:
            #     loss5 = loss5 / allusenum
            #     loss = loss1 * lossesfactor1 + loss2 * lossesfactor2 + loss3 * lossesfactor3 + loss4 * lossesfactor4 + loss5 * lossesfactor5    ###with loss
            loss = loss1 * lossesfactor1 + loss2 * lossesfactor2 + loss3 * lossesfactor3 + loss4 * lossesfactor4
            loss.backward()
            losses1.append(loss1.item())
            losses2.append(loss2.item())
            losses3.append(loss3.item())
            losses4.append(loss4.item())
            # losses5.append(loss5.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            optimizer.step()
        print('hard loss!:%.4f'%(loss.mean()))
    return model

def train_simple_model(config, model, mem_set, traindata, epochs, current_proto, ifusemem=False, batch_size = 5, steps = 0, use_self_neg_contrastive = False):
    train_set = traindata
    if ifusemem:
        mem_data = []
        if len(mem_set)!=0:
            for key in mem_set.keys():
                mem_data.extend(mem_set[key]['0'])
        # train_set.extend(mem_data)
        train_set = traindata + mem_data
    # else:
    #     mem_data = []
    #     if len(mem_set) != 0:
    #         for key in mem_set.keys():
    #             mem_data.extend(mem_set[key]['0'])
    #     # print(len(mem_data))
    #     train_set = train_set + mem_data

    data_loader = get_data_loader_bert(config, train_set, batch_size = batch_size)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config['learning_rate'], correct_bias=False)
    optimizer = AdamW(optimizer_grouped_parameters,lr=config['learning_rate'],
                      correct_bias=False)
    # scheduler=get_linear_schedule_with_warmup(optimizer,num_warmup_steps=warmup_step,num_training_steps=train_iter)
    model.train()
    criterion = nn.CrossEntropyLoss()
    lossfn = nn.MultiMarginLoss(margin=0.2)
    for epoch_i in range(epochs):
        model.set_memorized_prototypes(current_proto)
        losses1 = []
        losses2 = []
        losses3 = []
        losses4 = []

        lossesfactor1 = 0.0
        lossesfactor2 = 1.0
        lossesfactor3 = 1.0
        lossesfactor4 = 1.0

        for step, (labels, neg_labels, sentences, firstent, firstentindex, secondent, secondentindex, headid, tailid, rawtext,
                   lengths, typelabels, mask) in enumerate(data_loader):
            if not steps:
                model.zero_grad()
                sentences = sentences.to(config['device'])
                mask = mask.to(config['device'])
                logits, rep = model(sentences, mask,headid,tailid)
                logits_proto = model.mem_forward(rep)

                labels = labels.to(config['device'])
                loss1 = criterion(logits, labels)
                loss2 = criterion(logits_proto, labels)
                loss4 = lossfn(logits_proto, labels)
                loss3 = torch.tensor(0.0).to(config['device'])
                for index, logit in enumerate(logits):
                    score = logits_proto[index]
                    preindex = labels[index]
                    maxscore = score[preindex]
                    size = score.shape[0]
                    secondmax = -100000
                    for j in range(size):
                        if j != preindex and score[j] > secondmax:
                            secondmax = score[j]
                    if secondmax - maxscore + tempthre > 0.0:
                        loss3 += (secondmax - maxscore + tempthre).to(config['device'])

                loss3 /= logits.shape[0]
                loss = loss1 * lossesfactor1 + loss2 * lossesfactor2 + loss3 * lossesfactor3 + loss4 * lossesfactor4
                loss.backward()

                #for n, p in model.named_parameters():
                #    if p.requires_grad:
                #        print(n, p.grad.shape)
                losses1.append(loss1.item())
                losses2.append(loss2.item())
                losses3.append(loss3.item())
                losses4.append(loss4.item())
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                optimizer.step()
                # scheduler.step()
            else:
                model.zero_grad()
                neg_num = 1
                true_neg_num = 2 * neg_num
                self_neg = []
                self_mask = []
                self_headid = []
                self_tailid = []
                allnum = len(labels)
                for oneindex in range(allnum):
                    negres, maskres, headres, tailres = getnegfrombatch_bert_new(oneindex, firstent, firstentindex,
                                                                                 secondent,
                                                                                 secondentindex, headid, tailid,
                                                                                 sentences, lengths, neg_num,
                                                                                 allnum, labels, neg_labels, config,
                                                                                 sp_token)
                    for aa in negres:
                        self_neg.append(torch.tensor(aa))
                    for aa in maskres:
                        self_mask.append(torch.tensor(aa))
                    for aa in headres:
                        self_headid.append(torch.tensor(aa))
                    for aa in tailres:
                        self_tailid.append(torch.tensor(aa))
                self_negtensor = torch.stack(self_neg, 0)
                self_masktensor = torch.stack(self_mask, 0)
                self_headtensor = torch.stack(self_headid, 0)
                self_tailtensor = torch.stack(self_tailid, 0)
                sentences = torch.cat((sentences, self_negtensor), 0)
                mask = torch.cat((mask, self_masktensor), 0)
                headid = torch.cat((headid, self_headtensor), 0)
                tailid = torch.cat((tailid, self_tailtensor), 0)

                sentences = sentences.to(config['device'])
                mask = mask.to(config['device'])
                logits, rep = model(sentences, mask, headid, tailid)
                logits_drop, rep_drop = model(sentences, mask, headid, tailid)
                logits1, rep1 = logits[ : allnum], rep[ : allnum]
                logits_proto = model.mem_forward(rep1)

                labels = labels.to(config['device'])
                loss1 = criterion(logits1, labels)
                loss2 = criterion(logits_proto, labels)
                loss4 = lossfn(logits_proto, labels)
                loss3 = torch.tensor(0.0).to(config['device'])
                for index, logit in enumerate(logits1):
                    score = logits_proto[index]
                    preindex = labels[index]
                    maxscore = score[preindex]
                    size = score.shape[0]
                    secondmax = -100000
                    for j in range(size):
                        if j != preindex and score[j] > secondmax:
                            secondmax = score[j]
                    if secondmax - maxscore + tempthre > 0.0:
                        loss3 += (secondmax - maxscore + tempthre).to(config['device'])

                loss3 /= logits1.shape[0]
                loss = loss1 * lossesfactor1 + loss2 * lossesfactor2 + loss3 * lossesfactor3 + loss4 * lossesfactor4

                # self neg contrastive
                if use_self_neg_contrastive:
                    self_neg_lambda = 1
                    self_neg_emb = rep[allnum: allnum * (true_neg_num + 1)]
                    self_neg_emb = self_neg_emb.view(allnum, true_neg_num, -1)
                    # loss2 = infonce_loss(query, pos_mem_sent_emb, self_neg_emb, temp=0.12, type='self', threshold = self_threshold)
                    loss5 = infonce_loss(rep[ : allnum], rep_drop[: allnum], self_neg_emb, temp = 0.12, type= 'self', threshold = 0.9)
                    loss += self_neg_lambda * loss5

                loss.backward()

                # for n, p in model.named_parameters():
                #    if p.requires_grad:
                #        print(n, p.grad.shape)
                losses1.append(loss1.item())
                losses2.append(loss2.item())
                losses3.append(loss3.item())
                losses4.append(loss4.item())
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                optimizer.step()
        print('simple loss!:%.4f' % (loss.mean()))
    return model

def train_classify_model(config, model, mem_set, traindata, epochs, ifusemem=False):
    train_set = traindata
    if ifusemem:
        mem_data = []
        if len(mem_set)!=0:
            for key in mem_set.keys():
                mem_data.extend(mem_set[key]['0'])
        # train_set.extend(mem_data)
        train_set = traindata + mem_data
    # else:
    #     mem_data = []
    #     if len(mem_set) != 0:
    #         for key in mem_set.keys():
    #             mem_data.extend(mem_set[key]['0'])
    #     # print(len(mem_data))
    #     train_set = train_set + mem_data

    data_loader = get_data_loader_bert(config, train_set, batch_size = config['batch_size_per_step'])
    # param_optimizer = list(model.named_parameters())
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]
    optimizer = Adam([
        {'params': model.sentence_encoder.parameters(), 'lr': 0.00001},
        {'params': model.drop.parameters(), 'lr': 0.00001},
        {'params': model.fc.parameters(), 'lr': 0.001}
    ])
    # optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config['learning_rate'], correct_bias=False)
    # optimizer = AdamW(optimizer_grouped_parameters,lr=config['learning_rate'],
    #                   correct_bias=False)
    # scheduler=get_linear_schedule_with_warmup(optimizer,num_warmup_steps=warmup_step,num_training_steps=train_iter)
    model.train()
    criterion = nn.CrossEntropyLoss()
    # lossfn = nn.MultiMarginLoss(margin=0.2)
    for epoch_i in range(epochs):
        # model.set_memorized_prototypes(current_proto)
        losses1 = []
        lossesfactor1 = 1.0

        for step, (labels, neg_labels, sentences, firstent, firstentindex, secondent, secondentindex, headid, tailid, rawtext,
                   lengths, typelabels, mask) in enumerate(tqdm(data_loader)):
            model.zero_grad()
            sentences = sentences.to(config['device'])
            mask = mask.to(config['device'])
            logits, rep = model(sentences, mask,headid,tailid)
            labels = labels.to(config['device'])
            loss1 = criterion(logits, labels)
            loss = loss1 * lossesfactor1
            loss.backward()
            losses1.append(loss1.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            optimizer.step()
        print('classify loss!:%.4f' % (loss.mean()))
    return model



def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':

    select_thredsold_param = 0.65
    select_num = 1
    f = open("config/config_tacred_bert.json", "r")
    config = json.loads(f.read())
    f.close()
    config['device'] = torch.device('cuda' if torch.cuda.is_available() and config['use_gpu'] else 'cpu')
    config['n_gpu'] = torch.cuda.device_count()
    config['batch_size_per_step'] = int(config['batch_size'] / config["gradient_accumulation_steps"])
    config['neg_sampling'] = False
    config['pretrain'] = True

    bert_file='bert-base-uncased'
    if config['pretrain']:
        config["pretrained_model"] = bert_file
    config["max_length"] = 256
    # print(config["learning_rate"])

    # root_path = '.'
    # word2id = json.load(open(os.path.join(root_path, 'glove/word2id.txt')))
    # word2vec = np.load(os.path.join(root_path, 'glove/word2vec.npy'))

    tokenizer = BertTokenizer.from_pretrained(bert_file)
    sp_token = tokenizer.convert_tokens_to_ids("[unused4]")
    # distantpath = "data/distantdata/"
    # file1 = distantpath + "distant.json"
    # file2 = distantpath + "exclude_fewrel_distant.json"
    # list_data,entpair2scope = process_data(file1,file2)
    donum = 1
    topk = 16
    max_sen_length_for_select = 128
    max_sen_lstm_tokenize = 256
    select_thredsold = select_thredsold_param

    # add relation_info
    config['relation_info_file'] = './data/fewrel/fewrel_rel_info.json'

    #
    # print("********* load from ckpt ***********")
    # ckptpath = "simmodelckpt"
    # print(ckptpath)
    # ckpt = torch.load(ckptpath)
    # SimModel = BertModel.from_pretrained(bert_file, state_dict=ckpt["bert-base"]).to(config["device"])
    #
    # allunlabledata = np.load("allunlabeldata.npy").astype('float32')
    # d = 768 * 2
    # index = faiss.IndexFlatIP(d)
    # print(index.is_trained)
    # index.add(allunlabledata)
    # print(index.ntotal)
    ifuseckpt = False
    ckptpath=''
    usenewbert=True

    userelinfo = False

    if not userelinfo:
        config['relation_info_file'] = None

    for m in range(donum):
        # print(m)
        config["rel_cluster_label"] = "data/tacred/CFRLdata_10_100_10_5/rel_cluster_label_" + str(m) + ".npy"
        config['training_file'] = "data/tacred/CFRLdata_10_100_10_5/train_" + str(m) + ".txt"
        config['valid_file'] = "data/tacred/CFRLdata_10_100_10_5/valid_" + str(m) + ".txt"
        config['test_file'] = "data/tacred/CFRLdata_10_100_10_5/test_" + str(m) + ".txt"
        #
        # config["rel_cluster_label"] = "data/fewrel/CFRLdata_10_100_10_10/rel_cluster_label_" + str(m) + ".npy"
        # config['training_file'] = "data/fewrel/CFRLdata_10_100_10_10/train_" + str(m) + ".txt"
        # config['valid_file'] = "data/fewrel/CFRLdata_10_100_10_10/valid_" + str(m) + ".txt"
        # config['test_file'] = "data/fewrel/CFRLdata_10_100_10_10/test_" + str(m) + ".txt"

        # config["rel_cluster_label"] = "data/fewrel/CFRLdata_10_100_10_2/rel_cluster_label_" + str(m) + ".npy"
        # config['training_file'] = "data/fewrel/CFRLdata_10_100_10_2/train_" + str(m) + ".txt"
        # config['valid_file'] = "data/fewrel/CFRLdata_10_100_10_2/valid_" + str(m) + ".txt"
        # config['test_file'] = "data/fewrel/CFRLdata_10_100_10_2/test_" + str(m) + ".txt"

        if config['pretrain']:
            # if ifuseckpt:
            #     encoderforbase = BERTSentenceEncoder(config=config,ckptpath = ckptpath)
            # else:
            #     encoderforbase = BERTSentenceEncoder(config=config)
            if usenewbert:
                encoderforbase = BERTSentenceEncoder_new(config=config)
            else:
                encoderforbase = BERTSentenceEncoder(config=config)
        else:
            print("you should use bert!")
            exit -1
        sampler = data_sampler_bert(config, encoderforbase.tokenizer)
        modelforbase = proto_softmax_layer_bert(encoderforbase, num_class=len(sampler.id2rel), id2rel=sampler.id2rel, drop=0, config=config)
        modelforbase = modelforbase.to(config["device"])

        # word2vec_back = word2vec.copy()

        sequence_results = []
        result_whole_test = []

        for i in range(6):

            num_class = len(sampler.id2rel)
            # print(config['random_seed'] + 10 * i)
            # set_seed(config, config['random_seed'] + 10 * i)
            seed_everything(config['random_seed'] + i)
            sampler.set_seed(config['random_seed'] + i)

            mem_set = {}
            mem_relations = []

            past_relations = []

            savetest_all_data = None
            saveseen_relations = []

            proto_memory = []

            mem_data = []

            for j in range(len(sampler.id2rel)):
                proto_memory.append([sampler.id2rel_pattern[j]])
            oneseqres = []
            ifnorm = True
            for steps, (training_data, valid_data, test_data, test_all_data, seen_relations, current_relations) in enumerate(sampler):
                # print(len(training_data))
                savetest_all_data = test_all_data
                saveseen_relations = seen_relations

                currentnumber = len(current_relations)
                # print(currentnumber)
                # print(current_relations)
                divide_train_set = {}
                for relation in current_relations:
                    divide_train_set[relation] = []
                for data in training_data:
                    divide_train_set[data[0]].append(data)
                # print(len(divide_train_set))
                #
                # if steps == 0:
                #     print("train base model,not select most similar")
                #
                # else:
                #     print("train new model,select most similar")
                #     selectdata = select_similar_data_new_bert(training_data, tokenizer, entpair2scope, topk,
                #                                          max_sen_length_for_select, list_data, config, SimModel,
                #                                          select_thredsold, max_sen_lstm_tokenize,
                #                                          encoderforbase.tokenizer, index, ifnorm, select_num)
                #     print(len(selectdata))
                #     print(selectdata[0])
                #     print(training_data[0])
                #     training_data.extend(selectdata)
                #     print(len(training_data))

                if steps == 0:
                    epochnum = 4
                else:
                    epochnum = 4

                contrastive_epochs = 5

                # modelforbase = train_classify_model(config, modelforbase, mem_set, training_data, 1,  False)

                for j in range(1):
                    current_proto, _ = get_memory(config, modelforbase, proto_memory)
                    modelforbase = train_simple_model(config, modelforbase, mem_set, training_data, 1, current_proto, False, batch_size = 5)
                    # select_data(mem_set, proto_memory, config, modelforbase, divide_train_set,
                    #             config['rel_memory_size'], current_relations, 0)  ##config['rel_memory_size'] == 1
                select_data(mem_set, proto_memory, config, modelforbase, divide_train_set,
                            1, current_relations, 0)

                for j in range(epochnum):
                    current_proto, _ = get_memory(config, modelforbase, proto_memory)
                    # modelforbase = train_model_with_hard_neg(config, modelforbase, mem_set, training_data, 1, current_proto, ifnegtive=0)
                    modelforbase = train_simple_model(config, modelforbase, mem_set, training_data,
                                                      1, current_proto, True, batch_size = 5, steps = steps, use_self_neg_contrastive = True)
                if steps != 0:
                    for j in range(contrastive_epochs):
                        current_proto, mem_neg_features = get_memory(config, modelforbase, proto_memory, sp_token = sp_token, get_neg = True)
                        modelforbase = train_contrastive_model(config, modelforbase, mem_set,
                                                               training_data, 1, current_proto, seen_relations, mem_neg_features,
                                                               use_mem_self_contrastive = False,
                                                               batch_size = 8)
                current_proto, _ = get_memory(config, modelforbase, proto_memory)
                modelforbase.set_memorized_prototypes(current_proto)
                mem_relations.extend(current_relations)

                currentalltest = []
                for mm in range(len(test_data)):
                    currentalltest.extend(test_data[mm])

                thisstepres = eval_model(config, modelforbase, currentalltest, mem_relations , use_projector = False)
                print("step:\t",steps,"\taccuracy:\t",thisstepres)
                oneseqres.append(thisstepres)
            sequence_results.append(np.array(oneseqres))

            allres = eval_model(config, modelforbase, savetest_all_data, saveseen_relations, use_projector = False)
            result_whole_test.append(allres)

            print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            print("after one epoch allres:\t",allres)
            print(result_whole_test)

            # initialize the models
            modelforbase = modelforbase.to('cpu')
            del modelforbase
            gc.collect()
            if config['device'] == 'cuda':
                torch.cuda.empty_cache()

            if usenewbert:
                encoderforbase = BERTSentenceEncoder_new(config=config)
            else:
                encoderforbase = BERTSentenceEncoder(config=config)
            modelforbase = proto_softmax_layer_bert(encoderforbase, num_class=len(sampler.id2rel),
                                                    id2rel=sampler.id2rel, drop=0, config=config)
            modelforbase.to(config["device"])
        print("Final result!")
        print(result_whole_test)
        for one in sequence_results:
            for item in one:
                sys.stdout.write('%.4f, ' % item)
            print('')
        avg_result_all_test = np.average(sequence_results, 0)
        print("Final avg result!")
        for one in avg_result_all_test:
            sys.stdout.write('%.4f, ' % one)
        std_result_all_test = np.std(sequence_results, 0)
        print("Final std result!")
        for one in std_result_all_test:
            sys.stdout.write('%.4f, ' % one)
        print('')
        print("Finish training............................")