import random
import torch
import numpy as np
import re
import json
from collections import defaultdict
import torch.nn.functional as F

def set_seed(config, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if config['n_gpu'] > 0 and torch.cuda.is_available() and config['use_gpu']:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def getnegfrombatch_bert(oneindex,firstent,firstentindex,secondent,secondentindex,headid,tailid,sentences,lengths,getnegfromnum,allnum,labels,neg_labels,config):
    thissentence = sentences[oneindex].cpu().numpy().tolist()
    thislength = lengths[oneindex]
    thisfirstent = firstent[oneindex]
    thisfirstentindex = firstentindex[oneindex].numpy().tolist()
    headstart = thisfirstentindex[0]
    headend = thisfirstentindex[-1]
    posheadlength = len(thisfirstentindex)

    thisheadid = headid[oneindex]

    thissecondent = secondent[oneindex]
    thissecondentindex = secondentindex[oneindex].numpy().tolist()
    tailstart = thissecondentindex[0]
    tailend = thissecondentindex[-1]
    postaillength = len(thissecondentindex)

    thistailid = tailid[oneindex]

    negres = []
    maskres = []
    headidres=[]
    tailidres=[]

    for j in range(getnegfromnum):
        touseindex = (oneindex + j + 1) % allnum
        negusehead = firstent[touseindex].numpy().tolist()
        negheadlength = len(negusehead)
        negusetail = secondent[touseindex].numpy().tolist()
        negtaillength = len(negusetail)
        negsamplechangehead = thissentence[0:headstart] + negusehead + thissentence[headend + 1:]
        changeheadlength = thislength - posheadlength + negheadlength
        if len(negsamplechangehead) > config["max_length"]:
            negsamplechangehead = negsamplechangehead[0:config["max_length"]]
        for i in range(len(negsamplechangehead), config["max_length"]):
            negsamplechangehead.append(0)
        mask1 = []
        for i in range(0, changeheadlength):
            mask1.append(1)
        for i in range(changeheadlength, config["max_length"]):
            mask1.append(0)
        if len(mask1) > config["max_length"]:
            mask1 = mask1[0:config["max_length"]]

        negsamplechangetail = thissentence[0:tailstart] + negusetail + thissentence[tailend + 1:]
        changetaillength = thislength - postaillength + negtaillength
        if len(negsamplechangetail) > config["max_length"]:
            negsamplechangetail = negsamplechangetail[0:config["max_length"]]
        for i in range(len(negsamplechangetail), config["max_length"]):
            negsamplechangetail.append(0)
        mask2 = []
        for i in range(0, changetaillength):
            mask2.append(1)
        for i in range(changetaillength, config["max_length"]):
            mask2.append(0)
        if len(mask2) > config["max_length"]:
            mask2 = mask2[0:config["max_length"]]

        if len(mask1) != len(mask2):
            print(len(mask1))
            print(len(mask2))
            print(mask1)
            print(mask2)

        negres.append(negsamplechangehead)
        maskres.append(mask1)
        # headidres.append(thisheadid)
        # tailidres.append(min(thistailid- posheadlength + negheadlength, config["max_length"]))

        # tailid might be smaller than headid
        if thisheadid < thistailid:
            headidres.append(thisheadid)
            tailidres.append(min(thistailid - posheadlength + negheadlength, config["max_length"]))
        else:
            headidres.append(thisheadid)
            tailidres.append(thistailid)

        negres.append(negsamplechangetail)
        maskres.append(mask2)
        # headidres.append(thisheadid)
        # tailidres.append(thistailid)
        if thisheadid < thistailid:
            headidres.append(thisheadid)
            tailidres.append(thistailid)
        else:
            tailidres.append(thistailid)
            headidres.append(min(thisheadid - postaillength + negtaillength, config["max_length"]))



    return np.asarray(negres),np.asarray(maskres),np.asarray(headidres),np.asarray(tailidres)


def generate_neg_samples_bert(oneindex,firstent,firstentindex,secondent,secondentindex,headid,tailid,sentences,lengths,neg_num,allnum,labels,neg_labels,config, epoch_i, head_flag_list_all, touseindex_list_all):
    # random.seed(seed)
    thissentence = sentences[oneindex].cpu().numpy().tolist()
    thislength = lengths[oneindex]
    thisfirstent = firstent[oneindex]
    thisfirstentindex = firstentindex[oneindex].numpy().tolist()
    headstart = thisfirstentindex[0] #[ent11]
    headend = thisfirstentindex[-1] #[ent1n]
    posheadlength = len(thisfirstentindex)

    thisheadid=headid[oneindex] # [E11] pos

    thissecondent = secondent[oneindex]
    thissecondentindex = secondentindex[oneindex].numpy().tolist()
    tailstart = thissecondentindex[0] #[ent21]
    tailend = thissecondentindex[-1] #[ent2n]
    postaillength = len(thissecondentindex)

    thistailid=tailid[oneindex] # [E21] pos

    negres = []
    maskres = []
    headidres = []
    tailidres = []

    if epoch_i == 0:
        head_flag_list = []
        touseindex_list = []
    else:
        head_flag_list = head_flag_list_all[oneindex]
        touseindex_list = touseindex_list_all[oneindex]

    for j in range(neg_num):
        if epoch_i == 0:
            change_head_flag = random.randint(0, 1)
            head_flag_list.append(change_head_flag)
            while True:
                touseindex = random.randint(0, allnum-1)
                if touseindex != oneindex:
                    break
            touseindex_list.append(touseindex)
        else:
            change_head_flag = head_flag_list[j]
            touseindex = touseindex_list[j]

        if change_head_flag:
            negusehead = firstent[touseindex].numpy().tolist()
            negheadlength = len(negusehead)
            negsamplechangehead = thissentence[0:headstart] + negusehead + thissentence[headend + 1:]
            changeheadlength = thislength - posheadlength + negheadlength
            if len(negsamplechangehead) > config["max_length"]:
                negsamplechangehead = negsamplechangehead[0:config["max_length"]]
            for i in range(len(negsamplechangehead), config["max_length"]):
                negsamplechangehead.append(0)
            mask1 = []
            for i in range(0, changeheadlength):
                mask1.append(1)
            for i in range(changeheadlength, config["max_length"]):
                mask1.append(0)
            if len(mask1) > config["max_length"]:
                mask1 = mask1[0:config["max_length"]]
            negres.append(negsamplechangehead)
            maskres.append(mask1)
            # tailid might be smaller than headid
            if thisheadid < thistailid :
                headidres.append(thisheadid)
                tailidres.append(min(thistailid - posheadlength + negheadlength, config["max_length"]))
            else:
                headidres.append(thisheadid)
                tailidres.append(thistailid)
        else:
            negusetail = secondent[touseindex].numpy().tolist()
            negtaillength = len(negusetail)
            negsamplechangetail = thissentence[0:tailstart] + negusetail + thissentence[tailend + 1:]
            changetaillength = thislength - postaillength + negtaillength
            if len(negsamplechangetail) > config["max_length"]:
                negsamplechangetail = negsamplechangetail[0:config["max_length"]]
            for i in range(len(negsamplechangetail), config["max_length"]):
                negsamplechangetail.append(0)
            mask2 = []
            for i in range(0, changetaillength):
                mask2.append(1)
            for i in range(changetaillength, config["max_length"]):
                mask2.append(0)
            if len(mask2) > config["max_length"]:
                mask2 = mask2[0:config["max_length"]]
            negres.append(negsamplechangetail)
            maskres.append(mask2)
            # tailid might be smaller than headid
            if thisheadid < thistailid:
                headidres.append(thisheadid)
                tailidres.append(thistailid)
            else:
                tailidres.append(thistailid)
                headidres.append(min(thisheadid - postaillength + negtaillength, config["max_length"]))
    return np.asarray(negres), np.asarray(maskres), np.asarray(headidres), np.asarray(tailidres), head_flag_list, touseindex_list


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]


def transpose(x):
    return x.transpose(-2, -1)


def infonce_loss(query, pos_emb, neg_emb, temp=0.1, reduction='mean', type='mem', threshold=0.5):
    query, pos_emb, neg_emb = normalize(query, pos_emb, neg_emb)
    positive_logit = torch.sum(query * pos_emb, dim=1, keepdim=True)
    query = query.unsqueeze(1)
    negative_logits = query @ transpose(neg_emb)
    negative_logits = negative_logits.squeeze(1)
    if type == 'mem':
        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
        return F.cross_entropy(logits / temp, labels, reduction=reduction)
    else:
        negative_logits[abs(negative_logits) < threshold] = -1e9
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
        return F.cross_entropy(logits / temp, labels, reduction=reduction)




def handletoken(raw_text,h_pos_li,t_pos_li,tokenizer):
    h_pattern = re.compile("\* h \*")
    t_pattern = re.compile("\^ t \^")
    err = 0
    tokens = []
    h_mention = []
    t_mention = []
    raw_text_list = raw_text.split(" ")
    for i, token in enumerate(raw_text_list):
        token = token.lower()
        if i >= h_pos_li[0] and i <= h_pos_li[-1]:
            if i == h_pos_li[0]:
                tokens += ['*', 'h', '*']
            h_mention.append(token)
            continue
        if i >= t_pos_li[0] and i <= t_pos_li[-1]:
            if i == t_pos_li[0]:
                tokens += ['^', 't', '^']
            t_mention.append(token)
            continue
        tokens.append(token)
    text = " ".join(tokens)
    h_mention = " ".join(h_mention)
    t_mention = " ".join(t_mention)
    tokenized_text = tokenizer.tokenize(text)
    tokenized_head = tokenizer.tokenize(h_mention)
    tokenized_tail = tokenizer.tokenize(t_mention)

    p_text = " ".join(tokenized_text)
    p_head = " ".join(tokenized_head)
    p_tail = " ".join(tokenized_tail)
    p_text = h_pattern.sub("[unused0] " + p_head + " [unused1]", p_text)
    p_text = t_pattern.sub("[unused2] " + p_tail + " [unused3]", p_text)
    f_text = ("[CLS] " + p_text + " [SEP]").split()
    try:
        h_pos = f_text.index("[unused0]")
    except:
        err += 1
        h_pos = 0
    try:
        t_pos = f_text.index("[unused2]")
    except:
        err += 1
        t_pos = 0

    tokenized_input = tokenizer.convert_tokens_to_ids(f_text)

    return tokenized_input, h_pos, t_pos


def filter_sentence(sentence):
    head_pos = sentence["h"]["pos"][0]
    tail_pos = sentence["t"]["pos"][0]

    if sentence["h"]["name"] == sentence["t"]["name"]:  # head mention equals tail mention
        return True

    if head_pos[0] >= tail_pos[0] and head_pos[0] <= tail_pos[-1]:  # head mentioin and tail mention overlap
        return True

    if tail_pos[0] >= head_pos[0] and tail_pos[0] <= head_pos[-1]:  # head mentioin and tail mention overlap
        return True

    return False

def process_data(file1,file2):
    data1 = json.load(open(file1))
    #data2 = json.load(open(file2))
    data2 = {}
    max_num = 16 ###max number for every entity pair
    ent_data = defaultdict(list)
    for key in data1.keys():
        for sentence in data1[key]:
            if filter_sentence(sentence):
                continue
            head = sentence["h"]["id"]
            tail = sentence["t"]["id"]
            newsen = sentence
            newtokens = " ".join(newsen["tokens"]).lower().split(" ")
            newsen["tokens"] = newtokens
            ent_data[head + "#" + tail].append(newsen)
    for key in data2.keys():
        for sentence in data2[key]:
            if filter_sentence(sentence):
                continue
            head = sentence["h"]["id"]
            tail = sentence["t"]["id"]
            newsen = sentence
            newtokens = " ".join(newsen["tokens"]).lower().split(" ")
            newsen["tokens"] = newtokens
            ent_data[head + "#" + tail].append(newsen)
    ll = 0
    list_data = []
    entpair2scope = {}
    for key in ent_data.keys():
        list_data.extend(ent_data[key][0:max_num])
        entpair2scope[key] = [ll, len(list_data)]
        ll = len(list_data)
    return list_data,entpair2scope

def select_similar_data_new_bert(training_data,tokenizer,entpair2scope,topk,max_sen_length_for_select,list_data,config,SimModel,select_thredsold,max_sen_lstm_tokenize,enctokenizer,faissindex,ifnorm,select_num=2):
    selectdata = []
    alladdnum = 0
    has = 0
    nothas = 0
    for onedata in training_data:
        label = onedata[0]
        text = onedata[9]
        headid = onedata[7]
        tailid = onedata[8]
        headindex = onedata[4]
        tailindex = onedata[6]
        onedatatoken, onedatahead, onedatatail = handletoken(text, headindex, tailindex, tokenizer)

        onedicid = headid + "#" + tailid
        tmpselectnum = 0
        if onedicid in entpair2scope:
            has += 1
            thispairnum = entpair2scope[onedicid][1] - entpair2scope[onedicid][0]
            if True:
                alldisforthispair = []
                input_ids = np.zeros((thispairnum + 1, max_sen_length_for_select), dtype=int)
                mask = np.zeros((thispairnum + 1, max_sen_length_for_select), dtype=int)
                h_pos = np.zeros((thispairnum + 1), dtype=int)
                t_pos = np.zeros((thispairnum + 1), dtype=int)
                for index in range(entpair2scope[onedicid][0], entpair2scope[onedicid][1]):
                    oneres = list_data[index]
                    tokens = " ".join(oneres["tokens"])
                    hposstart = oneres["h"]["pos"][0][0]
                    hposend = oneres["h"]["pos"][0][-1]
                    tposstart = oneres["t"]["pos"][0][0]
                    tposend = oneres["t"]["pos"][0][-1]
                    tokenres, headpos, tailpos = handletoken(tokens, [hposstart, hposend], [tposstart, tposend],
                                                             tokenizer)
                    length = min(len(tokenres), max_sen_length_for_select)
                    input_ids[index - entpair2scope[onedicid][0]][0:length] = tokenres[0:length]
                    mask[index - entpair2scope[onedicid][0]][0:length] = 1
                    h_pos[index - entpair2scope[onedicid][0]] = min(headpos, max_sen_length_for_select - 1)
                    t_pos[index - entpair2scope[onedicid][0]] = min(tailpos, max_sen_length_for_select - 1)
                length = min(len(onedatatoken), max_sen_length_for_select)
                input_ids[thispairnum][0:length] = onedatatoken[0:length]
                mask[thispairnum][0:length] = 1
                h_pos[thispairnum] = min(onedatahead, max_sen_length_for_select - 1)
                t_pos[thispairnum] = min(onedatatail, max_sen_length_for_select - 1)
                input_ids = torch.from_numpy(input_ids).to(config["device"])
                mask = torch.from_numpy(mask).to(config["device"])
                h_pos = torch.from_numpy(h_pos).to(config["device"])
                t_pos = torch.from_numpy(t_pos).to(config["device"])
                outputs = SimModel(input_ids, mask)
                indice = torch.arange(input_ids.size()[0])
                h_state = outputs[0][indice, h_pos]
                t_state = outputs[0][indice, t_pos]
                state = torch.cat((h_state, t_state), 1)
                query = state[thispairnum, :].view(1, state.shape[-1])
                toselect = state[0:thispairnum, :].view(thispairnum, state.shape[-1])
                if ifnorm:
                    querynorm = query / query.norm(dim=1)[:, None]
                    toselectnorm = toselect / toselect.norm(dim=1)[:, None]
                    res = (querynorm * toselectnorm).sum(-1)
                else:
                    res = (query * toselect).sum(-1)
                pred = []
                for i in range(res.size(0)):
                    pred.append((res[i], i))
                pred.sort(key=lambda x: x[0], reverse=True)
                selectedindex = []
                tmpselectnum = 0
                prescore= -100.0
                for k in range(len(pred)):
                    thistext = " ".join(list_data[entpair2scope[onedicid][0] + pred[k][1]]["tokens"])
                    if thistext == text:
                        continue
                    if tmpselectnum < topk and pred[k][0] > select_thredsold:
                        selectedindex.append(pred[k][1])
                        prescore = pred[k][0]
                        tmpselectnum += 1
                for onenum in selectedindex:
                    oneres = list_data[entpair2scope[onedicid][0] + onenum]
                    onelabel = label
                    oneneg = [label]
                    onesen = " ".join(oneres["tokens"])
                    hposstart = oneres["h"]["pos"][0][0]
                    hposend = oneres["h"]["pos"][0][-1]
                    tposstart = oneres["t"]["pos"][0][0]
                    tposend = oneres["t"]["pos"][0][-1]
                    tokens, headpos, tailpos = handletoken(onesen, [hposstart, hposend], [tposstart, tposend],
                                                             tokenizer)


                    length = min(len(tokens), max_sen_lstm_tokenize)
                    if (len(tokens) > max_sen_lstm_tokenize):
                        tokens = tokens[:max_sen_lstm_tokenize]
                    newtokens = []
                    for i in range(0, length):
                        newtokens.append(tokens[i])
                    for i in range(length, max_sen_lstm_tokenize):
                        newtokens.append(0)
                    fakefirstent = [554, 555]
                    fakefirstindex = [0, 1]
                    fakesecondent = [665, 666]
                    fakesecondindex = [3, 4]
                    fakeheadid = "fheadid"
                    faketailid = "ftailid"
                    fakerawtext = "fakefake"
                    typelabel = 1  ###positive sample
                    mask = []
                    for i in range(0, length):
                        mask.append(1)
                    for i in range(length, max_sen_lstm_tokenize):
                        mask.append(0)
                    oneseldata = [onelabel, oneneg, newtokens, fakefirstent, fakefirstindex, fakesecondent, fakesecondindex, fakeheadid, faketailid, fakerawtext, length, typelabel, mask]
                    selectdata.append(np.asarray(oneseldata))
                alladdnum += tmpselectnum
        if onedicid not in entpair2scope or tmpselectnum == 0:
            nothas += 1
            topuse = select_num
            input_ids = np.zeros((1, max_sen_length_for_select), dtype=int)
            mask = np.zeros((1, max_sen_length_for_select), dtype=int)
            h_pos = np.zeros((1), dtype=int)
            t_pos = np.zeros((1), dtype=int)
            length = min(len(onedatatoken), max_sen_length_for_select)
            input_ids[0][0:length] = onedatatoken[0:length]
            mask[0][0:length] = 1
            h_pos[0] = min(onedatahead, max_sen_length_for_select - 1)
            t_pos[0] = min(onedatatail, max_sen_length_for_select - 1)

            input_ids = torch.from_numpy(input_ids).to(config["device"])
            mask = torch.from_numpy(mask).to(config["device"])
            h_pos = torch.from_numpy(h_pos).to(config["device"])
            t_pos = torch.from_numpy(t_pos).to(config["device"])
            outputs = SimModel(input_ids, mask)
            indice = torch.arange(input_ids.size()[0])
            h_state = outputs[0][indice, h_pos]
            t_state = outputs[0][indice, t_pos]
            state = torch.cat((h_state, t_state), 1)
            if ifnorm:
                state = state / state.norm(dim=1)[:, None]
            query = state.view(1, state.shape[-1]).cpu().detach().numpy()

            D, I = faissindex.search(query, topuse)
            newtouse = topuse
            newadd = 0
            for i in range(newtouse):
                thisdis = D[0][i]
                newadd += 1
                onenum = I[0][i]
                onelabel = label
                oneneg = [label]
                oneres = list_data[onenum]
                onesen = " ".join(oneres["tokens"])
                hposstart = oneres["h"]["pos"][0][0]
                hposend = oneres["h"]["pos"][0][-1]
                tposstart = oneres["t"]["pos"][0][0]
                tposend = oneres["t"]["pos"][0][-1]
                tokens, headpos, tailpos = handletoken(onesen, [hposstart, hposend], [tposstart, tposend],
                                                       tokenizer)

                length = min(len(tokens), max_sen_lstm_tokenize)
                if (len(tokens) > max_sen_lstm_tokenize):
                    tokens = tokens[:max_sen_lstm_tokenize]
                newtokens = []
                for i in range(0, length):
                    newtokens.append(tokens[i])
                for i in range(length, max_sen_lstm_tokenize):
                    newtokens.append(0)
                fakefirstent = [554, 555]
                fakefirstindex = [0, 1]
                fakesecondent = [665, 666]
                fakesecondindex = [3, 4]
                fakeheadid = "fheadid"
                faketailid = "ftailid"
                fakerawtext = "fakefake"
                typelabel = 1  ###positive sample
                mask = []
                for i in range(0, length):
                    mask.append(1)
                for i in range(length, max_sen_lstm_tokenize):
                    mask.append(0)
                oneseldata = [onelabel, oneneg, newtokens, fakefirstent, fakefirstindex, fakesecondent, fakesecondindex,
                              fakeheadid, faketailid, fakerawtext, length, typelabel,mask]
                selectdata.append(np.asarray(oneseldata))
            alladdnum += newadd
    return selectdata