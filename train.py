# coding=utf-8
import codecs
import sys
import re
import pandas as pd
import numpy as np

from collections import deque  

import torch
import torch.nn as nn
import torch.optim as optim

import torch.utils.data as D
from BiLSTM_ATT import BiLSTM_ATT
        

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


relation2id = {
"Other": 0,
"Cause-Effect": 1,
"Instrument-Agency":2,
"Product-Producer":3,
"Content-Container":4,
"Entity-Origin":5,
"Entity-Destination":6,
"Component-Whole":7,
"Member-Collection":8,
"Message-Topic":9
}

#Preprocesing training data
datas = deque()
labels = deque()
entity1 = deque()
entity2 = deque()
with codecs.open('./TRAIN_FILE.TXT','r','utf-8') as tra:
    linenum = 0
    for line in tra:
        linenum+=1
        if linenum%4==1:
            line = line.split('\t')[1]
            
            word_arr = line[1:-4].split()
            for index in range(len(word_arr)):
                if "<e1>" in word_arr[index]:
                    entity1.append(index)
                elif "<e2>" in word_arr[index]:
                    entity2.append(index)

            line = line.replace("<e1>","")
            line = line.replace("</e1>","")
            line = line.replace("<e2>","")
            line = line.replace("</e2>","")
            line = re.sub(r'[^\w\s]','',line)

            datas.append(line[0:-2].split())
        elif linenum%4==2:
            if line=="Other\r\n":
                labels.append(0)
            else:
                line = line.split('(')
                labels.append(relation2id[line[0]])
            
        else:
            continue

all_words = [d for data in datas for d in data]
sr_allwords = pd.Series(all_words)
sr_allwords = sr_allwords.value_counts()

set_words = sr_allwords.index
set_ids = range(1, len(set_words)+1)

word2id = pd.Series(set_ids, index=set_words)


id2word = pd.Series(set_words, index=set_ids)
word2id["BLANK"]=len(word2id)+1
word2id["UNKNOW"]=len(word2id)+1

max_len = 70

#function for padding sentences
def X_padding(words):
    ids = []
    for word in words:
        if word in word2id:
            ids.append(word2id[word])
        else:
            ids.append(word2id["UNKNOW"])

    if len(ids) >= max_len: 
        return ids[:max_len]
    ids.extend([word2id["BLANK"]]*(max_len-len(ids))) 
    return ids

def pos_padding(index):
    ids=[]    
    for i in range(max_len):
        ids.append(i-index+max_len)
    if max_len-index<0:
        print (index,ids)
    return ids

#padding sentences
x = deque()
pos_e1 = deque()
pos_e2 = deque()
for index in range(len(datas)):
    x.append(X_padding(datas[index]))
    pos_e1.append(pos_padding(entity1[index]))
    pos_e2.append(pos_padding(entity2[index]))

x = np.asarray(x)
y = np.asarray(labels)
pos_e1 = np.asarray(pos_e1)
pos_e2 = np.asarray(pos_e2)

#Preprocesing testing data
datas_t = deque()
labels_t = deque()
entity1_t = deque()
entity2_t = deque()
with codecs.open('./TEST_FILE_FULL.TXT','r','utf-8') as tes:
    linenum = 0
    for line in tes:
        linenum+=1
        if linenum%4==1:
            line = line.split('\t')[1]
            
            word_arr = line[1:-4].split()
            for index in range(len(word_arr)):
                if "<e1>" in word_arr[index]:
                    entity1_t.append(index)
                elif "<e2>" in word_arr[index]:
                    entity2_t.append(index)

            line = line.replace("<e1>","")
            line = line.replace("</e1>","")
            line = line.replace("<e2>","")
            line = line.replace("</e2>","")
            line = re.sub(r'[^\w\s]','',line)

            datas_t.append(line[0:-2].split())
        elif linenum%4==2:
            if line=="Other\r\n":
                labels_t.append(0)
            else:
                line = line.split('(')
                labels_t.append(relation2id[line[0]])
            
        else:
            continue
#padding sentences
x_t = deque()
pos_e1t = deque()
pos_e2t = deque()
for index in range(len(datas_t)):
    x_t.append(X_padding(datas_t[index]))
    pos_e1t.append(pos_padding(entity1_t[index]))
    pos_e2t.append(pos_padding(entity2_t[index]))

x_t = np.asarray(x_t)
y_t = np.asarray(labels_t)
pos_e1t = np.asarray(pos_e1t)
pos_e2t = np.asarray(pos_e2t)


#Storing training and testing data
train = x
position1 = pos_e1
position2 = pos_e2
labels = y

test = x_t
position1_t = pos_e1t
position2_t = pos_e2t
labels_t = y_t
print ("train len", len(train))
print ("test len", len(test) )
print ("word2id len",len(word2id))
print ("relation2id len", len(relation2id))

import torch
import torch.nn as nn
import torch.optim as optim

import torch.utils.data as D
from torch.autograd import Variable
from BiLSTM_ATT import BiLSTM_ATT

#Model Configuration
EMBEDDING_SIZE = len(word2id)+1        
EMBEDDING_DIM = 300
POS_SIZE = 150 
POS_DIM = 25
HIDDEN_DIM = 200
TAG_SIZE = len(relation2id)
BATCH = 10
EPOCHS = 100

config={}
config['EMBEDDING_SIZE'] = EMBEDDING_SIZE
config['EMBEDDING_DIM'] = EMBEDDING_DIM
config['POS_SIZE'] = POS_SIZE
config['POS_DIM'] = POS_DIM
config['HIDDEN_DIM'] = HIDDEN_DIM
config['TAG_SIZE'] = TAG_SIZE
config['BATCH'] = BATCH
config["pretrained"]=False

learning_rate = 0.005

#Using pretrained word embedding
embedding_pre = []
if len(sys.argv)==2 and sys.argv[1]=="pretrained":
    print ("use pretrained embedding")

    config["pretrained"]=True
    word2vec = {}
    with codecs.open('glove.6B.300d.txt','r') as input_data:
        for line in input_data.readlines():
            word2vec[line.split()[0]] = map(eval,line.split()[1:])

    unknow_pre = []
    unknow_pre.extend([1]* EMBEDDING_DIM)
    embedding_pre.append(unknow_pre) 
    for word in word2id:
        if word in word2vec:
            embedding_pre.append(word2vec[word])
        else:
            embedding_pre.append(unknow_pre)

    embedding_pre = np.asarray(embedding_pre)
    print (embedding_pre.shape)

#Constructing dataloader
train = torch.LongTensor(train[:len(train)-len(train)%BATCH])
position1 = torch.LongTensor(position1[:len(train)-len(train)%BATCH])
position2 = torch.LongTensor(position2[:len(train)-len(train)%BATCH])
labels = torch.LongTensor(labels[:len(train)-len(train)%BATCH])
train_datasets = D.TensorDataset(train, position1, position2, labels)
train_dataloader = D.DataLoader(train_datasets,BATCH,True,num_workers=2)


test = torch.LongTensor(test[:len(test)-len(test)%BATCH])
position1_t = torch.LongTensor(position1_t[:len(test)-len(test)%BATCH])
position2_t = torch.LongTensor(position2_t[:len(test)-len(test)%BATCH])
labels_t = torch.LongTensor(labels_t[:len(test)-len(test)%BATCH])
test_datasets = D.TensorDataset(test, position1_t, position2_t, labels_t)
test_dataloader = D.DataLoader(test_datasets,BATCH,True,num_workers=2)

#Building model and setting optimizer and loss fuction
model = BiLSTM_ATT(config,embedding_pre).to(device)
#continue training
# model = torch.load('model/model_01.pkl').to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss(size_average=True)

#start training
for epoch in range(EPOCHS):
    print ("epoch:",epoch)
    acc=0
    total=0
    
    #training
    for sentence, pos1, pos2, tag in train_dataloader:
    	#Load data in Gpu
        sentence, pos1, pos2, tag = sentence.to(device), pos1.to(device), pos2.to(device), tag.to(device)
        y = model(sentence, pos1, pos2)
        loss = criterion(y, tag)      
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
       
        #predicted class
        y = torch.argmax(y,dim=1)

        for y1,y2 in zip(y.data,tag.data):
            if y1==y2:
                acc+=1
            total+=1
    print ("train:",100*float(acc)/total,"%")

    #evaluating
    #10 classes
    count_predict = [0,0,0,0,0,0,0,0,0,0]
    count_total = [0,0,0,0,0,0,0,0,0,0]
    count_right = [0,0,0,0,0,0,0,0,0,0]

    for sentence, pos1, pos2, tag in test_dataloader:
        sentence, pos1, pos2, tag = sentence.to(device), pos1.to(device), pos2.to(device), tag.to(device)
    
        y = model(sentence, pos1, pos2)
        y = torch.argmax(y, dim=1)
        for y1,y2 in zip(y.data, tag.data):
            count_predict[y1] += 1
            count_total[y2] += 1
            if y1 == y2:
                count_right[y1] += 1
    #10 classes
    precision = [0,0,0,0,0,0,0,0,0,0]
    recall = [0,0,0,0,0,0,0,0,0,0]
    for i in range(len(count_predict)):
        if count_predict[i] != 0 :
            precision[i] = float(count_right[i])/count_predict[i]
            
        if count_total[i] != 0:
            recall[i] = float(count_right[i])/count_total[i]
    

    precision = sum(precision)/len(relation2id)
    recall = sum(recall)/len(relation2id)    
    print ("Precision",precision)
    print ("Recall",recall)
    print ("F Score：", (2*precision*recall)/(precision+recall)
    )

    #save model
    if epoch%20==0:
        model_name = "./model/model_epoch"+str(epoch)+".pkl"
        torch.save(model, model_name)
        print (model_name,"has been saved")


torch.save(model, "./model/model_01.pkl")
print ("model has been saved")


