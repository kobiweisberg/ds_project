from collections import Counter
from tqdm import tqdm
from sklearn.datasets import fetch_20newsgroups
import numpy as np
from tqdm import tqdm
import spacy
from gensim import corpora, models
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *

class DocEncoder(nn.Module):
    def __init__(self, opt):
        super(DocEncoder, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.dropout = opt.dropout
        self.num_classes = opt.num_classes
        self.drop_prob_lm = opt.drop_prob_lm

        # word embedding before lstm
        self.embed = nn.Sequential(nn.Embedding(self.vocab_size, self.input_encoding_size),
                                   nn.ReLU(),
                                   nn.Dropout(self.drop_prob_lm))
        #self.embed = nn.Embedding(self.vocab_size, self.input_encoding_size)
        # Build a LSTM
        self.lstm = nn.LSTM(self.input_encoding_size, self.rnn_size, self.num_layers, batch_first=True)
        # FC layer from lstm output to class size
        self.logit = nn.Linear(self.rnn_size, self.num_classes)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        #self.embed[0].weight.data.uniform_(-initrange, initrange)  # for sequential embedding
        self.embed[0].weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_()),
                Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_()))


    def forward(self, docs, docs_length, labels, iteration, train_flag=True):
        batch_size = docs.size(0)
        state = self.init_hidden(batch_size)
        outputs = torch.zeros(batch_size, self.num_classes)
        #finished = torch.zeros(batch_size)

        for i in range(docs.size(1)):
            it = docs[:, i].clone()
            it = it.type(torch.int64).cuda()
            xt = self.embed(it)
            xt = xt.unsqueeze(1)
            output, state = self.lstm(xt, state)
            output = F.log_softmax(self.logit(output.squeeze(1)))
            for doc_iter in range(batch_size):
                if docs_length[doc_iter] == i+1:
                    outputs[doc_iter] = output[doc_iter].clone()  # +1 because that i is 0-indexed

        loss, accuracy = loss_calc(outputs, labels, batch_size, train_flag)
        return loss, accuracy

def loss_calc(outputs, labels, bsz, train_flag):
    if train_flag:
        # calculate accuracy
        ind = torch.argmax(outputs, dim=1)
        accuracy = torch.as_tensor(sum(ind==labels), dtype=torch.float32)/bsz
        # xe loss
        outputs = outputs[range(bsz), labels]
        loss = -torch.sum(outputs)/bsz

        return loss, accuracy
    else:
        # calculate accuracy
        ind = torch.argmax(outputs, dim=1)
        accuracy = torch.as_tensor(sum(ind==labels), dtype=torch.float32)/outputs.size(0)
        # xe loss
        outputs = outputs[range(outputs.size(0)), labels]
        loss = -torch.sum(outputs)/outputs.size(0)
        return loss, accuracy

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)



class DocVec(nn.Module):
    def __init__(self, opt):
        super(DocVec, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.dropout = opt.dropout
        self.num_classes = opt.num_classes
        self.drop_prob_lm = opt.drop_prob_lm
        self.filter_len = opt.filter_len
        self.filter_num = opt.filter_num

        # word embedding before lstm
        self.embed = nn.Sequential(nn.Embedding(self.vocab_size, self.input_encoding_size),
                                   nn.ReLU(),
                                   nn.Dropout(self.drop_prob_lm))
        #self.embed = nn.Embedding(self.vocab_size, self.input_encoding_size)
        # Build a LSTM
        self.lstm = nn.LSTM(self.input_encoding_size, self.rnn_size, self.num_layers, batch_first=True)
        # FC layer from lstm output to class size

        self.conv = nn.Conv2d(1, self.filter_num, kernel_size=(self.filter_len, self.rnn_size))
        #self.conv = nn.Conv1d(42, self.filter_num, kernel_size=(self.filter_len, self.rnn_size))

        #self.max_pool = nn.AdaptiveMaxPool2d((self.filter_num , 1))
        self.max_pool = nn.AdaptiveMaxPool2d((self.filter_num, 1))  # parameter - output size
        self.logits = nn.Linear(self.filter_num, 20)
        # self.logit_1 = nn.Linear(self.filter_num, 100)
        # self.logit_2 = nn.Linear(100, 50)
        # self.logit_3 = nn.Linear(50, self.num_classes)
        # self.logits = {'logit_1':self.logit_1, 'logit_2':self.logit_2, 'logit_3':self.logit_3}
        self.ReLU = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        #self.embed[0].weight.data.uniform_(-initrange, initrange)  # for sequential embedding
        self.embed[0].weight.data.uniform_(-initrange, initrange)
        self.logits.bias.data.fill_(0)
        self.logits.weight.data.uniform_(-initrange, initrange)
        # for v in self.logits.values():
        #     v.bias.data.fill_(0)
        #     v.weight.data.uniform_(-initrange, initrange)


    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_()),
                Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_()))


    def forward(self, docs, docs_length, labels, iteration, train_flag=True, eval_flag=False):
        batch_size = docs.size(0)
        state = self.init_hidden(batch_size)

        outputs = torch.cuda.FloatTensor(docs.size(1), self.rnn_size).fill_(0) # number of words X rnn_size output
        outputs = outputs.unsqueeze(0)
        #finished = torch.zeros(batch_size)

        for i in range(docs.size(1)):
            it = docs[:, i].clone()
            it = it.type(torch.int64).cuda()
            xt = self.embed(it)
            xt = xt.unsqueeze(1)
            output, state = self.lstm(xt, state)

            outputs[0,i,:] = output.squeeze(0).clone()  # each output batch size is one, each column represent a word
        outputs = self.conv(outputs.unsqueeze(0))
        outputs = self.max_pool(outputs.squeeze(-1))
        if eval_flag == True:
            return torch.transpose(outputs.squeeze(0), 0, 1)
        # outputs = self.ReLU(self.logit_1(torch.t(outputs.squeeze(0))))
        # outputs = self.ReLU(self.logit_2(outputs))
        # outputs = F.log_softmax(self.logit_3(outputs))
        outputs = F.log_softmax(self.logits(torch.t(outputs.squeeze(0))))
        loss, accuracy = loss_conv(outputs, labels, batch_size, train_flag)
        return loss, accuracy


def loss_conv(outputs, labels, bsz, train_flag):
    if train_flag:
        # calculate accuracy
        ind = torch.argmax(outputs, dim=1)
        accuracy = torch.as_tensor(sum(ind==labels.cuda()), dtype=torch.float32)/bsz
        # xe loss
        outputs = outputs[range(bsz), labels]
        loss = -torch.sum(outputs)/bsz

        return loss, accuracy
    else:
        # calculate accuracy
        ind = torch.argmax(outputs, dim=1)
        accuracy = torch.as_tensor(sum(ind==labels.cuda()), dtype=torch.float32)/outputs.size(0)
        # xe loss
        outputs = outputs[range(outputs.size(0)), labels]
        loss = -torch.sum(outputs)/outputs.size(0)
        return loss, accuracy


class MaxSizeList(list):
    def __init__(self, maxlen):
        self._maxlen = maxlen

    def append(self, element):
        self.__delitem__(slice(0, len(self) == self._maxlen))
        super(MaxSizeList, self).append(element)