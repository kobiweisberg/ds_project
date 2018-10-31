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







