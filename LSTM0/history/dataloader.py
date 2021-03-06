# needed:
# encoded_docs --> [8109 docs] each with 2 dim tuple (doc_id (from 0 to 8108), index of doc words)
# decoder --> ix2word [6720] build like 997: 'nuclear'
# doc_decoder --> len (8109), doc_id (as in encoded_docs) to original index in the dataset, example (999: 1411)
# dataset.target_names --> topics names
# dataset.target --> labels (0 to 19)
# masks --> [8109, 4608] = [number of docs, max doc length]
from sklearn.datasets import fetch_20newsgroups
import numpy as np
from tqdm import tqdm
import spacy
import os
import sys
import torch
#sys.path.append('..')
#sys.path.append('/home/lab/vgilad/PycharmProjects/lda2vec/lda2vec-pytorch')
#from utils import preprocess, get_windows
import pickle
import json
import torch
import torch.utils.data as data
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import math

class Dataloader(data.Dataset):

    def  __init__(self, opt):
        self.opt = opt
        # load data
        cd = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'files'))
        os.chdir(sys.path[-1])

        #self.decoder = np.load('decoder.npy')  # ix2word
        #self.doc_decoder = np.load('doc_decoder.npy')  # doc counter to word index
        self.target_names = np.load('dataset.target_names.npy')  # topics names
        self.labels = np.load('labels.npy')  # labels
        docs_length = np.load('docs_length.npy')  # list of each doc length, can be used to take the right output of the lstm for each doc
        self.docs_length = np.asarray(docs_length)  # for now, it's ndarray
        # self.masks = np.load('masks.npy')  # [8109, 4608] = [number of docs, max doc length]
        #  add new_masks 1 in the location of the sec length and zeros o.w
        self.docs_length = torch.from_numpy(self.docs_length).type(torch.LongTensor) #*******
        # docs_length = docs_length.type(torch.LongTensor)
        """eye = torch.eye(docs_length.shape[0])
        self.new_masks = eye[docs_length]"""  # 1 only in the location of the doc length every thing else is 0

        # [8109 docs] each with 2 dim tuple (doc_id (from 0 to 8108), index of doc words)
        with open('encoded_docs_file', 'rb') as fp:
            self.encoded_docs = pickle.load(fp)

        with open('only_encoded_docs', 'rb') as fp:
            self.only_encoded_docs = pickle.load(fp)

        with open('vocab.json', 'rb') as f:
            self.decoder = json.load(f)

        with open('doc_decoder.json', 'rb') as f:
            self.doc_decoder = json.load(f)

        """used_docs = []
        for val in self.doc_decoder.values():  # get all the documents int use
            used_docs.append(val)

        self.used_docs = used_docs
        self.new_labels = self.labels[used_docs]
        os.chdir = cd"""

        """def cleanup():
            print('Terminating get_batch')
            del self.get_batch
        import atexit
        atexit.register(cleanup)"""


def get_batch(encoded_docs, docs_length, labels, opt, iteration, train_flag=True):
    if train_flag:
        # get the maximum length doc in the current batch
        max_len = 1
        iter_in_epoch = math.floor(len(encoded_docs)/opt.batch_size)
        iteration = (iteration + 1) % (iter_in_epoch + 1)  # to get the iteration in epoch (no absolute)
        start_iter = iteration * opt.batch_size

        #if (iteration+1) % (iter_in_epoch) != 0:
        if iteration != iter_in_epoch:
            #batch_docs = encoded_docs[iteration*opt.batch_size: (iteration+1)*opt.batch_size]  # docs for current iteration, [1] - for taking the doc without doc_id
        # curr_len = np.zeros(opt.batch_size)
            for i in range(iteration * opt.batch_size, (iteration + 1) * opt.batch_size):
                curr_len = len(encoded_docs[i])
                if curr_len > max_len:
                    max_len = curr_len

            # create a torch for the current docs instead of list
            batch_docs = torch.zeros(opt.batch_size, max_len)
            for i in range(iteration * opt.batch_size, (iteration + 1) * opt.batch_size):
                curr_len = len(encoded_docs[i])
                # temp_array[0,j] = np.asarray(encoded_docs[i])
                batch_docs[i - start_iter, :curr_len] = torch.from_numpy(np.asarray(encoded_docs[i]))

            batch_docs_length = docs_length[iteration*opt.batch_size: (iteration+1)*opt.batch_size]  # masks for current iteration
            batch_labels = torch.from_numpy(labels[iteration*opt.batch_size:(iteration+1)*opt.batch_size])  # labels for current iteration
            finished = 0
        else:
            # for last iteration in epoch, take the remain docs
            #batch_docs = encoded_docs[iteration*opt.batch_size:, :][1]  # docs for current iteration
            for i in range(iteration * opt.batch_size, len(encoded_docs)):
                curr_len = len(encoded_docs[i])
                if curr_len > max_len:
                    max_len = curr_len
            # create a torch for the current docs instead of list
            batch_docs = torch.zeros(len(encoded_docs) - start_iter, max_len)
            for i in range(start_iter, len(encoded_docs)):
                curr_len = len(encoded_docs[i])
                batch_docs[i - start_iter, :curr_len] = torch.from_numpy(np.asarray(encoded_docs[i]))

            batch_docs_length = docs_length[iteration*opt.batch_size:]  # masks for current iteration
            batch_labels = torch.from_numpy(labels[iteration*opt.batch_size:])  # labels for current iteration
            finished = 1  # 1 if epoch is finished
        return batch_docs, batch_docs_length, batch_labels, finished
    else: # evaluation
	# get the maximum length doc in the current batch
        max_len = 1
        
        for i in range(len(encoded_docs)):
            curr_len = len(encoded_docs[i])
            if curr_len > max_len:
                max_len = curr_len

        # create a torch for the current docs instead of list
        batch_docs = torch.zeros(len(encoded_docs), max_len)
        for i in range(len(encoded_docs)):
            curr_len = len(encoded_docs[i])
            batch_docs[i, :curr_len] = torch.from_numpy(np.asarray(encoded_docs[i]))
        batch_docs_length = docs_length  # masks for current iteration
        batch_labels = torch.from_numpy(labels)  # labels for current iteration


        return batch_docs, batch_docs_length, batch_labels














