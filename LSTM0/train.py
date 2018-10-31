"""from collections import Counter
from tqdm import tqdm
from sklearn.datasets import fetch_20newsgroups
import numpy as np
from tqdm import tqdm
import spacy
#from gensim import corpora, models
import os
import sys"""
from dataloader import *
import opts
from models import DocEncoder
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import time
from six.moves import cPickle

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def train(opt):

    # load data
    loader = Dataloader(opt)
    opt.vocab_size = len(loader.decoder)  # get vocab size
    encoded_docs, docs_length, labels = loader.only_encoded_docs[:-100], loader.docs_length[:-100], loader.labels[:-100]
    val_encoded_docs, val_docs_length, val_labels =  loader.only_encoded_docs[-100:], loader.docs_length[-100:], loader.labels[-100:]



    # load model
    model = DocEncoder.DocEncoder(opt)
    model.cuda()
    # Assure in training mode
    model.train()
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=opt.learning_rate,weight_decay=opt.weight_decay)
    iteration = 0  # set the iteration var, inside the while loop it will change every iteration
    epoch = 1
    train_loss_history = {}  # for saving loss train results every checkpoint
    while True:
        start = time.time()
        # Load data from train split (0)
        batch_docs, batch_masks, batch_labels, finished = get_batch(encoded_docs, docs_length, labels, opt, iteration)

        #print('Read data:', time.time() - start)

        torch.cuda.synchronize()
        start = time.time()
        optimizer.zero_grad()
        loss, accuracy = model(batch_docs, batch_masks, batch_labels, iteration)
        loss.backward()
        DocEncoder.clip_gradient(optimizer, opt.grad_clip)
        optimizer.step()
        train_loss = loss.data[0]
        torch.cuda.synchronize()
        end = time.time()
        print("iter {} (epoch {}), time/batch = {:.3f}, train_loss = {:.3f}, accuracy = {:.3f}" .format(iteration, epoch, end - start, train_loss, accuracy))

        if (iteration % opt.save_checkpoint_every == 0):
    	    # evaluation
            batch_docs, batch_masks, batch_labels = get_batch(val_encoded_docs, val_docs_length, val_labels, opt, iteration, train_flag=False)
            model.eval()
            torch.cuda.synchronize()
            loss, val_accuracy = model(batch_docs, batch_masks, batch_labels, iteration, train_flag=False)
            val_loss = loss.data[0]
            torch.cuda.synchronize()
            print("iter {} (epoch {}), val_loss = {:.3f}, val_accuracy = {:.3f}" .format(iteration, epoch, val_loss, val_accuracy))
            model.train()
	    # save history file
            train_loss_history[iteration] = {'train_loss': train_loss, 'train_accuracy': accuracy, 'val_loss':val_loss, 'val_accuracy':val_accuracy, 'epoch': epoch}
            with open(os.path.join(opt.checkpoint_path, 'histories' + '.pkl'), 'wb') as f:
                cPickle.dump(train_loss_history, f)
        if finished:
            epoch += 1

        iteration += 1
        #torch.cuda.empty_cache()

        # Stop if reaching max epochs
        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break

opt = opts.parse_opt()
train(opt)
