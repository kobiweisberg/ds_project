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
#import collections
#import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def train(opt):

    # load data
    val_accuracy_best = 0 # compare the best val to decide if to save new model
    val_loss_best = 10
    val_bch_sz = 1  # validation batch size - 1 for conv model 30 for regular model
    n_val = 2000#8500  # determine train\val set sizes
    histories_name = opt.model_name # histories_conv_model or hostories
    model_name = opt.model_name  # model_conv or model
    print('model name is {} \nwight decay is {}' .format(model_name, opt.weight_decay))
    loader = Dataloader(opt)
    opt.vocab_size = len(loader.decoder)  # get vocab size
    if opt.cnn_model:
        encoded_docs, docs_length, labels = loader.only_encoded_docs[-n_val:], loader.docs_length[-n_val:], loader.labels[-n_val:]
        val_encoded_docs, val_docs_length, val_labels = loader.only_encoded_docs[:-n_val], loader.docs_length[:-n_val], loader.labels[:-n_val]
    else:
        encoded_docs, docs_length, labels = loader.only_encoded_docs[:-n_val], loader.docs_length[:-n_val], loader.labels[:-n_val]
        val_encoded_docs, val_docs_length, val_labels = loader.only_encoded_docs[-n_val:], loader.docs_length[-n_val:], loader.labels[-n_val:]

    # unique, counts = np.unique(val_labels, return_counts=True)  # count occurances from each label
    # prop = dict(zip(unique, counts))
    # plt.scatter(prop.keys(), prop.values())
    #plt.show()
    #collections.Counter(val_labels)  - the same but it isn't by order


    # load model
    if opt.cnn_model:
        model = DocEncoder.DocVec(opt)
    else:
        model = DocEncoder.DocEncoder(opt)
    model.cuda()
    # Assure in training mode
    model.train()
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=opt.learning_rate,weight_decay=opt.weight_decay)
    iteration = 0  # set the iteration var, inside the while loop it will change every iteration
    epoch = 1
    train_loss_history = {}  # for saving loss train results every checkpoint
    train_list_size = n_val/20  # keep the last 200 train loss values
    train_loss_list = DocEncoder.MaxSizeList(train_list_size)  # limited list length, for averaging of the train loss (as batch size is 1)
    train_accuracy_list = DocEncoder.MaxSizeList(train_list_size)
    while True:

        start = time.time()
        # Load data from train split (0)
        batch_docs, batch_masks, batch_labels, finished = get_batch(encoded_docs, docs_length, labels, opt, iteration)
        if batch_docs.shape[1] < 5:
            iteration+=1
            continue

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
        train_loss_list.append(loss.data[0])
        train_accuracy_list.append(accuracy)
        if iteration%100==0:  # print every 100 iterations
            print("iter {} (epoch {}), time/batch = {:.3f}, train_loss = {:.3f}, accuracy = {:.3f}" .format(iteration, epoch, end - start, train_loss, accuracy))

        if (iteration % opt.save_checkpoint_every == 0) and iteration != 0:
    	    # evaluation
            # initialiazition #
            print('starting evaluation on validation set')

            val_loss_sum = 0
            val_accuracy_sum = 0
            val_loss = 0
            val_accuracy = 0
            # original
            #val_iters = math.ceil(n_val / (val_bch_sz*10))  # number of val iterations (n_val is just part of the validation set)
            val_iters=1000
            for val_iter in range(val_iters):
                batch_docs, batch_masks, batch_labels = get_batch(val_encoded_docs[val_iter*val_bch_sz:(val_iter+1)*val_bch_sz],\
                                                                  val_docs_length[val_iter*val_bch_sz:(val_iter+1)*val_bch_sz],\
                                                                  val_labels[val_iter*val_bch_sz:(val_iter+1)*val_bch_sz], opt,\
                                                                  iteration, train_flag=False)
                if batch_docs.shape[1]<5: continue

            #batch_docs, batch_masks, batch_labels = get_batch(val_encoded_docs, val_docs_length, val_labels, opt, iteration, train_flag=False)
                model.eval()
                torch.cuda.synchronize()
                loss, val_accuracy = model(batch_docs, batch_masks, batch_labels, iteration, train_flag=False)
                val_loss_sum += len(batch_labels)*loss.data  # sum of val loss of specific batch and sum it over all val set
                val_accuracy_sum += len(batch_labels)*val_accuracy.data # same as above for accuracy
                if val_iter % 100 == 0:
                    print('validation - iteration {} from {}'  .format(val_iter, val_iters))

                #val_loss = loss.data[0]
            val_loss = val_loss_sum/val_iters  # average over all val set
            val_accuracy = val_accuracy_sum/val_iters # same as above
            torch.cuda.synchronize()
            print("iter {} (epoch {}), val_loss = {:.3f}, val_accuracy = {:.3f}" .format(iteration, epoch, val_loss, val_accuracy))
            model.train()
	        # save history file
            train_loss = sum(train_loss_list)/ train_list_size  # averaging over last train loss results (as batch size is 1)
            accuracy = sum(train_accuracy_list) / train_list_size
            train_loss_history[iteration] = {'train_loss': train_loss, 'train_accuracy': accuracy, 'val_loss':val_loss, 'val_accuracy':val_accuracy, 'epoch': epoch}
            print("iter {} (epoch {}), train_loss = {:.3f}, train_accuracy = {:.3f}".format(iteration, epoch, train_loss, accuracy))

            with open(os.path.join(opt.checkpoint_path, 'histories' + histories_name + '.pkl'), 'wb') as f:
                cPickle.dump(train_loss_history, f)

            # check if the model is the best one
            #if val_accuracy > val_accuracy_best:
            if val_loss < val_loss_best:
                # save model
                checkpoint_path = os.path.join(opt.checkpoint_path, model_name + '.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print("model saved to {}".format(checkpoint_path))
                val_accuracy_best = val_accuracy
            print('finished evaluation on validation set')

        if finished:
            epoch += 1

        iteration += 1
        #torch.cuda.empty_cache()

        # Stop if reaching max epochs
        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break

opt = opts.parse_opt()
train(opt)
