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

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

def create_vec(opt):

    # load data
    val_accuracy_best = 0 # compare the best val to decide if to save new model
    val_bch_sz = 1  # validation batch size - 1 for conv model 30 for regular model
    n_val = 2000  # determine train\val set sizes
    histories_name = 'histories_conv_model' # histories_conv_model or hostories
    load_model_name = opt.load_model_name  # model_conv or model
    print('model name is {}' .format(load_model_name))

    loader = Dataloader(opt)
    opt.vocab_size = len(loader.decoder)  # get vocab size

    encoded_docs, docs_length, labels = loader.only_encoded_docs[:], loader.docs_length[:], loader.labels[:]

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
    model.eval()
    # load model
    checkpoint_path = opt.checkpoint_path + load_model_name
    model.load_state_dict(torch.load(checkpoint_path))
    print("model load from {}".format(checkpoint_path))
    iteration = 0  # set the iteration var, inside the while loop it will change every iteration
    epoch = 1
    vecs_rep_all = torch.zeros(len(docs_length),opt.filter_num)
    for iteration in range(len(docs_length)):
            start = time.time()
            # Load data from train split (0)
            batch_docs, batch_masks, batch_labels, finished = get_batch(encoded_docs, docs_length, labels, opt, iteration)

            #print('Read data:', time.time() - start)

            torch.cuda.synchronize()

            vec_rep = model(batch_docs, batch_masks, batch_labels, iteration, eval_flag=True)
            vecs_rep_all[iteration] = vec_rep

            torch.cuda.synchronize()




        #torch.cuda.empty_cache()

        # Stop if reaching max epochs


opt = opts.parse_opt()
create_vec(opt)
