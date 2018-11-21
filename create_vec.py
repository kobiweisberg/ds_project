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
import numpy as np
from six.moves import cPickle
#import collections
#import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

def create_vec(opt):

    load_model_name = opt.load_model_name  # model_conv or model
    print('model name is {}' .format(load_model_name))

    loader = Dataloader(opt)
    opt.vocab_size = len(loader.decoder)  # get vocab size

    encoded_docs, docs_length, labels = loader.only_encoded_docs[:11], loader.docs_length[:11], loader.labels[:11]



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
    vecs_rep_all = torch.zeros(len(docs_length),opt.filter_num).detach()
    for iteration in range(len(docs_length)):
            start = time.time()
            # Load data from train split (0)
            batch_docs, batch_masks, batch_labels, finished = get_batch(encoded_docs, docs_length, labels, opt, iteration)

            #print('Read data:', time.time() - start)

            torch.cuda.synchronize()

            vec_rep = model(batch_docs, batch_masks, batch_labels, iteration, eval_flag=True)
            vecs_rep_all[iteration] = vec_rep

            torch.cuda.synchronize()
            if iteration % round(len(docs_length)/10)==0:
                print('finished {}/{} that\'s {} % ' .format(iteration+1, len(docs_length), 100*round((iteration+1)/len(docs_length),2)))
    if opt.save_file:  # default is not saving
        np.save('vecs_rep_all.npy', vecs_rep_all.detach().numpy())
    return vecs_rep_all.detach().numpy()

        #torch.cuda.empty_cache()

        # Stop if reaching max epochs


opt = opts.parse_opt()
create_vec(opt)
