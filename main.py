from sklearn.datasets import fetch_20newsgroups
'''from pprint import pprint
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time'''
import itertools
# our imports
from utils import *
import preprocess as pp
import vectorizer as vr
import cluster as clst
import analyze as anlz
import warnings
import argparse
from LM_vectorizer import batchify,plot_tsne,get_docs_repr

import torch
from torch.autograd import Variable

#import data
from dataloader import *
#from LM_hagai import repackage_hidden

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

parser = argparse.ArgumentParser(description='Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./LSTM0/files/',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='model.pth',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f,map_location='cpu').to(device)
model.eval()

corpus = Dataloader(args)
ntokens = len(corpus.decoder)
num_of_documents = len(corpus.only_encoded_docs)
labels = corpus.labels
pp_docs = [' '.join([corpus.decoder[str(w)] for w in doc]) for doc in corpus.only_encoded_docs]
emails = pp_docs
#emails = corpus.raw_data
#labels = corpus.raw_labels

for example_idx in range(20):
    with open('/home/lab/taitelh/PycharmProjects/ds_project/results_hagai/examples/' + str(example_idx) + '.txt','w') as wf:
        wf.write('Label (pp): %s\n' % corpus.target_names[corpus.labels[example_idx]])
        wf.write('Label (raw): %s\n' % corpus.target_names[corpus.raw_labels[corpus.encoded_docs[example_idx][0]]])
        wf.write('\n------------------------------\n')
        wf.write(pp_docs[example_idx])
        wf.write('\n------------------------------\n')
        wf.write(corpus.raw_data[corpus.encoded_docs[example_idx][0]])
number_of_labels = 20 #TODO magic number
max_df = 0.05
min_df = 1e-4
all_k = [20 , 50, 200]
all_vect = [vect_tfidf, vect_w2v, vect_bow]
assert (len(emails) == len(labels))
print(len(emails))
prarameters = [Params(vect_LM, clust_kneams, aff_euclidean, link_ward, min_df, max_df, 20, num_of_documents)]
'''for vect, k in itertools.product(all_vect,all_k):
    prarameters = prarameters + \
    [
        Params(vect, clust_hirarchical, aff_euclidean, link_ward, min_df, max_df, k, num_of_documents),
        Params(vect, clust_hirarchical, aff_euclidean, link_complete, min_df, max_df, k, num_of_documents),
        Params(vect, clust_hirarchical, aff_euclidean, link_avarage, min_df, max_df, k, num_of_documents),
        Params(vect, clust_kneams, aff_euclidean, link_ward, min_df, max_df, k, num_of_documents),
        Params(vect, clust_hirarchical, aff_cosine, link_complete, min_df, max_df, k, num_of_documents),
        Params(vect, clust_hirarchical, aff_cosine, link_avarage, min_df, max_df, k, num_of_documents),
    ]
for vect, k in itertools.product([vect_tfidf, vect_w2v],all_k):
    prarameters = prarameters + \
    [
        Params(vect, clust_kneams, aff_cosine, link_ward, min_df, max_df, k, num_of_documents),
    ]
    '''
files = os.listdir(os.path.curdir)
for file in files:
    if file.startswith('linkage_table'):
        warnings.warn('warning: csv file already exists')
        print('delete %s?' % file)
        ans = input().lower()
        print(ans)
        if(ans == 'y'):
            os.remove(file)
            print('%s deleted!' % file)


#params




for idx,param in enumerate(prarameters):
    try:
        #setup
        #t = time.localtime()
        #timestamp = time.strftime('%b_%d_%Y_%H%M', t)
        #if not os.path.exists('results'):
        #    os.mkdir('results')
        #if not os.path.exists('results/' + timestamp):
        #    os.mkdir('results/' + timestamp)

        np.random.seed(4)
        print('iter {}/{}'.format(idx,len(prarameters)))

        emails = emails[:param.max_num]

        #preprocess
        #emails = [pp.clean_text(e) for e in emails]
        print('vectorizing...')
        ##vectorizing
        if(param.vectorizing == vect_bow):
            #BOW
            cv,tokenized_emails = vr.tokenizer(emails,param.min_df,param.max_df)
            #{k: v for k, v in zip(cv.get_feature_names(), sample_vec.toarray()[0]) if v > 0}
            # to get vocabulary names use: cv.get_feature_names()
            voc_names = cv.get_feature_names()
            #to get BOW for each mail use: tokenized_emails.toarray()
            emails_representation = tokenized_emails.toarray()
        elif(param.vectorizing == vect_tfidf):
            #TF-IDF
            tf_idf = vr.tf_idf(emails,param.min_df,param.max_df)
            emails_representation = tf_idf.toarray()
        elif(param.vectorizing == vect_w2v):
            #Word2Vec
            emails_representation = vr.BOW_w2v(emails)
        elif param.vectorizing == vect_LM:
            data = batchify(corpus.only_encoded_docs, 1, device)
            emails_representation = get_docs_repr(model, data)
        #anlz.tsne_plot(emails_representation, labels, args.seed)
        print('clustering...')
        ##clustering
        if(param.clustering == clust_kneams):
            if(param.affine == 'cosine'):
                #k-means cosine dist
                clusters,kmns_class = clst.kmeans_cosine_dist(emails_representation,param.k)
            elif(param.affine == 'euclidean'):#k-means euclidean
                clusters,kmns_class = clst.kmeans(emails_representation,param.k)
            else:
                raise ValueError('kmenas is not supported with affinity: ' + param.affine)
        elif(param.clustering == clust_hirarchical):
            #hirarchical
            (clusters, _) = clst.hirarchical(emails_representation, param.k,aff=param.affine,link=param.linkage)
        else:
            raise ValueError('clustering is not supported with: ' + param.clustering)
        #analyze

        random_clst = np.random.randint(0, param.k, param.max_num)
        results = anlz.analyze_clustering(labels[:param.max_num],clusters,number_of_labels)
        results_rand = anlz.analyze_clustering(labels[:param.max_num], random_clst, number_of_labels)
        results_super_class = anlz.analyze_clustering(corpus.super_class_labels[:param.max_num],clusters,number_of_labels)
        #print('accuracy: ', results.acc)
        #print('accuracy random labeling: ', results_rand.acc)
        print('analyzing and saving...')

        acc = save_results(param,results)
        print('accuracy = %f' % results.get_list()[0])
        print('accuracy by super_class= %f' % results_super_class.get_list()[0])
        '''
        #visualization
        plt.figure(1, figsize=(20,20))
        plt.imshow(results.conf_mat[0:number_of_labels-1,:])
        #plt.colorbar()
        plt.savefig('results/' + timestamp + '/conf_mat.png')
    
        plt.figure(2,figsize=(20,20))
        plt.imshow(results_rand.conf_mat[0:number_of_labels-1,:])
        #plt.colorbar()
    
        plt.savefig('results/' + timestamp + '/conf_mat_rand.png')
    
    
        plt.figure(4)
        plt.plot(range(1, 1 + number_of_labels), results.precision, '*r')
        plt.plot(range(1, 1 + number_of_labels), results_rand.precision, '*')
        plt.xticks(range(1,21))
        plt.ylim([0, 1])
        plt.title('precision')
        plt.savefig('results/' + timestamp + '/precision.png')
    
        plt.figure(5)
        plt.plot(range(1, 1 + number_of_labels), results.recall, '*r')
        plt.plot(range(1, 1 + number_of_labels), results_rand.recall, '*')
        plt.xticks(range(1,21))
        plt.ylim([0, 1])
        plt.title('recall')
        plt.savefig('results/' + timestamp + '/recall.png')
    
        #labels histogram
    
        plt.figure(4)
        plt.hist(labels,rwidth=0.85,bins = range(21) )
        plt.xticks(range(20))
        plt.show()
        '''
    except Exception as e:
        print('error: %s' %  e)
        raise


#total_linkage_matrix = anlz.calc_total_linkage_matrix(num_of_documents)
