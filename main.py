
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
from LM_vectorizer import batchify, get_docs_repr
import create_vec

import torch
from torch.autograd import Variable

# import data
from dataloader import *

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return os.path.abspath(dir)
# from LM_hagai import repackage_hidden

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
parser.add_argument('--ncut_only', action='store_true',
                    help='delete old linkage table')
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

corpus = Dataloader(args)
labels_names = [corpus.target_names[x] for x in corpus.labels]
super_class_labels_names = corpus.super_class_labels_by_name
super_class_labels = corpus.super_class_labels
#os.chdir('/home/lab/vgilad/PycharmProjects/lstm_ds_project/18_12')
ntokens = len(corpus.decoder)
num_of_documents = len(corpus.only_encoded_docs)
labels = corpus.labels
pp_docs = [' '.join([corpus.decoder[str(w)] for w in doc]) for doc in corpus.only_encoded_docs]
emails = pp_docs
# emails = corpus.raw_data
# labels = corpus.raw_labels
examples_dir = create_dir('examples')
tsne_dir = create_dir('tsne')
ncut_dir = create_dir('ncut')
results_dir = create_dir('results')

for example_idx in range(200):
    with open(os.path.join(examples_dir,str(example_idx) + '.txt'), 'w') as wf:
        wf.write('Label (pp): %s\n' % corpus.target_names[corpus.labels[example_idx]])
        wf.write('Label (raw): %s\n' % corpus.target_names[corpus.raw_labels[corpus.encoded_docs[example_idx][0]]])
        wf.write('\n------------------------------\n')
        wf.write(pp_docs[example_idx])
        wf.write('\n------------------------------\n')
        wf.write(corpus.raw_data[corpus.encoded_docs[example_idx][0]])
number_of_labels = 20  # TODO magic number
number_of_labels_super = 6  # TODO magic number

max_df = 1.  # 0.05
min_df = 0.  # 1e-4
all_k = [20, 50] #, 200]
all_vect = [vect_tfidf, vect_w2v, vect_bow]
assert (len(emails) == len(labels))
num_of_documents = len(emails)

#prarameters = [Params(vect_tfidf, clust_kneams, aff_euclidean, link_ward, min_df, max_df, 20, num_of_documents,True)]
prarameters = generate_params([vect_gilad], min_df, max_df, all_k, num_of_documents,True)

for file in os.listdir(ncut_dir):
    if file.startswith('linkage_table'):
        warnings.warn('warning: csv file already exists')
        ans = args.ncut_only
        if (not ans):
            os.rename(os.path.join(ncut_dir,file),os.path.join(ncut_dir,'_' + file))
            print('%s renamed!' % file)

# params
if(args.ncut_only):
    print('executing ncut...')
    for k in [20]:
        results_ncut = anlz.ncut_clustering(ncut_dir, k, labels)
        print('ncut accuracy (k=%d) = %f' % (k, results_ncut.get_list()[0]))
        save_results(Params(ncut_stam, ncut_stam, ncut_stam, ncut_stam, min_df, max_df, k, num_of_documents, True), results_ncut,
                     results_ncut, results_dir)
    exit(0)

for idx, param in enumerate(prarameters):
    try:
        labels = corpus.labels

        # setup
        # t = time.localtime()
        # timestamp = time.strftime('%b_%d_%Y_%H%M', t)
        # if not os.path.exists('results'):
        #    os.mkdir('results')
        # if not os.path.exists('results/' + timestamp):
        #    os.mkdir('results/' + timestamp)

        np.random.seed(4)
        print('iter {}/{}'.format(idx+1, len(prarameters)))

        #emails = emails[:param.max_num]

        # preprocess
        # emails = [pp.clean_text(e) for e in emails]
        print('vectorizing...')
        ##vectorizing
        if (param.vectorizing == vect_bow):
            # BOW
            cv, tokenized_emails = vr.tokenizer(emails, param.min_df, param.max_df)
            # {k: v for k, v in zip(cv.get_feature_names(), sample_vec.toarray()[0]) if v > 0}
            # to get vocabulary names use: cv.get_feature_names()
            voc_names = cv.get_feature_names()
            # to get BOW for each mail use: tokenized_emails.toarray()
            emails_representation = tokenized_emails.toarray()
        elif (param.vectorizing == vect_tfidf):
            # TF-IDF
            tf_idf = vr.tf_idf(emails, param.min_df, param.max_df)
            emails_representation = tf_idf.toarray()
        elif (param.vectorizing == vect_w2v):
            # Word2Vec
            emails_representation = vr.BOW_w2v(emails,"w2v.pickle")
        elif param.vectorizing == vect_LM:
            with open(args.checkpoint, 'rb') as f:
                if args.cuda:
                    model = torch.load(f).to(device)
                else:
                    model = torch.load(f, map_location='cpu').to(device)
            model.eval()
            data = batchify(corpus.only_encoded_docs, 1, device)
            emails_representation = get_docs_repr(model, data)
        elif param.vectorizing == vect_gilad:
            # data = batchify(corpus.only_encoded_docs, 1, device)
            import opts

            # take only test documents (fist 1000 are val last 2000 are train)
            ntokens = len(corpus.decoder)
            num_of_documents = len(corpus.only_encoded_docs[1000:-2000])
            labels = corpus.labels[1000:-2000]
            pp_docs = [' '.join([corpus.decoder[str(w)] for w in doc]) for doc in
                       corpus.only_encoded_docs]
            emails = pp_docs[1000:-2000]
            super_class_labels = corpus.super_class_labels[1000:-2000]
            super_class_labels_names = corpus.super_class_labels_by_name[1000:-2000]
            labels_names = [corpus.target_names[x] for x in labels]

            opt = opts.parse_opt()
            emails_representation, labels_not_used = create_vec.create_vec(opt)  # get numpy matrix

            # emails_representation = emails_representation[1000:-2000]
        else:
            raise ValueError('vectorizing is not supported with: ' + param.clustering)

        print('saving tsne...')
        fname = tsne_dir + '/' + param.vectorizing + '.PNG'
        fname_super = tsne_dir + '/' + param.vectorizing + '_super.PNG'
        if not os.path.exists(fname):
            anlz.plot_tsne(emails_representation, (labels_names, super_class_labels_names), seed=4, perplexity=30, alpha=0.3, fpath = (fname,fname_super))
        # anlz.tsne_plot(emails_representation, labels, args.seed)

        print('clustering...')
        ##clustering
        if (param.clustering == clust_kneams):
            if (param.affine == 'cosine'):
                # k-means cosine dist
                clusters, kmns_class = clst.kmeans_cosine_dist(emails_representation, param.k)
            elif (param.affine == 'euclidean'):  # k-means euclidean
                clusters, kmns_class = clst.kmeans(emails_representation, param.k)
                print('kmeans sum of distanses = %f' % kmns_class.inertia_)
            else:
                raise ValueError('kmenas is not supported with affinity: ' + param.affine)
        elif (param.clustering == clust_hirarchical):
            # hirarchical
            (clusters, _) = clst.hirarchical(emails_representation, param.k, aff=param.affine, link=param.linkage)
        else:
            raise ValueError('clustering is not supported with: ' + param.clustering)
        # analyze
        print('analyzing...')
        random_clst = np.random.randint(0, param.k, param.max_num)
        results,results_super_class = anlz.analyze_clustering(labels, clusters, number_of_labels,super_class_labels, number_of_labels_super)
        #results_rand = anlz.analyze_clustering(labels[:param.max_num], random_clst, number_of_labels)
        #results_super_class = anlz.analyze_clustering(corpus.super_class_labels[:param.max_num], clusters,
        #                                              number_of_labels)
        # print('accuracy: ', results.acc)
        # print('accuracy random labeling: ', results_rand.acc)
        print('saving...')

        save_results(param, results, results_super_class, results_dir, ncut_dir)
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
        print('error: %s' % e)
for k in [20]:
    results_ncut = anlz.ncut_clustering(ncut_dir,k,labels)
    print('ncut accuracy (k=%d) = %f' % (k,results_ncut.get_list()[0]))
    save_results(Params(ncut_stam, ncut_stam, ncut_stam, ncut_stam, min_df, max_df, k, param.max_num, True), results, results, results_dir)

# total_linkage_matrix = anlz.calc_total_linkage_matrix(num_of_documents)
