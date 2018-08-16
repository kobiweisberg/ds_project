from sklearn.datasets import fetch_20newsgroups
from pprint import pprint
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time

# our imports
from utils import *
import preprocess as pp
import vectorizer as vr
import cluster as clst
import analyze as anlz

num_of_documents = 1000
prarameters = [Params(vect_tfidf,clust_hirarchical, aff_euclidean, link_ward, 1e-4, 0.05, 20, num_of_documents)]
for param in prarameters:
    #setup
    t = time.localtime()
    timestamp = time.strftime('%b_%d_%Y_%H%M', t)
    np.random.seed(4)
    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists('results/' + timestamp):
        os.mkdir('results/' + timestamp)



    #params

    newsgroups_train = fetch_20newsgroups(subset='train')
    labels = newsgroups_train.target
    labels_names = newsgroups_train.target_names
    #TODO: consider take only body
    emails = newsgroups_train.data
    emails = emails[:param.max_num]
    number_of_labels = 20 #TODO magic number
    #preprocess
    emails = [pp.clean_text(e) for e in emails]

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


    ##clustering
    if(param.clustering == clust_kneams):
        if(param.affine == 'cosine'):
            #k-means cosine dist
            print('compute kmeans clusters. vector shape:{}'.format(emails_representation.shape[1]))
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
    print('accuracy: ', results.acc)
    print('accuracy random labeling: ', results_rand.acc)

    save_results(param,results)
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