from sklearn.datasets import fetch_20newsgroups
from pprint import pprint
import pickle
import preprocess as pp
import numpy as np
import vectorizer as vr
import cluster as clst
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time
t = time.localtime()
timestamp = time.strftime('%b_%d_%Y_%H%M', t)
np.random.seed(4)
newsgroups_train = fetch_20newsgroups(subset='train')

labels = newsgroups_train.target
#TODO: consider take only body
emails = newsgroups_train.data
labels_names = newsgroups_train.target_names
idx = np.random.randint(0,100,8)
#print(pp.clean_text(emails[0]))
emails = [pp.clean_text(e) for e in emails]



if __name__=='__main__':
    ###################
    # parameters
    ###################
    min_df = 1e-6#1e-10
    max_df = 0.05#1
    k = 40
    files = emails[:1000]
    ###################
    # algo
    ###################
    print('count words in corpus and tokenize')
    cv,tokenized_emails = vr.tokenizer(files,min_df,max_df)
    # {k: v for k, v in zip(cv.get_feature_names(), sample_vec.toarray()[0]) if v > 0}
    # to get vocabulary names use: cv.get_feature_names()
    voc_names = cv.get_feature_names()
    # to get BOW for each mail use: tokenized_emails.toarray()
    BOW = tokenized_emails.toarray()
    '''
    print('compute tf-idf')
    tf_idf = vr.tf_idf(files,min_df,max_df)
    tf_idf_vecs = tf_idf.toarray()
    print('loading word2vec')
    w2v = vr.BOW_w2v(files)
    '''
    ######TODO: not good, only euclid dist
    print('compute kmeans clusters')
    kmns_clusters,kmns_class = clst.kmeans_cosine_dist(BOW,k)
    print('compute confusion matrix')
    conf_mat = confusion_matrix(labels[:1000], kmns_clusters)
    plt.figure(1)
    plt.imshow(conf_mat)
    plt.show()
    with open(timestamp+'_kmeans.pkl','wb') as wf:
        pickle.dump(conf_mat,wf)

    pass




    #with open('check.txt','w') as fp:
    #    fp.write('\nNEW NEWS\n\n\n'.join([emails[i] for i in idx]))



