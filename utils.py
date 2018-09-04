import time
import numpy as np
import pandas as pd
import pickle
import csv
import os.path

csv_path = 'results.csv'
pickle_path = 'results.pkl'

vect_bow = 'bow'
vect_tfidf = 'tfidf'
vect_w2v = 'w2v'
vectorizing_options = [vect_bow,vect_tfidf,vect_w2v]
clust_kneams = 'kmeans'
clust_hirarchical = 'hirarchical'
clustering_options = [clust_kneams,clust_hirarchical]
aff_euclidean = 'euclidean'
aff_cosine = 'cosine'
affinity_options = [aff_euclidean, aff_cosine, 'l1', 'l2', 'manhattan'] #explicit affinity is accetped as well
link_ward = 'ward'
link_complete = 'complete'
link_avarage = 'average'
linkage_options = [link_ward, link_complete, link_avarage]

def check_in_options(option,options_list):
    if (not option in options_list):
        raise ValueError(str(option) + 'is not legal option')
class Params:
    def __init__(self, vectorizing, clustering, affine, linkage, min_df, max_df, k, max_num):
            check_in_options(vectorizing,vectorizing_options)
            check_in_options(clustering, clustering_options)
            check_in_options(affine, affinity_options)
            check_in_options(linkage, linkage_options)
            self.vectorizing = vectorizing
            self.clustering = clustering
            self.affine = affine
            self.linkage = linkage
            self.min_df = min_df
            self.max_df = max_df
            self.k = k
            self.max_num = max_num
    def get_list(self):
        return [self.vectorizing, self.clustering, self.affine, self.linkage, self.min_df, self.max_df, self.k, self.max_num]

def save_results(params, results):
    '''
    if(os.path.exists(pickle_path)):
        with open(pickle_path,'rb') as pkl_file:
            pickle_data = pickle.load(pkl_file)
    else:
        pickle_data = []
    #total_results = pd.DataFrame(params.get_list()+results.get_list())
    pickle_data.append({'date':t,'params':params,'results':results})
    with open(pickle_path,'wb') as pkl_file:
        pickle.dump(pickle_data,pkl_file)
    '''
    t = time.localtime()
    timestamp = time.strftime('%b_%d_%Y_%H%M', t)
    with open(csv_path,'a') as ff:
        wr = csv.writer(ff, dialect='excel')
        wr.writerows([[timestamp] + params.get_list() + results.get_list(),\
        [timestamp] + ['precision'] + results.get_list_full_precision(),\
        [timestamp] + ['recall'] + results.get_list_full_recall()])
    csv_table_k = 'linkage_table_' + str(params.k) + '.csv'
    if(os.path.exists(csv_table_k)):
        data = np.genfromtxt(csv_table_k, delimiter=',')
    else:
        data = np.zeros((params.max_num,params.max_num))
    new_data = data + results.linkage_table
    np.savetxt(csv_table_k, new_data, fmt = '%d' ,delimiter=",")


def load_linkage_table(k):
    csv_table_k = 'linkage_table_' + str(k) + '.csv'
    data = np.genfromtxt(csv_table_k, delimiter=',')
    return data


def load_results():
    with open(pickle_path, 'rb') as pkl_file:
        pickle_data = pickle.load(pkl_file)
    return pickle_data
