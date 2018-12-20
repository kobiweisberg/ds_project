import time
import numpy as np
import pandas as pd
import pickle
import csv
import os.path
import itertools

csv_path = 'results.csv'
pickle_path = 'results.pkl'
ncut_stam = 'ncut'
vect_bow = 'bow'
vect_tfidf = 'tfidf'
vect_w2v = 'w2v'
vect_LM = 'lm'
vect_gilad = 'gilad'
vectorizing_options = [vect_bow,vect_tfidf,vect_w2v,vect_LM,vect_gilad,ncut_stam]
clust_kneams = 'kmeans'
clust_hirarchical = 'hirarchical'
clustering_options = [clust_kneams,clust_hirarchical,ncut_stam]
aff_euclidean = 'euclidean'
aff_cosine = 'cosine'
affinity_options = [aff_euclidean, aff_cosine, 'l1', 'l2', 'manhattan',ncut_stam] #explicit affinity is accetped as well
link_ward = 'ward'
link_complete = 'complete'
link_avarage = 'average'
link_single = 'single'
linkage_options = [link_ward, link_complete, link_avarage, link_single,ncut_stam]

SUPER_CLASS_LIST = ['comp','rec','sci','misc','politics','religion']
SUPER_CLASS2IX = {k:v for v,k in enumerate(SUPER_CLASS_LIST)}
SUPER_CLASS_DICT = {'comp.graphics':'comp','comp.os.ms-windows.misc':'comp','comp.sys.ibm.pc.hardware':'comp',
                            'comp.sys.mac.hardware':'comp','comp.windows.x':'comp',
                            'rec.autos':'rec','rec.motorcycles':'rec','rec.sport.baseball':'rec','rec.sport.hockey':'rec',
                            'sci.crypt':'sci','sci.electronics':'sci','sci.med':'sci','sci.space':'sci',
                            'misc.forsale':'misc',
                            'talk.politics.misc':'politics','talk.politics.guns':'politics','talk.politics.mideast':'politics',
                            'talk.religion.misc':'religion','alt.atheism':'religion','soc.religion.christian':'religion'
                            }

def check_in_options(option,options_list):
    if (not option in options_list):
        raise ValueError(str(option) + 'is not legal option')
class Params:
    def __init__(self, vectorizing, clustering, affine, linkage, min_df, max_df, k, max_num, ncut):
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
            self.ncut = ncut
    def get_list(self):
        return [self.vectorizing, self.clustering, self.affine, self.linkage, self.min_df, self.max_df, self.k, self.max_num, str(self.ncut)]

def generate_params(all_vect, min_df, max_df, all_k, num_of_documents,ncut):
    prarameters = []
    for vect, k in itertools.product(all_vect,all_k):
        '''prarameters = prarameters + \
        [
            Params(vect, clust_hirarchical, aff_euclidean, link_ward, min_df, max_df, k, num_of_documents,ncut),
            Params(vect, clust_hirarchical, aff_euclidean, link_complete, min_df, max_df, k, num_of_documents,ncut),
            Params(vect, clust_hirarchical, aff_euclidean, link_avarage, min_df, max_df, k, num_of_documents,ncut),
            #Params(vect, clust_hirarchical, aff_euclidean, link_single, min_df, max_df, k, num_of_documents, ncut),
            Params(vect, clust_kneams, aff_euclidean, link_ward, min_df, max_df, k, num_of_documents,ncut),
            Params(vect, clust_hirarchical, aff_cosine, link_complete, min_df, max_df, k, num_of_documents,ncut),
            Params(vect, clust_hirarchical, aff_cosine, link_avarage, min_df, max_df, k, num_of_documents,ncut),
            #Params(vect, clust_hirarchical, aff_cosine, link_single, min_df, max_df, k, num_of_documents, ncut),
        ]
        '''
    for vect, k in itertools.product(all_vect,all_k):
        prarameters = prarameters + \
        [
            Params(vect, clust_kneams, aff_cosine, link_ward, min_df, max_df, k, num_of_documents,ncut),
        ]
    return prarameters
def save_results(params, results, results_super, results_dir, ncut_dir=None):
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
    with open(results_dir + '/results.csv','a') as ff:
        wr = csv.writer(ff, dialect='excel')
        wr.writerows([[timestamp] + params.get_list() + results.get_list() + results_super.get_list()])
    with open(results_dir + '/results_extended.csv','a') as ff:
        wr = csv.writer(ff, dialect='excel')
        wr.writerows([[timestamp] + params.get_list() + results.get_list() + results_super.get_list(),\
        [timestamp] + ['precision'] + results.get_list_full_precision(),\
        [timestamp] + ['recall'] + results.get_list_full_recall(), \
        [timestamp] + ['precision_super'] + results_super.get_list_full_precision(), \
        [timestamp] + ['recall_super'] + results_super.get_list_full_recall()])

    if(ncut_dir):
        csv_table_k = ncut_dir + '/linkage_table_' + str(params.k) + '.csv'
        if(params.ncut):
            if(os.path.exists(csv_table_k)):
                data = load_linkage_table(params.k, ncut_dir) #read saved matrix to data
            else:
                data = np.zeros((params.max_num,params.max_num)) #initilize matrix with zeros
            new_data = data + results.linkage_table
            np.savetxt(csv_table_k, new_data, fmt = '%d' ,delimiter=",")

def load_linkage_table(k, ncut_dir):
    csv_table_k = ncut_dir + '/linkage_table_' + str(k) + '.csv'
    data = np.genfromtxt(csv_table_k, delimiter=',')
    return data


def load_results():
    with open(pickle_path, 'rb') as pkl_file:
        pickle_data = pickle.load(pkl_file)
    return pickle_data
