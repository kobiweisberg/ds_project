import time

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

def save_results(params, results, file_path = 'a.txt'): #results is a list of [acc, precision,recall]
    t = time.localtime()
    timestamp = time.strftime('%b_%d_%Y_%H%M', t)
    with open(file_path,'a') as ff:
        ff.write('time=' + timestamp + '\n')
        ff.write('params=' + str(params.get_list()) + '\n')
        ff.write('results=' + str(results) + '\n')
