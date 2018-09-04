import numpy as np
from sklearn.metrics import confusion_matrix
from utils import load_results

def evaluate_many2one(mat, num_of_labels):
    mat = mat[:num_of_labels,:]
    correct_pred = np.zeros(num_of_labels)
    for col in mat.T:
        correct_pred[np.argmax(col)] +=np.max(col)

    return correct_pred/np.sum(mat,axis=1)

"""
def evaluate_many2one_conf_mat(mat, num_of_labels):
    mat = mat[:num_of_labels, :]
    labes_conf_mat = np.zeros((num_of_labels,num_of_labels))
    for col in mat.T:
        predicted_label = np.argmax(col)
        labes_conf_mat[:,predicted_label] += col

    return labes_conf_mat


def conf_mat2scores(cm):
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    TN = []
    for i in range(cm.shape[0]):
        temp = np.delete(cm, i, 0)  # delete ith row
        temp = np.delete(temp, i, 1)  # delete ith column
        TN.append(sum(sum(temp)))
    accuracy = np.sum(TP)/np.sum(cm)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return accuracy,precision,recall
###
"""
## analyze functions
def evaluate_many2one_conf_mat(mat, num_of_labels):
    mat = mat[:num_of_labels, :]
    labes_conf_mat = np.zeros((num_of_labels,num_of_labels))
    for col in mat.T:
        predicted_label = np.argmax(col)
        labes_conf_mat[:,predicted_label] += col

    return labes_conf_mat

def conf_mat2scores(cm):
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    TN = []
    for i in range(cm.shape[0]):
        temp = np.delete(cm, i, 0)  # delete ith row
        temp = np.delete(temp, i, 1)  # delete ith column
        TN.append(sum(sum(temp)))
    accuracy = np.sum(TP)/np.sum(cm)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    precision = np.nan_to_num(precision,0)
    recall = np.nan_to_num(recall, 0)
    return accuracy,precision,recall


def calc_linkage_table(clustars):
    N = len(clustars)
    table = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
           table[i,j] = int(clustars[i] == clustars[j])
    return table

class Results:
    def __init__(self,acc, precision, recall, conf_mat, labels_conf_mat, linkage_table):
        self.acc = acc
        self.precision = precision
        self.recall = recall
        self.conf_mat = conf_mat
        self.labels_conf_mat = labels_conf_mat
        self.linkage_table =  linkage_table
    def __str__(self):
        return 'acc = {}, precision = {}, recall = {}'.format(self.acc,np.mean(self.precision),np.mean(self.recall))
    def get_list(self):
        return [self.acc, np.mean(self.precision), np.mean(self.recall)]
    def get_list_full_precision(self):
        return list(self.precision)
    def get_list_full_recall(self):
        return list(self.recall)
def analyze_clustering(labels, clusters, number_of_labels):
    conf_mat = confusion_matrix(labels, clusters)
    linkage_table = calc_linkage_table(clusters)
    labels_conf_mat = evaluate_many2one_conf_mat(conf_mat, number_of_labels)
    acc, precision, recall = conf_mat2scores(labels_conf_mat)
    return Results(acc, precision, recall, conf_mat, labels_conf_mat, linkage_table)


def calc_total_linkage_matrix(number_of_docs,k):
    results = load_results()
    all_link_mat = [res['results'].linkage_table for res in results if (res['params'].get_list()[-1] == number_of_docs)]
    total_link = np.zeros((number_of_docs, number_of_docs))
    for link_mat in all_link_mat:
        total_link = total_link + link_mat
    return total_link


if __name__=='__main__':
    import matplotlib.pyplot as plt
    import pickle
    '''with open('results/','rb') as rf:
        mat = pickle.load(rf)
    ret = evaluate_many2one(mat,20)
    plt.figure(1)
    plt.plot(range(1,21),ret,'*')
    plt.show()'''
    #a = np.array([[38,2,200],[47,53,4]])
    #print(evaluate_many2one_conf_mat(a,2))
    #a = np.array([1, 2, 1, 1, 0, 1, 0])
    #t = calc_linkage_table(a)
    #print(t)
    results = load_results()
    all_link_mat = [res['results'].linkage_table for res in results if res['params'].get_list()[-1]==250]
    total_link = np.zeros((250,250))
    for link_mat in all_link_mat:
        total_link = total_link + link_mat

    print(total_link)
