from sklearn.metrics import confusion_matrix
from sklearn.cluster import SpectralClustering
from utils import load_results, load_linkage_table
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE as tsne

#import dataframe as df
def plot_tsne(high_dim_repr,labels,seed=4,perplexity=30,alpha=0.3,fpath = None):
    if not(isinstance(labels,list) or isinstance(labels,tuple)):
        raise ValueError('labels can be only list of lables or tuple of lists, got {}'.format(type(labels)))
    df = genrate_tsne(high_dim_repr,seed=seed,perplexity=perplexity)
    plot_tsne_df(df,labels = labels,alpha=alpha,fpath = fpath)
    return df

def genrate_tsne(high_dim_repr,seed=4,perplexity=30):
    print('compute tsne with perplexity {} and seed {}'.format(perplexity, seed))
    tsne_components = tsne(n_components=2, perplexity=perplexity, random_state=seed)
    transformed = tsne_components.fit_transform(high_dim_repr)
    df = pd.DataFrame()

    df['c1'] = transformed[:, 0]
    df['c2'] = transformed[:, 1]
    return df

def plot_tsne_df(df,labels,fpath = None):
    if isinstance(labels,tuple):
        if ( (fpath is not None) and ( len(fpath) != len(labels))):
            raise ValueError('several sets except to list of pathes')
        print('tsne by several sets of labels')
        for i,lst in enumerate(labels):
            print('label set #{}'.format(i))
            print(len(lst))
            df['label_{}'.format(i)] = lst
            print('arrived to tsne')
            plt.figure(i+1000)
            ax = plt.subplot(111)
            sns.scatterplot(data=df, x='c1', y='c2', hue='label_{}'.format(i), alpha=alpha)
            print('arrived to tsne2')
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            if(fpath):
                plt.savefig(fpath[i])
            else:
                plt.show()
            #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        #plt.show()
        #plt.savefig(fname)
    #df = pd.DataFrame({'label':labels})
    elif isinstance(labels,list):
        print('tsne by one set of labels')
        df['label']= labels
        sns.scatterplot(data=df, x='c1', y='c2', hue='label_1', alpha=alpha)
        ax = plt.subplot(111)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        if(fpath):
            plt.savefig(fpath)
        else:
            plt.show()
    else:
        raise ValueError('labels can be only list of lables or tuple of lists, got {}'.format(type(labels)))

'''def plot_embedding(X,y, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], '.',
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    ''''''if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)''''''
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

def tsne_plot(X,y,random_state=0):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=random_state)
    X_tsne = tsne.fit_transform(X)
    #Xdf=df.DataFrame(data=X_tsne,index=y)
    colormap = cm.viridis
    colorlist = [colors.rgb2hex(colormap(i)) for i in np.linspace(0, 0.9, len(set(y))+1)]
    fig, ax = plt.subplots()
    for i,c in enumerate(colorlist):
            ax.scatter(X_tsne[y==i, 0], X_tsne[y==i, 1], s=50, linewidth=0.1,c=c)
    #plot_embedding(X_tsne,y,"t-SNE")
    plt.show()'''

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

def analyze_clustering(labels, clusters, number_of_labels, labels_super=None, number_of_labels_super=None, calc_linkage = True):
    conf_mat = confusion_matrix(labels, clusters)
    labels_conf_mat = evaluate_many2one_conf_mat(conf_mat, number_of_labels)
    acc, precision, recall = conf_mat2scores(labels_conf_mat)
    if(labels_super):
        conf_mat_super = confusion_matrix(labels_super, clusters)
        labels_conf_mat_super = evaluate_many2one_conf_mat(conf_mat_super, number_of_labels_super)
        acc_super, precision_super, recall_super = conf_mat2scores(labels_conf_mat_super)
    if(calc_linkage):
        linkage_table = calc_linkage_table(clusters)
    else:
        linkage_table = None
    if (labels_super):
        return (Results(acc, precision, recall, conf_mat, labels_conf_mat, linkage_table),Results(acc_super, precision_super , recall_super, conf_mat_super, labels_conf_mat_super, None))
    else:
        return Results(acc, precision, recall, conf_mat, labels_conf_mat, linkage_table)


def calc_total_linkage_matrix(number_of_docs,k):
    results = load_results()
    all_link_mat = [res['results'].linkage_table for res in results if (res['params'].get_list()[-1] == number_of_docs)]
    total_link = np.zeros((number_of_docs, number_of_docs))
    for link_mat in all_link_mat:
        total_link = total_link + link_mat
    return total_link


def ncut_clustering(ncut_dir, k, true_labels):
    try:
        data = load_linkage_table(k, ncut_dir)
    except IOError:
        raise ValueError('error loading linkage table from %s with k=%d.\n maybe you should use --ncut_only to keep the name of the file.' % (ncut_dir,k))

    sclust = SpectralClustering(affinity='precomputed', n_clusters=k, random_state=4)
    sclust.fit(data)

    return analyze_clustering(true_labels[:len(sclust.labels_)], sclust.labels_, 20)


if __name__=='__main__':
    '''import matplotlib.pyplot as plt
    import pickle
    with open('results/','rb') as rf:
        mat = pickle.load(rf)
    ret = evaluate_many2one(mat,20)
    plt.figure(1)
    plt.plot(range(1,21),ret,'*')
    plt.show()
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

    print(total_link)'''
    plot_tsne(np.array([[1,2,3],[2,5,6],[-8,-2,-1]]),[1, 1, 2])
