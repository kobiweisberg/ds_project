from utils import load_linkage_table
from sklearn.cluster import SpectralClustering
import analyze

from sklearn.datasets import fetch_20newsgroups

newsgroups_train = fetch_20newsgroups(subset='train')
labels = newsgroups_train.target
labels_names = newsgroups_train.target_names
#TODO: consider take only body
emails = newsgroups_train.data
number_of_labels = 20 #TODO magic number



k=20
data = load_linkage_table(k)

sclust = SpectralClustering(affinity='precomputed', n_clusters=k, random_state=4)
sclust.fit(data)

res = analyze.analyze_clustering(labels[:100],sclust.labels_,k)
print(res.acc)
print(res.precision)
print(res.recall)
