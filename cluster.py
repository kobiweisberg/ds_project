from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import numpy as np

def kmeans(vecs,k):
    normed_vecs = normalize(vecs, axis=1, norm='l2')
    kmeans_class = KMeans(n_clusters=k, init='random',random_state=4)
    clusters = kmeans_class.fit_predict(normed_vecs)
    return clusters,kmeans_class