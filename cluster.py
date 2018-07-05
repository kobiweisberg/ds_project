from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import numpy as np
from kmeans_cosine_distance import KMeans as KMeans_cos_dist

def kmeans(vecs,k):
    normed_vecs = normalize(vecs, axis=1, norm='l2')
    kmeans_class = KMeans(n_clusters=k, init='random',random_state=4)
    clusters = kmeans_class.fit_predict(normed_vecs)
    return clusters,kmeans_class

def kmeans_cosine_dist(vecs,k):
    kmeans_class = KMeans_cos_dist(k,list(vecs))
    kmeans_class.main_loop()
    return kmeans_class.indexes,kmeans_class