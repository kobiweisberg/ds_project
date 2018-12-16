
# our imports
from utils import *
import warnings
import argparse

# import data
from dataloader import *


parser = argparse.ArgumentParser(description='Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./LSTM0/files/',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='model.pth',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
args = parser.parse_args()

corpus = Dataloader(args)
labels_names = [corpus.target_names[x] for x in corpus.labels]
super_class_labels_names = corpus.super_class_labels_by_name

ntokens = len(corpus.decoder)
num_of_documents = len(corpus.only_encoded_docs)
labels = corpus.labels
pp_docs = [' '.join([corpus.decoder[str(w)] for w in doc]) for doc in corpus.only_encoded_docs]
emails = pp_docs
# emails = corpus.raw_data
# labels = corpus.raw_labels
examples_dir = create_dir('examples')
tsne_dir = create_dir('tsne')
ncut_dir = create_dir('ncut')
results_dir = create_dir('results')

number_of_labels = 20  # TODO magic number

assert (len(emails) == len(labels))
num_of_documents = len(emails)


def ncut_clustering(ncut_dir,k):
    k=20
    data = load_linkage_table(k,ncut_dir)

    sclust = SpectralClustering(affinity='precomputed', n_clusters=k, random_state=4)
    sclust.fit(data)

    res = analyze.analyze_clustering(labels[:len(sclust.labels_)],sclust.labels_,k)
    print(res.acc)
    print(res.precision)
    print(res.recall)
