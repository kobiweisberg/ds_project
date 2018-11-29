from sklearn.datasets import fetch_20newsgroups
from pprint import pprint
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
import itertools
import seaborn as sns
import pandas as pd
import nltk
from collections import defaultdict
import re
# our imports
from utils import *




def plot_hist(data,names):
    df = pd.DataFrame({'data':data})
    df['name'] = df['data'].apply(lambda x: names[x])
    sns.countplot(data=df,y='name')
    plt.show()

def plot_lengths_by_cat(lengths,cat,names):
    df = pd.DataFrame({'len': lengths, 'cat':cat})
    df['name'] = df['cat'].apply(lambda x: names[x])
    sns.violinplot(data=df,x='len',y='name')
    plt.xlabel('length')
    plt.ylabel('category')
    plt.show()

def plot_lengths(lengths):
    df = pd.DataFrame({'len': lengths,})
    sns.violinplot(data=df,y='len')
    plt.ylabel('length')
    plt.show()

def print_common_words(data):
    tokens = []
    for doc in data:
        [tokens.append(w) for w in doc.split()]
    print("Total terms: {}".format(len(tokens)))
    num_unique_terms = len(set(tokens))
    print("Unique terms: {}".format(num_unique_terms))
    text = nltk.Text(tokens)
    fdist1 = nltk.probability.FreqDist(text)
    print("Term frequencies:")
    for w in fdist1.most_common(100):
        print("{:>6}, {}".format(w[1], w[0]))

def plot_df(data):
    DF = defaultdict(int)
    for doc in data:
        words = re.findall(r'\w+', doc.lower())
        for word in set(words):
            #if len(word) >= 3 and word.isalpha():
                DF[word] += 1  # defaultdict simplifies your "if key in word_idf: ..." part.
    df = pd.DataFrame.from_dict(DF,orient='index')
    df.rename(index=str, columns={0: 'DF'},inplace=True)
    df['word'] = df.index
    df_sorted = df.sort_values(by=['DF'], ascending=False)
    print(df_sorted.head(100))
    sns.violinplot(data=df,y='DF')
    plt.title('document frequency (df)')
    plt.ylim([-1,np.floor(len(data)/3)])
    plt.ylabel('df')
    plt.show()


if __name__=='__main__':
    newsgroups_train = fetch_20newsgroups(subset='train')
    labels = newsgroups_train.target
    labels_names = newsgroups_train.target_names
    emails = newsgroups_train.data
    number_of_labels = 20  # TODO magic number
    lengths = [len(x.split()) for x in emails]
    plot_df(emails)
    #print_common_words(emails)
    #plot_lengths(lengths)
    #plot_lengths_by_cat(lengths, labels, labels_names)
    #plot_hist(labels,labels_names)