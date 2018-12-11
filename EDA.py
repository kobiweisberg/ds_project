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
    #sns.violinplot(data=df,x='len')
    sns.distplot(df['len'], hist=False)
    plt.ylabel('Probabilty')
    plt.xlabel('length')
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

def tf_vs_df(data):
    DF = defaultdict(int)
    for doc in data:
        words = re.findall(r'\w+', doc.lower())
        for word in set(words):
            # if len(word) >= 3 and word.isalpha():
            DF[word] += 1  # defaultdict simplifies your "if key in word_idf: ..." part.
    df = pd.DataFrame.from_dict(DF, orient='index')
    df.rename(index=str, columns={0: 'DF count'}, inplace=True)
    df['DF percentage'] = df['DF count']/len(data)
    print(max(df['DF percentage']))
    df['word'] = df.index

    TF = defaultdict(int)
    for doc in data:
        words = re.findall(r'\w+', doc.lower())
        for word in words:
            # if len(word) >= 3 and word.isalpha():
            TF[word] += 1  # defaultdict simplifies your "if key in word_idf: ..." part.
    df_tf = pd.DataFrame.from_dict(TF, orient='index')
    df_tf.rename(index=str, columns={0: 'TF'}, inplace=True)
    df_tf['word'] = df_tf.index
    assert (len(df_tf) == len(df))

    tot_df = pd.merge(df,df_tf,on='word')
    assert (len(tot_df)==len(df))
    tot_df = tot_df.reset_index()
    sorted_by_tf = tot_df.sort_values(by='TF', ascending=False)
    print(sorted_by_tf[['word','DF count','DF percentage','TF']].head(10))
    sorted_by_df = tot_df.sort_values(by='DF count', ascending=False)
    print(sorted_by_df[['word','DF count','DF percentage','TF']].head(10))

    print(tot_df[tot_df['TF']<20][['word','DF count','DF percentage','TF']])
    print(len(tot_df[tot_df['TF'] < 20][['word', 'DF count', 'DF percentage', 'TF']]))

    plt.figure()
    sns.distplot(tot_df['DF percentage'], hist=False)
    plt.ylabel('Probabilty')
    plt.xlabel('df')
    plt.show()

    plt.figure()
    sns.distplot(tot_df['TF'], hist=False)
    plt.ylabel('Probabilty')
    plt.xlabel('tf')
    plt.show()

if __name__=='__main__':
    newsgroups_train = fetch_20newsgroups(subset='train')
    labels = newsgroups_train.target
    labels_names = newsgroups_train.target_names
    emails = newsgroups_train.data
    number_of_labels = 20  # TODO magic number

    ###super_class
    super_class_target_names = SUPER_CLASS_LIST
    super_class2_ix = SUPER_CLASS2IX
    super_class_dict = SUPER_CLASS_DICT
    super_class_labels_by_name = [SUPER_CLASS_DICT[labels_names[l]] for l in labels]
    super_class_labels = [SUPER_CLASS2IX[name] for name in super_class_labels_by_name]

    ### hists
    plot_hist(labels, labels_names)
    plot_hist(super_class_labels, super_class_target_names)

    ### lengths
    lengths = [len(x.split()) for x in emails]
    print(max(lengths))
    plot_lengths(lengths)
    plot_lengths_by_cat(lengths, labels, labels_names)
    plot_lengths_by_cat(lengths, super_class_labels, super_class_target_names)

    #print_common_words(emails)
    #plot_df(emails)

    tf_vs_df(emails)