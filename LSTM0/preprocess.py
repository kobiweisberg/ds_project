from collections import Counter
from tqdm import tqdm
from sklearn.datasets import fetch_20newsgroups
import numpy as np
from tqdm import tqdm
import spacy
from gensim import corpora, models
import os
import sys
import pickle
import argparse
import json
import re
from sklearn.feature_extraction import text as txt

def preprocess(docs, nlp, min_length, min_counts, max_counts,number_of_common_to_ignore):
    """Tokenize, clean, and encode documents.

    Arguments:
        docs: A list of tuples (index, string), each string is a document.
        nlp: A spaCy object, like nlp = spacy.load('en').
        min_length: An integer, minimum document length.
        min_counts: An integer, minimum count of a word.
        max_counts: An integer, maximum count of a word.

    Returns:
        encoded_docs: A list of tuples (index, list), each list is a document
            with words encoded by integer values.
        decoder: A dict, integer -> word.
        word_counts: A list of integers, counts of words that are in decoder.
            word_counts[i] is the number of occurrences of word decoder[i]
            in all documents in docs.
    """

    def clean_and_tokenize(doc,lemmatize=False):
        doc = doc.replace('.',' ')
        text = ' '.join(doc.split())  # remove excessive spaces
        #text = ' '.join(text.split('-'))
        #digits = re.compile(r"\d[\d\.\$]*")
        not_allowed = re.compile(r"[^\s\w<>_]")
        text = re.sub(r'[^\w\s]', '', text)
        text_nlp = nlp(text, disable = ['parser','ner'])
        temp = []
        for t in text_nlp:
            if not_allowed.match(t.lower_):
                continue
            elif t.lower_.isdigit():
                temp += ['<nnuumm>']
            else:
                if lemmatize:
                    temp += [t.lemma_]
                else:
                    temp += [t.lower_]
        return temp

        # text = not_allowed.sub("", digits.sub("<num>", text.lower()))
        # #return [t.lemma_ for t in text if t.is_alpha and len(t) > 2 and not t.is_stop]
        # #return [t.lower_ for t in text if t.is_alpha and len(t) > 2 and not t.is_stop]  # remove .lemma and add lower case operation
        # text = [t for t in text if t.is_alpha and not t.is_stop]
        # return text

    tokenized_docs = [(i, clean_and_tokenize(doc)) for i, doc in tqdm(docs)]

    # remove short documents
    n_short_docs = sum(1 for i, doc in tokenized_docs if len(doc) < min_length)
    tokenized_docs = [(i, doc) for i, doc in tokenized_docs if len(doc) >= min_length]
    print('number of removed short documents:', n_short_docs)

    # remove some tokens
    counts = _count_unique_tokens(tokenized_docs)  # counts the quantity of each word
    tokenized_docs = _remove_tokens(tokenized_docs, counts, min_counts, max_counts,number_of_common_to_ignore)  # docs after removal of words with too low\high quantity
    n_short_docs = sum(1 for i, doc in tokenized_docs if len(doc) < min_length)  # number of short docs
    tokenized_docs = [(i, doc) for i, doc in tokenized_docs if len(doc) >= min_length]  # docs > min_length
    print('number of additionally removed short documents:', n_short_docs)

    counts = _count_unique_tokens(tokenized_docs)  # quantity of words that shown in doc in the right amount
    encoder, decoder, word_counts = _create_token_encoder(counts)  # w2ix

    print('\nminimum word count number:', word_counts[-1])
    print('this number can be less than MIN_COUNTS because of document removal')

    encoded_docs = _encode(tokenized_docs, encoder)  # all the doc is encoded as indexes instead of words using ix2word
    return encoded_docs, decoder, word_counts, encoder


def _count_unique_tokens(tokenized_docs):
    tokens = []
    for i, doc in tokenized_docs:
        tokens += doc
    return Counter(tokens)


def _encode(tokenized_docs, encoder):
    return [(i, [encoder[t] for t in doc]) for i, doc in tokenized_docs]


def _remove_tokens(tokenized_docs, counts, min_counts, max_counts, number_of_common_to_ignore):
    """
    Words with count < min_counts or count > max_counts
    will be removed.
    """
    total_tokens_count = sum(
        count for token, count in counts.most_common()
    )
    print('total number of tokens:', total_tokens_count)

    # unknown_tokens_count = sum(
    #     count for token, count in counts.most_common()
    #     if count < min_counts or count > max_counts
    # )
    # print('number of tokens to be removed:', unknown_tokens_count)

    # keep = {}
    # for token, count in counts.most_common():
    #     keep[token] = count >= min_counts and count <= max_counts
    #
    # return [(i, [t for t in doc if keep[t]]) for i, doc in tokenized_docs]
    # #####################3
    keep = {}
    counter = 0
    for token, count in counts.most_common():
        if counter < number_of_common_to_ignore:
            if(token!='<nnuumm>'):
                keep[token] = 'more'
            else:
                keep[token] = 'keep'
        elif count < min_counts:
            keep[token] = 'less'
        #elif count > max_counts:
        #    keep[token] = 'more'
        else:
            keep[token] = 'keep'
        counter += 1
    ret = []
    for i, doc in tokenized_docs:
        temp = []
        for t in doc:
            if keep[t] == 'keep':
                temp += [t]
            elif keep[t] == 'less':
                temp += ['<UNK_less>']
            else:
                temp += ['<UNK_more>']
        ret.append((i, temp))
    return ret


# convert unk_more to unk_less - if we want to do so
def convert_more_to_less(ret):
    converted_i = []
    for i,text_list in ret:
        converted = [x if x!='<UNK_more>' else '<UNK_less>' for x in text_list]
        converted_i += (i, converted)
    return converted_i

def _create_token_encoder(counts):

    total_tokens_count = sum(
        count for token, count in counts.most_common()
    )
    print('total number of tokens:', total_tokens_count)

    encoder = {}
    decoder = {}
    word_counts = []
    i = 0

    for token, count in counts.most_common():  # vocab
        # counts.most_common() is in decreasing count order
        encoder[token] = i
        decoder[i] = token
        word_counts.append(count)
        i += 1

    return encoder, decoder, word_counts

def get_windows(doc, hws=5):
    """
    For each word in a document get a window around it.

    Arguments:
        doc: a list of words.
        hws: an integer, half window size.

    Returns:
        a list of tuples, each tuple looks like this
            (word w, window around w),
            window around w equals to
            [hws words that come before w] + [hws words that come after w],
            size of the window around w is 2*hws.
            Number of the tuples = len(doc).
    """
    length = len(doc)
    assert length > 2*hws, 'doc is too short!'

    inside = [(w, doc[(i - hws):i] + doc[(i + 1):(i + hws + 1)]) for i, w in enumerate(doc[hws:-hws], hws)]

    # for words that are near the beginning or
    # the end of a doc tuples are slightly different
    beginning = [(w, doc[:i] + doc[(i + 1):(2*hws + 1)])
                 for i, w in enumerate(doc[:hws], 0)]

    end = [(w, doc[-(2*hws + 1):i] + doc[(i + 1):])
           for i, w in enumerate(doc[-hws:], length - hws)]

    return beginning + inside + end


#################################################################
def main(params):
    if params['df']:
        print('df is ON')
        sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'df_files_max_df_{}_min_df_{}'.format(params['max_df'],params['min_df'])))
    else:
        sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'files'))
    #sys.path.append('..')
    #sys.path.append('/home/lab/vgilad/PycharmProjects/lda2vec/lda2vec-pytorch')
    #from utils import preprocess, get_windows


    MIN_COUNTS = 20
    MAX_COUNTS = 2000 # 1800
    MAX_COUNTS_PRECETAGE = 0.01
    number_of_common_to_ignore = 100 #the 25 most common words will be replaced to <UNK_more>
    # words with count < MIN_COUNTS
    # and count > MAX_COUNTS
    # will be removed
    MIN_NON_ENK_WORDS = 5
    #MIN_LENGTH = 1
    MIN_LENGTH = 8 # 15
    # minimum document length
    # (number of words)
    # after preprocessing

    # half the size of the context around a word
    #HALF_WINDOW_SIZE = 5
    # it must be that 2*HALF_WINDOW_SIZE < MIN_LENGTH

    # load nlp model

    nlp = spacy.load('en')
    #nlp = None ## TODO:spacy
    # Load dataset

    #dataset = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    dataset = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    docs = dataset['data']
    raw_data = docs

    # number of documents
    len(docs)
    num_of_words = [len(d) for d in docs]
    tot_num_of_words = sum(num_of_words)
    # store an index with a document
    docs = [(i, doc) for i, doc in enumerate(docs)]

    # preprocess dataset and create windows1
    print(np.round(MAX_COUNTS_PRECETAGE*tot_num_of_words))
    encoded_docs, decoder, word_counts, encoder = preprocess(docs, nlp, MIN_LENGTH, MIN_COUNTS, np.round(MAX_COUNTS_PRECETAGE*tot_num_of_words),number_of_common_to_ignore)


    new_indxs = []
    for i, j in encoded_docs:
        if ((len(j) - j.count(0) - j.count(1) - j.count(2)) > MIN_NON_ENK_WORDS):
            if (len(j) > MIN_LENGTH):
                new_indxs.append(i)  # list of the encoded docs without the doc id
    raw_labels = dataset.target
    only_encoded_docs = [ed[1] for ed in encoded_docs if ed[0] in new_indxs]

    if params['df']:
        new_new_indxs = []
        only_from_pp = []
        pp_docs = [' '.join([decoder[w] for w in doc]) for doc in only_encoded_docs]
        cv = txt.CountVectorizer(min_df=params['min_df'], max_df=params['max_df'])  # , analyzer=my_analyzer)
        cv.fit(pp_docs)
        analyzer = cv.build_analyzer()
        for i,d in zip(new_indxs,pp_docs):
            d_df = ' '.join([word for word in analyzer(d) if word not in cv.stop_words_])
            if len(d_df.split()) > MIN_NON_ENK_WORDS:
                only_from_pp.append([encoder[x] for x in d_df.split()])
                new_new_indxs.append(i)
        new_indxs = new_new_indxs
        #only_encoded_docs = [ed[1] for ed in encoded_docs if ed[0] in new_indxs]
        encoded_docs = [(idx, ed) for ed, idx in zip(only_from_pp, new_indxs)]
        only_encoded_docs = [ed[1] for ed in encoded_docs if ed[0] in new_indxs]
    labels = []
    # encoded_docs_cleaned = []
    for i in range(len(new_indxs)):  # take only the labels for the docs we are going to use
        labels.append(dataset.target[new_indxs[i]])
    #encoded_docs = [(idx,ed) for ed,idx in zip(only_encoded_docs,new_indxs)]

    #only_encoded_docs = []
    #for i, j in encoded_docs_cleaned:
    #    only_encoded_docs.append(j)  # list of the encoded docs without the doc id
    # new ids will be created for the documents.
    # create a way of restoring initial ids:
    doc_decoder = {i: doc_id for i, (doc_id, doc) in enumerate(encoded_docs)}  # now we don't have all the docs (len < 15) so new connection between original doc id to running index

    """data = []
    # new ids are created here
    for index, (_, doc) in tqdm(enumerate(encoded_docs)):
        windows = get_windows(doc, HALF_WINDOW_SIZE)
        # index represents id of a document,
        # windows is a list of (word, window around this word),
        # where word is in the document
        data += [[index, w[0]] + w[1] for w in windows]

    data = np.array(data, dtype='int64')

    # a row in 'data' contains:
    # id of a document, id of a word in this document, a window around this word
    # 1 + 1 + 10
    data.shape[1]

    # number of windows (equals to the total number of tokens)
    data.shape[0]"""

    #get unigram distribution
    word_counts = np.array(word_counts)
    unigram_distribution = word_counts/sum(word_counts)
    #prepare word vectors

    vocab_size = len(decoder)  # ix to word

    # get the max doc length
    max_length = 0
    doc_ind = []
    for ind, doc in encoded_docs:
        doc_ind.append(ind) # list of indices of our docs
        if max_length < len(doc):
            max_length = len(doc)


    # create masks
    masks = np.zeros([len(doc_decoder), max_length])  # shape [# docs, max length]
    for ind, doc in enumerate(encoded_docs):
        masks[ind,:len(doc[1])]=1

    # save to a list only the labels of the docs that we are going to use
    """i=0
    labels = []
    for ind, label in enumerate(dataset.target):
        if label in doc_ind:
            labels.append(label)
            i += 1"""


    # save data
    try:
        os.makedirs(sys.path[-1])
    except OSError:
        pass
    os.chdir(sys.path[-1])
    json.dump(decoder, open(params['output_vocab'], 'w'))
    json.dump(encoder, open(params['encoder'], 'w'))
    #np.save('decoder.npy', decoder)  # ix2word
    json.dump(doc_decoder, open(params['output_doc_decoder'], 'w'))
    #np.save('doc_decoder.npy', doc_decoder)  # doc counter to word index
    np.save('dataset.target_names.npy', dataset.target_names)  # topics names
    np.save('labels.npy', labels)  # labels
    np.save('masks.npy', masks)  # [8109, 4608] = [number of docs, max doc length]
    docs_length = np.sum(masks, 1)
    np.save('docs_length.npy', docs_length)  # list of each doc length, can be used to take the right output of the lstm for each doc

    # [8109 docs] each with 2 dim tuple (doc_id (from 0 to 8108), index of doc words)
    with open('encoded_docs_file', 'wb') as fp:
        pickle.dump(encoded_docs, fp)

    with open('only_encoded_docs', 'wb') as fp:
        pickle.dump(only_encoded_docs, fp)

    with open('raw_data', 'wb') as fp:
        pickle.dump(raw_data, fp)

    with open('raw_labels', 'wb') as fp:
        pickle.dump(raw_labels, fp)


if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  # input json
  parser.add_argument('--output_vocab', default='vocab.json', help='output json file')
  parser.add_argument('--output_doc_decoder', default='doc_decoder.json', help='output json file')
  parser.add_argument('--encoder', default='encoder.json', help='encoder json file')
  parser.add_argument('-l-','--lemma',help='use lemmas instead of words',action='store_true')
  parser.add_argument('--df',help='remove word > max_df or < min_df',action='store_true')
  parser.add_argument('--min_df', help='min_df', type=float, default=1e-4)
  parser.add_argument('--max_df', help='max_df', type=float, default=0.05)


  args = parser.parse_args()
  params = vars(args)  # convert to ordinary dict
  main(params)

