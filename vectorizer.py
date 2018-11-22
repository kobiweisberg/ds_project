import collections
#import tqdm
import  nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction import text
from gensim.models import Word2Vec
import numpy as np
from LM_vectorizer import get_docs_repr

def create_voc_by_dist(files):
    word_count = collections.Counter()
    for x in files:
        word_count += collections.Counter(x.split())
    total_word_count = sum(word_count.values())
    #word_count_hist = collections.Counter(word_count.values())
    print ("Total word count: "+str(total_word_count))
    #TODO: consider take only top viewed words
    return word_count


def LSTM_vect(files):
    get_docs_repr(files)

def vocab_coverage(word_count,lb=10,ub=10000):
    word_count_hist = collections.Counter(word_count.values())
    total_word_count = sum(word_count.values())
    words_covered = sum([wc*n for wc, n in word_count_hist.items() if lb<wc<ub])
    corpus_percentage = words_covered/total_word_count
    vector_size = len([1 for wc, n in word_count_hist.items() if lb<wc<ub])
    return "Corpus Coverage: {c:.2f}%\n Vector Size: {v}".format(c=corpus_percentage*100,v=vector_size)

def my_analyzer(txt):
    stemmer = PorterStemmer()
    analyzer = text.CountVectorizer().build_analyzer()
    return [stemmer.stem(w) for w in analyzer(txt)]

def tokenizer(files,min_df=1e-06,max_df=0.05):
    cv = text.CountVectorizer(min_df=min_df, max_df=max_df)#, analyzer=my_analyzer)
    cv.fit(files)
    tokenized_files = cv.transform(files)
    return cv,tokenized_files

def tf_idf(files,min_df=0,max_df=np.inf):
    tfidf = text.TfidfVectorizer(min_df=min_df, max_df=max_df)#, analyzer=my_analyzer)
    tfidf.fit(files)
    tf_idf_vecs = tfidf.transform(files)
    return tf_idf_vecs

def BOW_w2v(files):
    nltk.download('punkt')
    w2v_file = "../../w2v.pickle"
    avg_w2v = AvgWordVectors()
    avg_w2v.fit(w2v_file)
    return np.array(avg_w2v.transform(files))

class AvgWordVectors:
    def __init__(self, weights={}):
        self.weights = weights
    def fit(self, w2v_file):
        self.wv=Word2Vec.load(w2v_file).wv
    def transform(self, docs):
        ret = []
        for doc in docs:
            v = None
            for w in nltk.word_tokenize(doc):
                try:
                    if v is not None:
                        v+=self.weights.get(w, 1.0)*self.wv[w]
                        n+=self.weights.get(w, 1.0)
                    else:
                        v=self.weights.get(w, 1.0)*np.copy(self.wv[w])
                        n=self.weights.get(w, 1.0)
                except KeyError:
                    continue
            try:
                ret.append(v/n)
            except:
                print('ERROR in document #%d' % len(ret))
                raise
        return ret

if __name__=='__main__':
    pass
