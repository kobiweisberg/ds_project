from sklearn.datasets import fetch_20newsgroups
from pprint import pprint
import pickle
import preprocess as pp
import numpy as np
import vectorizer as vr
np.random.seed(4)
newsgroups_train = fetch_20newsgroups(subset='train')
#pprint(list(newsgroups_train.target_names))
with open('DB20.pkl','wb') as f:
    pickle.dump(newsgroups_train,f)

labels = newsgroups_train.target
#TODO: consider take only body
emails = newsgroups_train.data
labels_names = newsgroups_train.target_names
idx = np.random.randint(0,100,8)
#print(pp.clean_text(emails[0]))
emails = [pp.clean_text(e) for e in emails]



if __name__=='__main__':
    min_df = 1e-10
    max_df = 1
    files = emails[:5]
    cv,tokenized_emails = vr.tokenizer(files,min_df,max_df)
    # {k: v for k, v in zip(cv.get_feature_names(), sample_vec.toarray()[0]) if v > 0}
    # to get vocabulary names use: cv.get_feature_names()
    voc_names = cv.get_feature_names()
    # to get BOW for each mail use: tokenized_emails.toarray()
    BOW = tokenized_emails.toarray()
    tf_idf = vr.tf_idf(files,min_df,max_df)
    tf_idf_vecs = tf_idf.toarray()
    w2v = vr.BOW_w2v(files)
    pass




    #with open('check.txt','w') as fp:
    #    fp.write('\nNEW NEWS\n\n\n'.join([emails[i] for i in idx]))



