from sklearn.datasets import fetch_20newsgroups
from pprint import pprint
import pickle
newsgroups_train = fetch_20newsgroups(subset='train')
pprint(list(newsgroups_train.target_names))
with open('DB20.pkl','wb') as f:
    pickle.dump(newsgroups_train,f)

