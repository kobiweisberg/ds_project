import os, sys, re, collections, string, itertools
from operator import itemgetter as at
import numpy as np
#from tqdm import tqdm
sys.path.append("../python")

#Replacing uninformative tokens
digits = re.compile(r"\d[\d\.\$]*")
#sample_doc = digits.sub("<NUM>", sample_doc)
#print (sample_doc)
#Remove punctuation and unknown charactars
not_allowed= re.compile(r"[^\s\w<>_]")
#sample_doc = not_allowed.sub("", sample_doc)
#print (sample_doc)

def clean_text(text):
    return not_allowed.sub("", digits.sub("<NUM>",text.lower()))

if __name__=='__main__':
    test_text = 'Hi hagai 3 *** \nhh jg'
    print(clean_text(test_text))