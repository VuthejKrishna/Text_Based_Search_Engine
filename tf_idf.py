import numpy as np
from collections import Counter

def doc_freq(word):
    c = 0
    try:
        c = DF[word]
    except:
        pass
    return c

def tf_idf_titles(N,processed_text,processed_bookname):
    doc = 0

    tf_idf_title = {}

    for i in range(N):

        tokens = processed_bookname[i]
        counter = Counter(tokens + processed_text[i])
        words_count = len(tokens + processed_text[i])

        for token in np.unique(tokens):

            tf = counter[token]/words_count
            df = doc_freq(token)
            idf = np.log((N+1)/(df+1)) #numerator is added 1 to avoid negative values

            tf_idf_title[doc, token] = tf*idf

        doc += 1
    return tf_idf_title

def tf_idf(N,processed_text,processed_bookname):
    

    doc = 0

    tf_idf = {}

    for i in range(N):

        tokens = processed_text[i]

        counter = Counter(tokens + processed_bookname[i])
        words_count = len(tokens + processed_bookname[i])

        for token in np.unique(tokens):

            tf = counter[token]/words_count
            df = doc_freq(token)
            idf = np.log((N+1)/(df+1))

            tf_idf[doc, token] = tf*idf

        doc += 1
    
    tf_idf_title = tf_idf_titles(N,processed_text,processed_bookname)
    
    alpha=0.3
    
    for i in tf_idf:
        tf_idf[i] *= alpha
    
    for i in tf_idf_title:
        tf_idf[i] = tf_idf_title[i]
    
    return tf_idf, df