import numpy as np
from collections import Counter
import math
from preprocess_data import preprocess
from nltk.tokenize import word_tokenize

def doc_freq(word, DF):
    c = 0
    try:
        c = DF[word]
    except:
        pass
    return c

def cosine_sim(a, b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return cos_sim

def zero_vector(tf_idf,total_vocab,N,total_vocab_size):

    D = np.zeros((N, total_vocab_size))

    for i in tf_idf:
        try:
            ind = total_vocab.index(i[1])
            D[i[0]][ind] = tf_idf[i]
        except:
            pass
    return D

def gen_vector(DF, tokens,books_data, total_vocab):
    
    N = books_data.shape[0]
    
    Q = np.zeros((len(total_vocab)))
    
    counter = Counter(tokens)
    words_count = len(tokens)
    
    for token in np.unique(tokens):
        
        tf = counter[token]/words_count
        df = doc_freq(token, DF)
        idf = math.log((N+1)/(df+1))

        try:
            ind = total_vocab.index(token)
            Q[ind] = tf*idf
        except:
            pass
    return Q


def cosine_similarity(books_data, DF, tf_idf, total_vocab,total_vocab_size, k, query):
    final_dict = dict()

    D = zero_vector(tf_idf,total_vocab, books_data.shape[0],total_vocab_size)
    
    preprocessed_query = preprocess(query)
    tokens = word_tokenize(str(preprocessed_query))

    print("")
    
    d_cosines = []
    
    query_vector = gen_vector(DF, tokens,books_data, total_vocab)

    for d in D:
        cosine_rate = cosine_sim(query_vector, d)
        
        if math.isnan(cosine_rate):
            d_cosines.append(0.0)
        else:
            d_cosines.append(cosine_rate)
        
    out = np.array(d_cosines).argsort()[-k:][::-1]

    cosine_val = sorted(d_cosines, reverse=True)[:k]
    
    # for each in out:
    #     result= np.where(out == each)
    #     case = {each: {'bookname':books_data['bookname'][each], 'author':books_data['author'][each], 'chapter':books_data['chapter'][each], 'similarity':cosine_val[result[0][0]]}}
    #     final_dict.update(case)
    for each in out:
        result= np.where(out == each)
        if cosine_val[result[0][0]] == 0.0:
            pass
        else:
            case = {each: {'bookname':books_data['bookname'][each], 'author':books_data['author'][each], 'chapter':books_data['chapter'][each], 'similarity':cosine_val[result[0][0]]}}
            final_dict.update(case)
    
    return final_dict