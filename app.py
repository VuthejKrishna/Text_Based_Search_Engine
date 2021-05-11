# change directory to env and execute below command
# source bin/activate

from flask import Flask, render_template, url_for, request
# from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
from num2words import num2words

# import nltk
import os
import string
import numpy as np
import copy
import pandas as pd
import pickle
import re
import math

from load_file import load_file
from preprocess_data import process_data
from build_df import build_DF
from tf_idf import tf_idf
from cosine_similarity import cosine_similarity

app = Flask(__name__)

books_data = load_file()
N = books_data.shape[0]
processed_bookname, processed_text = process_data(books_data)
DF, total_vocab_size, total_vocab = build_DF(N,processed_text,processed_bookname)
tf_idf_val, df = tf_idf(N,processed_text,processed_bookname)

@app.route('/')

def home():
    return render_template('search.html')

@app.route('/search/results', methods=['GET', 'POST'])

def search_request():
    search_term = request.form.get("input")
    Q = cosine_similarity(books_data = books_data,DF = DF, tf_idf = tf_idf_val,total_vocab = total_vocab, total_vocab_size = total_vocab_size, k = 5, query = search_term)
    if len(Q) < 1:
        Qtext = 'No Results matching you query...'
        Qfinal = (Q,Qtext)
    else:
        Qtext = 'Results matching your query...'
        Qfinal = (Q,Qtext)
    return render_template('results.html', res=Qfinal )

if __name__ == "__main__":
    app.run(debug=True)
