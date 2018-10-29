#!/usr/bin/python
# -*- coding: utf-8 -*-

from gensim import corpora, models
from collections import defaultdict

documents = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "minors A survey A survey A"]

# word split
doc_text_list = []
for d in documents:
    doc_text_list.append( d.split(' ') )

# get dict
dic = corpora.Dictionary(doc_text_list)
print(dic)# Dictionary(45 unique tokens: ['System', 'Graph', 'machine', 'quasi', 'A']...)

# get corpus
corpus = [dic.doc2bow(text) for text in doc_text_list]
print(corpus)# corpus is the word:word_count for each sentence

# create tf-idf model
tfidf_model = models.TfidfModel(corpus) 
# get tf-idf features
corpus_tfidf = tfidf_model[corpus]

for tfidf in corpus_tfidf:
    print(tfidf)
