from gensim import corpora
from collections import defaultdict

documents = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]

# word split
doc_text_list = []
for d in documents:
    doc_text_list.append( d.split(' ') )

# get dict
dic = corpora.Dictionary(doc_text_list)

print(dic)# Dictionary(45 unique tokens: ['System', 'Graph', 'machine', 'quasi', 'A']...)

# dict property
print(dic.token2id)# word:word_id, such as {'abc': 1, 'applications': 2, 'computer': 3}
print(dic.dfs)# word_id:word_count, such as {1: 1, 2: 1}



