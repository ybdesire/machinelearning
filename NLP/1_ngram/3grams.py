from nltk.util import ngrams
sentence = 'this is a foo bar sentences and i want to ngramize it'
n = 3
threegrams = ngrams(sentence.split(), n)
for grams in threegrams:
  print(grams)