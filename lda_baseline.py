import pandas as pd
# baseline implementation of LDA for comparison purposes in future
from urllib.parse import urlparse
import ast
import numpy as np
from nltk.corpus import stopwords
import gensim
from gensim.utils import simple_preprocess
from gensim import corpora

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

# method to remove stopwords
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

# method to bigrams
def make_bigrams(texts,bigram_mod):
    return [bigram_mod[doc] for doc in texts]

# method for making tri-grams
def make_trigrams(texts,bigram_mod,trigram_mod):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

# method for sentence to words generation
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

# function to compute topics
def lda_complete(text,bigram_min_count,threshold,num_topics):
	"""
	params:

	1. text - list of words
	2. bigram_min_count - minimum frequency to form a bigram
	3. threshold - max number of bigrams
	4. num_topics - number of topics
	"""
	bigram = gensim.models.Phrases(text, min_count=bigram_min_count, threshold=threshold)
	trigram = gensim.models.Phrases(bigram[text], threshold=threshold)
	bigram_mod = gensim.models.phrases.Phraser(bigram)
	trigram_mod = gensim.models.phrases.Phraser(trigram)
	texts  = make_bigrams(remove_stopwords(data_words),bigram_mod)
	id2word = corpora.Dictionary(texts)
	corpus = [id2word.doc2bow(text) for text in texts]
	lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=id2word,num_topics=num_topics)
	return print(lda_model.print_topics())