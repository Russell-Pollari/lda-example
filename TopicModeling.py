#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 13:25:18 2020

@author: minaekramnia
"""

import pandas as pd
from pandas import read_excel
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk
nltk.download('wordnet')
from gensim import corpora, models
import matplotlib.pyplot as plt


#Reading Data in to DAtaframe
#Data Path
data = pd.read_excel('/Users/minaekramnia/Downloads/SampleData_IEGKC_DS_STC_Test.xlsx')
data_text = data[['Pragraph']]
data_text['index'] = data_text.index
documents = data_text

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

processed_docs = documents['Pragraph'].map(preprocess)

#Bag of words on the dataset
#creating a dic that convert indexes into words
dictionary = gensim.corpora.Dictionary(processed_docs)
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
#each word has an index - looping through each doc, converting into bow representation
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

#TF-IDF Model
tfidf = models.TfidfModel(bow_corpus)   
corpus_tfidf = tfidf[bow_corpus]

#Running LDA using Bag of Words
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=5, id2word=dictionary, passes=2, workers=2)

#Topics generated:
for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))
    
#Running LDA using TF-IDF
lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=5, id2word=dictionary, passes=2, workers=4)

#Topics generated_TF-IDF
for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))

topic_label = {0: 'Government', 1: 'Public Health', 2: 'Financial Planning', 3: 'Water Sector', 4: 'Policy Implementation'} 
print(topic_label)
    
#doc_label = topic_label[i]
    
#for index, score in sorted(lda_model[bow_corpus[1]], key=lambda tup: -1*tup[1]):
#    print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 5)))

#Performance evaluation by classifying sample document using LDA TF-IDF model
#for index, score in sorted(lda_model_tfidf[bow_corpus[1]], key=lambda tup: -1*tup[1]):
#    print("\nScore: {}\t \nTopic: {}".format(score, lda_model_tfidf.print_topic(index, 5)))  

topics_doc=[]
for doc in bow_corpus:
    topics_doc.append(lda_model_tfidf.get_document_topics(doc, minimum_probability=0.02, minimum_phi_value=None, per_word_topics=False))   

data['Topic_ID_Prob']=topics_doc

topic_name=[]
for i in range(size(data['Topics_ID_Prob'])):
    topic_name.append(topic_label[data['Topics_ID_Prob'][i][0][0]])
data['Topic_Name']=topic_name

#Testing model on unseen document
#unseen_document = 'How a Pentagon deal became an identity crisis for Google'
#unseen_document = data_text
#bow_vector = dictionary.doc2bow(preprocess(unseen_document))

#for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
#    print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('new_file_topics.xlsx', engine='xlsxwriter')
# Convert the dataframe to an XlsxWriter Excel object.
data.to_excel(writer, sheet_name='Sheet1')
# Close the Pandas Excel writer and output the Excel file.
writer.save()
    
#Visualize Topics
data.groupby(['Countries','Topic_Name']).size().unstack().plot(kind='bar',stacked=True)
plt.show()

#Plot Sectors
data.groupby(['Sectors','Topic_Name']).size().unstack().plot(kind='bar',stacked=True)
plt.show()

#matplotlib inline
#import pyLDAvis
#import pyLDAvis.gensim
#vis = pyLDAvis.gensim.prepare(topic_model=lda_model, corpus=corpus, dictionary=dictionary_LDA)
#pyLDAvis.enable_notebook()
#pyLDAvis.display(vis)