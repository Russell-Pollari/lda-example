#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 13:25:18 2020

@author: minaekramnia
"""

import pandas as pd
import gensim
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import numpy as np
import nltk
from gensim import models
import matplotlib.pyplot as plt


nltk.download('wordnet')
np.random.seed(2018)


def load_data(filename):
    '''
    Load excel file into Dataframe
    '''
    data = pd.read_excel(filename)
    return data


def lemmatize_stemming(text):
    stemmer = PorterStemmer()
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


filename = '/Users/minaekramnia/Downloads/SampleData_IEGKC_DS_STC_Test.xlsx'
data = load_data(filename)

processed_docs = data['Pragraph'].map(preprocess)

#Bag of words on the dataset
#creating a dic that convert indexes into words
dictionary = gensim.corpora.Dictionary(processed_docs)
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
#each word has an index - looping through each doc, converting into bow representation
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

#TF-IDF Model
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]

#Running LDA using TF-IDF
lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=5, id2word=dictionary, passes=2, workers=4)

#Topics generated_TF-IDF
for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))

topic_label = {0: 'Government', 1: 'Public Health', 2: 'Financial Planning', 3: 'Water Sector', 4: 'Policy Implementation'}
print(topic_label)

topics_doc=[]
for doc in bow_corpus:
    topics_doc.append(lda_model_tfidf.get_document_topics(doc, minimum_probability=0.02, minimum_phi_value=None, per_word_topics=False))

data['Topic_ID_Prob']=topics_doc

topic_name=[]
for i in range(size(data['Topic_ID_Prob'])):
    topic_name.append(topic_label[data['Topic_ID_Prob'][i][0][0]])
data['Topic_Name']=topic_name


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
