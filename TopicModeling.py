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
from gensim import models
import matplotlib.pyplot as plt

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
    stopwords = gensim.parsing.preprocessing.STOPWORDS
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in stopwords and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


def print_topics(lda_model):
    '''
    Print the words in each topic created with lda_model
    '''
    for idx, topic in lda_model.print_topics(-1):
        print('Topic: {} Word: {}'.format(idx, topic))


def save_data(dataframe, save_as='new_file_topics.xlxs'):
    '''
    Save dataframe as save_as
    '''
    with pd.ExcelWriter(save_as, engine='xlsxwriter') as writer:
        dataframe.to_excel(writer, sheet_name='Sheet1')


def main(filename):
    data = load_data(filename)
    processed_docs = data['Pragraph'].map(preprocess)

    dictionary = gensim.corpora.Dictionary(processed_docs)
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]

    lda_model_tfidf = gensim.models.LdaMulticore(
        corpus_tfidf,
        num_topics=5,
        id2word=dictionary,
        passes=2,
        workers=4
    )

    print_topics(lda_model_tfidf)

    topic_label_map = {
        0: 'Government',
        1: 'Public Health',
        2: 'Financial Planning',
        3: 'Water Sector',
        4: 'Policy Implementation'
    }

    topics_doc = []
    topic_names = []
    for doc in bow_corpus:
        topics = lda_model_tfidf.get_document_topics(
            doc,
            minimum_probability=0.02,
            minimum_phi_value=None,
            per_word_topics=False
        )
        topic_label = topic_label_map[topics[0][0]]
        topics_doc.append(topics)
        topic_names.append(topic_label)

    data['Topic_ID_Prob'] = topics_doc
    data['Topic_Name'] = topic_names

    save_data(data, save_as='new_file_topics.xlsx')

    # Visualize Topics
    data.groupby(['Countries', 'Topic_Name']).size().unstack().plot(kind='bar', stacked=True)
    plt.savefig('Countries.png')

    # Plot Sectors
    data.groupby(['Sectors', 'Topic_Name']).size().unstack().plot(kind='bar', stacked=True)
    plt.savefig('Sectors.png')


if __name__ == '__main__':
    filename = '/Users/minaekramnia/Downloads/SampleData_IEGKC_DS_STC_Test.xlsx'
    main(filename)
