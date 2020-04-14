# Topic Modeling/LDA
Topic modeling is a type of statistical modeling for discovering the abstract “topics” that occur in a collection of documents. Latent Dirichlet Allocation (LDA) is an example of topic model and is used to classify text in a document to a particular topic. It builds a topic per document model and words per topic model, modeled as Dirichlet distributions.
Here we are going to apply LDA to the Dataset “Sample_Data_IEGKC.xlsx” and split them into topics. 

Dataset “Sample_Data_IEGKC.xlsx”* contains a text column with paragraphs on lessons learned from IEG’s Project Performance Assessment Reports (PPARs), along with country and sector information.

## Code
The code that runs the model is in TopicModeling.py

## Before Running the Program: 

Please run "pip install -r requirements.txt" in your terminal first. This will load all the required libraries for the code including gensim and nltk libraries. 

## Methodology

Steps for Data Pre-processing: 

Tokenization: Split the text into sentences and the sentences into words. Lowercase the words and remove punctuation.
Words that have fewer than 3 characters are removed.
All stopwords are removed.
Words are lemmatized — words in third person are changed to first person and verbs in past and future tenses are changed into present.
Words are stemmed — words are reduced to their root form.

The model we are using is Bag of Words (how many times a word is used, no weighting) and TF_IDF (weights the words in the sentence based on their frequency and rank them accordingly). 

## Results

Results are printing in a #new_file_topics.xlsx, which demonstrates the topic distribution for the whole document. Each element in the list Topics column is a pair of a topic’s id, and the probability that was assigned to it. I set the minimum probability to 0.02 so that we have the more probable topics. Then it is our manual task to interpret the meaning of each topic ourself and label the topic numbers. 

After running the model we get 5 topics for the whole documents which are interpreted through 10 main words coming out of the model. These 5 Topics are the closest approximation of abstract model I could get from those words but can be improved based on more knowledge from the context. 

The Topic IDs are:
Topic 0: 'Government' 
Topic 1: 'Public Health'
Topic 2: 'Financial Planning'
Topic 3: 'Water Sector' 
Topic 4: 'Policy Implementation'

Graphs are plotted based on the Countries and Sectors. However, due to the big number of countries and sectors, the x axis is abit crowded which can be modified by clustering those or other methods of visualization.
