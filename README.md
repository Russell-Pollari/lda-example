# Topic Modeling/LDA
Topic modeling is a type of statistical modeling for discovering the abstract “topics” that occur in a collection of documents. Latent Dirichlet Allocation (LDA) is an example of topic model and is used to classify text in a document to a particular topic. It builds a topic per document model and words per topic model, modeled as Dirichlet distributions.
Here we are going to apply LDA to the Dataset “Sample_Data_IEGKC.xlsx” and split them into topics. 

Dataset “Sample_Data_IEGKC.xlsx”* contains a text column with paragraphs on lessons learned from IEG’s Project Performance Assessment Reports (PPARs), along with country and sector information.

Loading gensim and nltk libraries

Steps for Data Pre-processing

Tokenization: Split the text into sentences and the sentences into words. Lowercase the words and remove punctuation.
Words that have fewer than 3 characters are removed.
All stopwords are removed.
Words are lemmatized — words in third person are changed to first person and verbs in past and future tenses are changed into present.
Words are stemmed — words are reduced to their root form.

The model we are using is TF_IDF that weights the words in the sentence based on their frequency and rank them accordingly. 

Results is the topic distribution for the whole document. Each element in the list is a pair of a topic’s id, and the probability that was assigned to it. I set the minimum probability to 0.02 so that we have the more probable topics. 

Then it is our manual task to interpret the meaning of each topic ourself and label the topic numbers. 


## Results
After running the model we get 5 topics for the whole documents which are interpreted through 10 main words coming out of the model. These 5 Topics are the closes approximation of abstract model I could get from those words but can be improved based on more knowledge from the context. 
Topic 0: 'Government' 
Topic 1: 'Public Health'
Topic 2: 'Financial Planning'
Topic 3: 'Water Sector' 
Topic 4: 'Policy Implementation'

## Before Running the Program: 

Please run "pip install -r requirements.txt" in your terminal first. This will run all the required libraries for the code. 

