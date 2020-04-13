# Topic Modeling/LDA
Topic modeling is a type of statistical modeling for discovering the abstract “topics” that occur in a collection of documents. Latent Dirichlet Allocation (LDA) is an example of topic model and is used to classify text in a document to a particular topic. It builds a topic per document model and words per topic model, modeled as Dirichlet distributions.
Here we are going to apply LDA to the Dataset “Sample_Data_IEGKC.xlsx” and split them into topics. 

Dataset “Sample_Data_IEGKC.xlsx”* contains a text column with paragraphs on lessons learned from IEG’s Project Performance Assessment Reports (PPARs), along with country and sector information.

The model we are using is TF_IDF that weights the words in the sentence based on their frequency and rank them accordingly. 

Results are a list of topics for each paragraph with their corresponding probability. 
