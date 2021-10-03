# legal_text_retrieval_coliee
Information Retrieval project on COLIEE legal dataset of English translated Japanese Civil Code. 

Folder structure : 

Data : This folder contains the statute laws for training and testing. 
Embeddings : The law2vec and GloVe word embeddings are stored in this folder. 
src : 
    - data_analysis : The notebook for parsing the input data and Exploratory Data Analysis is present in this folder
    - classical_approach : The TF-IDF and BM25 implementation notebook is present in this folder.
    - modern_approach :
      - DistilBERT : 
        - COLIEE_tranformer.ipynb : The binary relevance classification using DistilBERT notebook is present
        - create_classification_Dataset.ipynb : The creation of classification dataset for DistilBERT model.
        - create_downsample_classification.ipynb : Notebook for creating downsample dataset for DistilBERT model.
      - WMD : 
        - wmd.py : The Word Movers Distance script
