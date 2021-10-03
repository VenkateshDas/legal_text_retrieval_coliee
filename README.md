# Legal Information Retrieval Using Traditional Approaches and Binary Relevance Classification Approach on Translated Text
Information Retrieval project on COLIEE legal dataset of English translated Japanese Civil Code. 

Folder structure : 

Data : This folder contains the statute laws for training and testing. </br>
Embeddings : The law2vec and GloVe word embeddings are stored in this folder. </br>
src : </br>
    - data_analysis : The notebook for parsing the input data and Exploratory Data Analysis is present in this folder</br>
    - classical_approach : The TF-IDF and BM25 implementation notebook is present in this folder.</br>
    - modern_approach :</br>
      - DistilBERT : </br>
        - COLIEE_tranformer.ipynb : The binary relevance classification using DistilBERT notebook is present </br>
        - create_classification_Dataset.ipynb : The creation of classification dataset for DistilBERT model. </br>
        - create_downsample_classification.ipynb : Notebook for creating downsample dataset for DistilBERT model. </br>
      - WMD : </br>
        - wmd.py : The Word Movers Distance script </br>
