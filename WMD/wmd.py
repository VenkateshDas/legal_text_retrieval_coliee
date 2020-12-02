import os
import re
import ast
import pandas as pd
import word_mover_distance.model as model
import numpy as np
from numpy import zeros, dtype, float32 as REAL, ascontiguousarray, fromstring
import gensim.models
from gensim import utils
from gensim.models import Word2Vec


def change_hyphen(article_number):
    return re.sub("_", "-", article_number)


def remmove_space(article_number):
    return re.sub(" ", "", article_number)


# conversion of Law2Vec text to Word2vec object

# https://stackoverflow.com/questions/45981305/convert-python-dictionary-to-word2vec-object
def law2vec_to_word2vec_format(fname, vocab, vectors, binary=True, total_vec=2):
    """Store the input-hidden weight matrix in the same format used by the original
    C word2vec-tool, for compatibility.

    Parameters
    ----------
    fname : str
        The file path used to save the vectors in.
    vocab : dict
        The vocabulary of words.
    vectors : numpy.array
        The vectors to be stored.
    binary : bool, optional
        If True, the data wil be saved in binary word2vec format, else it will be saved in plain text.
    total_vec : int, optional
        Explicitly specify total number of vectors
        (in case word vectors are appended with document vectors afterwards).

    """
    if not (vocab or vectors):
        raise RuntimeError("no input")
    if total_vec is None:
        total_vec = len(vocab)
    vector_size = vectors.shape[1]
    assert (len(vocab), vector_size) == vectors.shape
    with utils.smart_open(fname, "wb") as fout:
        print(total_vec, vector_size)
        fout.write(utils.to_utf8("%s %s\n" % (total_vec, vector_size)))
        # store in sorted order: most frequent words at the top
        for word, row in vocab.items():
            if binary:
                row = row.astype(REAL)
                fout.write(utils.to_utf8(word) + b" " + row.tostring())
            else:
                fout.write(
                    utils.to_utf8(
                        "%s %s\n" % (word, " ".join(repr(val) for val in row))
                    )
                )


def wmd_retrieval(
    query, articles, article_number, n, threshold, ground_truth, query_id
):
    total_retrieved = 0
    true_positive = 0
    total_relevent = 0
    for i in range(len(ground_truth)):
        total_relevent += len(ground_truth[i])

    query_article_sim_dict = {}
    top_n = n
    result = []

    for queries in range(len(query)):

        similarity_list = []
        print(queries)

        # if type(query) == list:
        q = query[queries]
        # else:
        #     query = query.iloc[queries]
        for i in range(len(articles)):

            article = articles.iloc[i]
            distance = word_vectors.wmdistance(q, article)
            similarity = 1 / (1 + distance)

            similarity_list.append((article_number[i], similarity))  # change

        similarity_list.sort(key=lambda elem: -elem[1])

        query_article_sim_dict.update({query_id.iloc[queries]: similarity_list})
    i = 0
    for key, values in query_article_sim_dict.items():
        rank = 1
        value = values[:top_n]
        for doc_number, score in value:

            retrieved_article_number = article_number[doc_number]

            if rank == 1:
                print(
                    str(key)
                    + " "
                    + "Q0"
                    + " "
                    + str(retrieved_article_number)
                    + " "
                    + str(rank)
                    + " "
                    + str(score)
                    + " "
                    + "OVGU"
                    + " "
                    + str(ground_truth[i])
                )
                rank += 1
                total_retrieved += 1
                if str(retrieved_article_number) in ground_truth[i]:
                    true_positive += 1
            elif score > threshold:  # change
                print(
                    str(key)
                    + " "
                    + "Q0"
                    + " "
                    + str(retrieved_article_number)
                    + " "
                    + str(rank)
                    + " "
                    + str(score)
                    + " "
                    + "OVGU"
                    + " "
                    + str(ground_truth[i])
                )
                rank += 1
                total_retrieved += 1
                if str(retrieved_article_number) in ground_truth[i]:
                    true_positive += 1
        i += 1
    return total_retrieved, true_positive, total_relevent, result


def evaluation(true_positive, total_relevent, total_retrieved):
    print(
        "total retrieved = "
        + str(total_retrieved)
        + "\ntp= "
        + str(true_positive)
        + "\ntotal relevent="
        + str(total_relevent)
    )
    false_positive = total_retrieved - true_positive
    false_negative = total_relevent - true_positive
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f_score = (5 * precision * recall) / ((4 * precision) + recall)
    print(
        "Precion = " + str(precision),
        "Recall=" + str(recall),
        "F2score = " + str(f_score),
        sep="\n",
    )


df_query_list = pd.read_pickle("cleaned_ground_truth.pkl")
df_expanded_query_list = pd.read_pickle("cleaned_extended_ground_truth.pkl")
df_civil_code_list = pd.read_pickle("cleaned_civil_code.pkl")

query_id = df_query_list["ID"]
dataset_query = df_query_list["Query"]
dataset_query_tokens = df_query_list["Query_tokens"]
dataset_query_lemmas = df_query_list["Query_lemma"]

dataset_expanded_query_tokens = df_expanded_query_list["Expanded_query_tokens"]
dataset_expanded_query_lemmas = df_expanded_query_list["Expanded_query_lemma"]

dataset_articles = df_civil_code_list["Article_description"]
dataset_article_tokens = df_civil_code_list["Article_description_tokens"]
dataset_article_lemmas = df_civil_code_list["Article_description_lemmas"]

df_civil_code_list["Article_number"] = df_civil_code_list["Article_number"].apply(
    change_hyphen
)
df_civil_code_list["Article_number"] = df_civil_code_list["Article_number"].apply(
    remmove_space
)
dataset_article_number = df_civil_code_list["Article_number"]


ground_truth = []
for line in range(len(df_query_list)):
    l = []
    l = ast.literal_eval(df_query_list["Article_numbers"].iloc[line])
    ground_truth.append(l)


model_name = input("Enter the name of the Embedding --> 'law2vec' or 'glove' : ")
# Creation of Law2Vec object that can be used by the Gensim library.
if model_name == "law2vec":
    model = model.WordEmbedding(
        model_fn="/Users/venkateshmurugadas/Documents/coliee_retrieval/active_data/COLIEE Dry Run Data /embedding/Law2Vec.200d.txt"
    )
    m = gensim.models.keyedvectors.Word2VecKeyedVectors(vector_size=200)
    m.vocab = model.model
    m.vectors = np.array(list(model.model.values()))
    law2vec_to_word2vec_format(
        binary=True,
        fname="law2vec.bin",
        total_vec=len(model.model),
        vocab=m.vocab,
        vectors=m.vectors,
    )
    word_vectors = gensim.models.keyedvectors.Word2VecKeyedVectors.load_word2vec_format(
        "law2vec.bin", binary=True
    )
    word_vectors.init_sims(replace=True)  # change
    print("law2vec loaded")
elif model_name == "glove":
    word_vectors = gensim.models.KeyedVectors.load_word2vec_format(
        "/Users/venkateshmurugadas/Documents/coliee_retrieval/active_data/COLIEE Dry Run Data /embedding/GoogleNews-vectors-negative300.bin.gz",
        binary=True,
    )
    word_vectors.init_sims(replace=True)  # change
    print("glove loaded")


query_list = []
for line in range(len(dataset_query_lemmas)):
    q = ast.literal_eval(dataset_query_lemmas.iloc[line])
    query_list.append(q)


expanded_query_list = dataset_expanded_query_lemmas.tolist()

article_number = []
for line in range(len(dataset_article_number)):
    an = ast.literal_eval(dataset_article_number.iloc[line])
    article_number.append(an)

number_of_articles = int(input("Enter the number of articles to be retrieved : "))
threshold = float(input("Enter the threshold for the similarity score : "))
feature = int(input(" Enter 1 for query lemmas or 2 for query expansion : "))

if feature == 1:
    total_retrieved, true_positive, total_relevent, result = wmd_retrieval(
        query_list,
        dataset_article_lemmas,
        article_number,
        number_of_articles,
        threshold,
        ground_truth,
        query_id,
    )

elif feature == 2:
    total_retrieved, true_positive, total_relevent, result = wmd_retrieval(
        expanded_query_list,
        dataset_article_lemmas,
        article_number,
        number_of_articles,
        threshold,
        ground_truth,
        query_id,
    )

evaluation(true_positive, total_relevent, total_retrieved)

