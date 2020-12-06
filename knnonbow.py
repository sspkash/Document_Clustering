from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from gensim.models.ldamodel import LdaModel
import pickle
from collections import OrderedDict

def main():
    news_df = pd.read_pickle("news_df.pkl")
    no_of_topics = 20
    # detokens = [TreebankWordDetokenizer().detokenize(token_list) for token_list in news_df['DocTokens']]
    # bow_vectorizer = CountVectorizer()
    # X_bow = bow_vectorizer.fit_transform(detokens)
    # print(X_bow.shape)
    # X = X_bow
    # y = news_df['Topic']
    # X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size= 0.2, random_state=1)
    # X_test, X_validation, y_test, y_validation = train_test_split(X_val_test, y_val_test, test_size= 0.5, random_state=1)
    #
    # statistics = [["Training", len(set(y_train)), X_train.shape[0], X_train.shape[1]],
    #               ["Validation", len(set(y_validation)), X_validation.shape[0], X_validation.shape[1]],
    #               ["Test", len(set(y_test)), X_test.shape[0], X_test.shape[1]]]
    # print(tabulate(statistics, headers=["Dataset", "# of classes", "# of samples", "# of feature dimension"]))
    # # with open('q4_statistics.txt', 'w') as outputfile:
    # #     outputfile.write(tabulate(statistics, headers=["Dataset", "# of classes", "# of samples", "# of feature dimension"]))
    #
    # k_range = {1,3,5,8,15,20}
    # #k_range = {5,20}
    # scores_validation_list = []
    # scores_test_list = []
    #
    # i=0
    # for k in k_range:
    #     knn = KNeighborsClassifier(n_neighbors=k)
    #     knn.fit(X_train,y_train)
    #     y_validation_pred = knn.predict(X_validation)
    #     y_test_pred = knn.predict(X_test)
    #     scores_validation_list.append(metrics.accuracy_score(y_validation,y_validation_pred))
    #     scores_test_list.append(metrics.accuracy_score(y_test,y_test_pred))
    #
    # print(scores_validation_list)
    # print(scores_test_list)
    #
    # tfidf_vectorizer = TfidfVectorizer()
    # X_tfidf = tfidf_vectorizer.fit_transform(detokens)
    # X = X_tfidf
    # y = news_df['Topic']
    # X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.2, random_state=1)
    # X_test, X_validation, y_test, y_validation = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=1)
    #
    # statistics = [["Training", len(set(y_train)), X_train.shape[0], X_train.shape[1]],
    #               ["Validation", len(set(y_validation)), X_validation.shape[0], X_validation.shape[1]],
    #               ["Test", len(set(y_test)), X_test.shape[0], X_test.shape[1]]]
    # print(tabulate(statistics, headers=["Dataset", "# of classes", "# of samples", "# of feature dimension"]))
    # # with open('q4_statistics.txt', 'w') as outputfile:
    # #     outputfile.write(tabulate(statistics, headers=["Dataset", "# of classes", "# of samples", "# of feature dimension"]))
    #
    # k_range = {1, 3, 5, 8, 15, 20}
    # # k_range = {5,20}
    # scores_validation_list = []
    # scores_test_list = []
    #
    # i = 0
    # for k in k_range:
    #     knn = KNeighborsClassifier(n_neighbors=k)
    #     knn.fit(X_train, y_train)
    #     y_validation_pred = knn.predict(X_validation)
    #     y_test_pred = knn.predict(X_test)
    #     scores_validation_list.append(metrics.accuracy_score(y_validation, y_validation_pred))
    #     scores_test_list.append(metrics.accuracy_score(y_test, y_test_pred))
    #
    # print(scores_validation_list)
    # print(scores_test_list)
    #
    # doc2vec_model_1 = Doc2Vec.load('doc2vec_voc1.model')
    # X_doc2vec = doc2vec_model_1.docvecs.vectors_docs
    # X = X_doc2vec
    # y = news_df['Topic']
    # X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.2, random_state=1)
    # X_test, X_validation, y_test, y_validation = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=1)
    #
    # statistics = [["Training", len(set(y_train)), X_train.shape[0], X_train.shape[1]],
    #               ["Validation", len(set(y_validation)), X_validation.shape[0], X_validation.shape[1]],
    #               ["Test", len(set(y_test)), X_test.shape[0], X_test.shape[1]]]
    # print(tabulate(statistics, headers=["Dataset", "# of classes", "# of samples", "# of feature dimension"]))
    # # with open('q4_statistics.txt', 'w') as outputfile:
    # #     outputfile.write(tabulate(statistics, headers=["Dataset", "# of classes", "# of samples", "# of feature dimension"]))
    #
    # k_range = {1, 3, 5, 8, 15, 20}
    # # k_range = {5,20}
    # scores_validation_list = []
    # scores_test_list = []
    #
    # i = 0
    # for k in k_range:
    #     knn = KNeighborsClassifier(n_neighbors=k)
    #     knn.fit(X_train, y_train)
    #     y_validation_pred = knn.predict(X_validation)
    #     y_test_pred = knn.predict(X_test)
    #     scores_validation_list.append(metrics.accuracy_score(y_validation, y_validation_pred))
    #     scores_test_list.append(metrics.accuracy_score(y_test, y_test_pred))
    #
    # print(scores_validation_list)
    # print(scores_test_list)

    # doc2vec_model_1 = Doc2Vec.load('doc2vec_voc1.model')
    # X_doc2vec = doc2vec_model_1.docvecs.vectors_docs
    lda_model_1 = LdaModel.load('ldamodel_voc1.model')
    bow_voc1_corpus = pickle.load(open('bow_voc1_corpus.pkl', 'rb'))
    top_dist = []
    keys = []
    for d in bow_voc1_corpus:
        tmp = {i: 0 for i in range(no_of_topics)}
        tmp.update(dict(lda_model_1[d]))
        vals = list(OrderedDict(tmp).values())
        top_dist += [vals]
        keys += [np.argmax(vals)]
    X = top_dist
    y = news_df['Topic']
    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_test, X_validation, y_test, y_validation = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=1)

    # statistics = [["Training", len(set(y_train)), X_train.shape[0], X_train.shape[1]],
    #               ["Validation", len(set(y_validation)), X_validation.shape[0], X_validation.shape[1]],
    #               ["Test", len(set(y_test)), X_test.shape[0], X_test.shape[1]]]
    # print(tabulate(statistics, headers=["Dataset", "# of classes", "# of samples", "# of feature dimension"]))
    # # with open('q4_statistics.txt', 'w') as outputfile:
    # #     outputfile.write(tabulate(statistics, headers=["Dataset", "# of classes", "# of samples", "# of feature dimension"]))

    k_range = {1, 3, 5, 8, 15, 20}
    # k_range = {5,20}
    scores_validation_list = []
    scores_test_list = []

    i = 0
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_validation_pred = knn.predict(X_validation)
        y_test_pred = knn.predict(X_test)
        scores_validation_list.append(metrics.accuracy_score(y_validation, y_validation_pred))
        scores_test_list.append(metrics.accuracy_score(y_test, y_test_pred))

    print(scores_validation_list)
    print(scores_test_list)

if __name__ == '__main__':
    main()