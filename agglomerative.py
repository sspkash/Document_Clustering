from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans, SpectralClustering, DBSCAN, AffinityPropagation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.metrics.cluster import normalized_mutual_info_score
import pandas as pd
from gensim.models.ldamodel import LdaModel
import pickle
import numpy as np
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from collections import OrderedDict
from scipy.sparse import csr_matrix

def main():
    news_df = pd.read_pickle("news_df.pkl")
    no_of_topics = 20
    detokens = [TreebankWordDetokenizer().detokenize(token_list) for token_list in news_df['DocTokens']]
    bow_vectorizer = CountVectorizer()
    X_bow = bow_vectorizer.fit_transform(detokens)
    print(X_bow.shape)
    km_bow = AgglomerativeClustering(n_clusters=no_of_topics)
    y_bow = km_bow.fit_predict(X_bow.todense())
    print(len(y_bow))
    nmi_score = normalized_mutual_info_score(news_df['Topic'], y_bow)
    print(nmi_score)

    # ToDO t-SNE


    tfidf_vectorizer = TfidfVectorizer()
    X_tfidf = tfidf_vectorizer.fit_transform(detokens)
    print(X_tfidf.shape)
    km_tfidf = MiniBatchKMeans(n_clusters=no_of_topics)
    y_tfdif = km_tfidf.fit_predict(X_tfidf)
    print(len(y_tfdif))
    nmi_score = normalized_mutual_info_score(news_df['Topic'], y_tfdif)
    print(nmi_score)

    # tsne = TSNE(n_components=2)
    # #tsne = TruncatedSVD(n_components=2)
    # # tsne to our document vectors
    # tsne_tfidf = tsne.fit_transform(X_tfidf)
    # plt.figure(figsize=(10, 10))
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=14)
    # plt.xlabel('tsne - 1', fontsize=20)
    # plt.ylabel('tsne - 2', fontsize=20)
    # plt.title("t-sne for Clustering TF IDF- 1", fontsize=20)
    # plt.scatter(tsne_tfidf[:, 0], tsne_tfidf[:, 1], c=y_tfdif, cmap=plt.cm.tab20)
    # plt.savefig('clustering_tfidf.png')
    # plt.show()

    lda_model_1 = LdaModel.load('ldamodel_voc1.model')
    bow_voc1_corpus = pickle.load(open('bow_voc1_corpus.pkl', 'rb'))
    predict_topics = []
    for i in range(len(bow_voc1_corpus)):
        topic = 0
        temp_pred = 0
        for (x, y) in lda_model_1[bow_voc1_corpus[i]]:
            if y > temp_pred:
                temp_pred = y
                topic = x
        predict_topics.append(topic)

    km_topic_dist = AgglomerativeClustering(n_clusters=no_of_topics)
    y_topic_dist = km_topic_dist.fit_predict(np.array(predict_topics).reshape(-1, 1))
    print(len(y_topic_dist))
    nmi_score = normalized_mutual_info_score(news_df['Topic'], y_topic_dist)
    print(nmi_score)

    top_dist = []
    keys = []
    for d in bow_voc1_corpus:
        tmp = {i: 0 for i in range(no_of_topics)}
        tmp.update(dict(lda_model_1[d]))
        vals = list(OrderedDict(tmp).values())
        top_dist += [vals]
        keys += [np.argmax(vals)]


    doc2vec_model_1 = Doc2Vec.load('doc2vec_voc1.model')
    X_doc2vec = doc2vec_model_1.docvecs.vectors_docs
    km_doc2vec = AgglomerativeClustering(n_clusters=no_of_topics)
    y_doc2vec = km_doc2vec.fit_predict(X_doc2vec)
    print(len(y_doc2vec))
    nmi_score = normalized_mutual_info_score(news_df['Topic'], y_doc2vec)
    print(nmi_score)

    # tsne = TSNE(n_components=2, perplexity=50, early_exaggeration=15, learning_rate=500, n_iter=2000)
    # X_tsne = tsne.fit_transform(X_doc2vec)
    # plt.figure(figsize=(10, 10))
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=14)
    # plt.xlabel('tsne - 1', fontsize=20)
    # plt.ylabel('tsne - 2', fontsize=20)
    # plt.title("t-sne for LDA Model 1", fontsize=20)
    # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_doc2vec, cmap=plt.cm.tab20)
    # plt.legend(loc='best')
    # plt.savefig('clustering_ldamodel1.png')
    # plt.show()


if __name__ == '__main__':
    main()