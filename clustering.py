from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.metrics.cluster import normalized_mutual_info_score
import pandas as pd
from gensim.models.ldamodel import LdaModel
import pickle
import numpy as np
from gensim.models.doc2vec import TaggedDocument, Doc2Vec


def main():
    news_df = pd.read_pickle("news_df.pkl")
    detokens = [TreebankWordDetokenizer().detokenize(token_list) for token_list in news_df['DocTokens']]
    bow_vectorizer = CountVectorizer()
    X_bow = bow_vectorizer.fit_transform(detokens)
    print(X_bow.shape)
    km_bow = KMeans(n_clusters=20, init='k-means++', max_iter=100, n_init=1)
    y_bow = km_bow.fit_predict(X_bow)
    print(len(y_bow))
    nmi_score = normalized_mutual_info_score(news_df['Topic'], y_bow)
    print(nmi_score)

    tfidf_vectorizer = TfidfVectorizer()
    X_tfidf = tfidf_vectorizer.fit_transform(detokens)
    print(X_tfidf.shape)
    km_tfidf = KMeans(n_clusters=20, init='k-means++', max_iter=100, n_init=1)
    y_tfdif = km_tfidf.fit_predict(X_tfidf)
    print(len(y_tfdif))
    nmi_score = normalized_mutual_info_score(news_df['Topic'], y_tfdif)
    print(nmi_score)

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

    km_topic_dist = KMeans(n_clusters=20, init='k-means++', max_iter=100, n_init=1)
    y_topic_dist = km_topic_dist.fit_predict(np.array(predict_topics).reshape(-1, 1))
    print(len(y_topic_dist))
    nmi_score = normalized_mutual_info_score(news_df['Topic'], y_topic_dist)
    print(nmi_score)

    doc2vec_model_1 = Doc2Vec.load('doc2vec_voc1.model')
    X_doc2vec = doc2vec_model_1.docvecs.vectors_docs
    km_doc2vec = KMeans(n_clusters=20, init='k-means++', max_iter=100, n_init=1)
    y_doc2vec = km_doc2vec.fit_predict(X_doc2vec)
    print(len(y_doc2vec))
    nmi_score = normalized_mutual_info_score(news_df['Topic'], y_doc2vec)
    print(nmi_score)

if __name__ == '__main__':
    main()