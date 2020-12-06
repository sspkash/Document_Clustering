from sklearn.datasets import fetch_20newsgroups
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.cluster import normalized_mutual_info_score
import pickle
import numpy as np
import matplotlib.pyplot as plt
from numpy import unique
from numpy import where
from sklearn.decomposition import TruncatedSVD

def main():
    news_corpus = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'), shuffle=False)
    # bow_voc1_corpus = pickle.load(open('bow_voc1_corpus.pkl', 'rb'))
    # dt = np.dtype('int,int')
    # array = np.array(bow_voc1_corpus,dtype=dt)
    #[[(0,1),(2,3)], []]
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(news_corpus.data)
    print(X.shape)
    # print(news_corpus.data)
    # vectorizer2 = CountVectorizer(stop_words='english')
    # X2 = vectorizer2.fit_transform(news_corpus.data)
    # print(X2.shape)

    # km = KMeans(n_clusters=20, init='k-means++', max_iter=100, n_init=1)
    # y_model = km.fit_predict(X)
    # print(y_model)
    # # nmi_score = normalized_mutual_info_score(news_corpus.target, y_model)
    # # print(nmi_score)
    #
    # # km2 = KMeans(n_clusters=20, init='k-means++', max_iter=100, n_init=1)
    # # y_model2 =  km2.fit_predict(X2)
    # # nmi_score2 = normalized_mutual_info_score(news_corpus.target, y_model2)
    # # print(nmi_score2)
    # # data = news_corpus.data
    # # labels = news_corpus.target
    pca = TruncatedSVD(n_components=2)
    principalComponents = pca.fit_transform(X)

    plt.figure(figsize=(10, 10))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel('tsne - 1', fontsize=20)
    plt.ylabel('tsne - 2', fontsize=20)
    plt.title("t-sne for LDA Model 1", fontsize=20)
    # clusters = unique(y_model)
    # # create scatter plot for samples from each cluster
    # for cluster in clusters:
    #     # get row indexes for samples with this cluster
    #     row_ix = where(y_model == cluster)
    #     # create scatter of these samples
    #     plt.scatter(news_corpus.dat[row_ix, 0], news_corpus.data[row_ix, 1])
    # # show the plot
    #
    plt.scatter(principalComponents[:, 0], principalComponents[:, 1], c=news_corpus.target[:], cmap=plt.cm.tab20)
    # #plt.scatter(X[:, 0], X[:, 1])
    # # plt.savefig('tsne_ldamodel1.png')
    plt.show()

if __name__ == '__main__':
    main()