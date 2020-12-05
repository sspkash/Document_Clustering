import pandas as pd
from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.ldamodel import LdaModel
import pickle
import pyLDAvis.gensim
import numpy as np
from sklearn.manifold import TSNE
from collections import OrderedDict
import matplotlib.pyplot as plt
import pandas as pd


def explore_topic(lda_model, topic_number, topn, output=True):
    """
    accept a ldamodel, atopic number and topn vocabs of interest
    prints a formatted list of the topn terms
    """
    terms = []
    for term, frequency in lda_model.show_topic(topic_number, topn=topn):
        terms += [term]
        if output:
            print(u'{:20} {:.3f}'.format(term, round(frequency, 3)))

    return terms


def main():
    no_of_topics = 20
    news_df = pd.read_pickle("news_df.pkl")
    # LDA Model - Vocab 1
    dictionary1:Dictionary = Dictionary.load('vocabulary1.gensim')
    bow_voc1_corpus = pickle.load(open('bow_voc1_corpus.pkl', 'rb'))
    # lda_model_1 = LdaMulticore(corpus=bow_voc1_corpus,
    #                          id2word=dictionary1,
    #                          random_state=100,
    #                          num_topics=no_of_topics,
    #                          passes=10,
    #                          chunksize=1000,
    #                          batch=False,
    #                          alpha='asymmetric',
    #                          decay=0.5,
    #                          offset=64,
    #                          eta=None,
    #                          eval_every=0,
    #                          iterations=100,
    #                          gamma_threshold=0.001)
    # lda_model_1.save('ldamodel_voc1.model')

    # LDA Model 1 - LdaVis
    lda_model_1 = LdaModel.load('ldamodel_voc1.model')
    lda_model_1_display = pyLDAvis.gensim.prepare(lda_model_1, bow_voc1_corpus, dictionary1, sort_topics=False)
    pyLDAvis.save_html(lda_model_1_display, 'lda_model_1_display.html')
    pyLDAvis.show(lda_model_1_display)

    # LDA Model 2 - Vocab 2
    # dictionary2: Dictionary = Dictionary.load('vocabulary2.gensim')
    # bow_voc2_corpus = pickle.load(open('bow_voc2_corpus.pkl', 'rb'))
    # lda_model_2 = LdaMulticore(corpus=bow_voc2_corpus,
    #                          id2word=dictionary2,
    #                          random_state=100,
    #                          num_topics=no_of_topics,
    #                          passes=10,
    #                          chunksize=1000,
    #                          batch=False,
    #                          alpha='asymmetric',
    #                          decay=0.5,
    #                          offset=64,
    #                          eta=None,
    #                          eval_every=0,
    #                          iterations=100,
    #                          gamma_threshold=0.001)
    # lda_model_2.save('ldamodel_voc2.model')


    # LDA Model 2 - LdaVis
    # lda_model_2 = LdaModel.load('ldamodel_voc2.model')
    # for c in lda_model_2[bow_voc2_corpus[1:10]]:
    #     print("Document Topics      : ", c[0])  # [(Topics, Perc Contrib)]
    #     print("------------------------------------------------------\n")
    #
    # lda_model_2 = LdaModel.load('ldamodel_voc2.model')
    # lda_model_2_display = pyLDAvis.gensim.prepare(lda_model_2, bow_voc2_corpus, dictionary2, sort_topics=False)
    # pyLDAvis.save_html(lda_model_2_display, 'lda_model_1_display.html')
    # pyLDAvis.show(lda_model_2_display)

    # LDA Model 1 - TSNE
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

    tsne = TSNE(n_components=2, perplexity=50, early_exaggeration=15, learning_rate=500, n_iter=2000)
    X_tsne = tsne.fit_transform(top_dist)
    plt.figure(figsize=(10, 10))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel('tsne - 1', fontsize=20)
    plt.ylabel('tsne - 2', fontsize=20)
    plt.title("t-sne for LDA Model 1", fontsize=20)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=keys, cmap=plt.cm.tab20)
    plt.legend(loc='best')
    plt.savefig('tsne_ldamodel1.png')
    plt.show()

    # LDA Model 2 - TSNE
    # lda_model_2 = LdaModel.load('ldamodel_voc2.model')
    # bow_voc2_corpus = pickle.load(open('bow_voc2_corpus.pkl', 'rb'))
    # top_dist2 = []
    # keys2 = []
    # for d in bow_voc2_corpus:
    #     tmp = {i: 0 for i in range(no_of_topics)}
    #     tmp.update(dict(lda_model_2[d]))
    #     vals = list(OrderedDict(tmp).values())
    #     top_dist2 += [vals]
    #     keys2 += [np.argmax(vals)]
    #
    # tsne2 = TSNE(n_components=2, perplexity=50, early_exaggeration=15, learning_rate=500, n_iter=2000)
    # X_tsne2 = tsne2.fit_transform(top_dist2)
    # plt.figure(figsize=(10, 10))
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=14)
    # plt.xlabel('tsne - 1', fontsize=20)
    # plt.ylabel('tsne - 2', fontsize=20)
    # plt.title("t-sne for LDA Model 2", fontsize=20)
    # plt.scatter(X_tsne2[:, 0], X_tsne2[:, 1], c=keys2, cmap=plt.cm.tab20)
    # # plt.legend(loc='best')
    # plt.savefig('tsne_ldamodel2.png')
    # plt.show()

if __name__ == '__main__':
    main()