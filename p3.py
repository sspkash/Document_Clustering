from gensim.models.ldamodel import LdaModel as ldamodel
import pickle
import gensim
import pyLDAvis.gensim
import numpy as np
from sklearn.manifold import TSNE
from collections import OrderedDict
import matplotlib.pyplot as plt
import pandas as pd

#ToDo: Make Interactive tSNE
# # Bokeh
# from bokeh.io import output_notebook
# from bokeh.plotting import figure, show, save
# from bokeh.models import HoverTool, CustomJS, ColumnDataSource, Slider
# from bokeh.layouts import column
# from bokeh.palettes import all_palettes
# from bokeh.resources import INLINE
# output_notebook(INLINE)


no_of_topics = 20

def main():
    global no_of_topics
    # LDA for Voc1 using BOW
    vocabulary1 = gensim.corpora.Dictionary.load('vocabulary1.gensim')
    bow_voc1_corpus = pickle.load(open('bow_voc1_corpus.pkl', 'rb'))
    ldamodel1 = ldamodel(bow_voc1_corpus, num_topics=no_of_topics, id2word=vocabulary1, passes=15)
    ldamodel1.save('ldamodel1.gensim')

    # ToDO: LDA for Voc2 using BOW
    vocabulary2 = gensim.corpora.Dictionary.load('vocabulary2.gensim')
    bow_voc2_corpus = pickle.load(open('bow_voc2_corpus.pkl', 'rb'))
    ldamodel2 = ldamodel(bow_voc2_corpus, num_topics=no_of_topics, id2word=vocabulary2, passes=15)
    ldamodel2.save('ldamodel2.gensim')


    # ToDo: TSNE for LDA Model 1
    ldamodel1fortsne = ldamodel.load('ldamodel1.gensim')
    top_dist = []
    keys = []
    for d in bow_voc1_corpus:
        tmp = {i: 0 for i in range(no_of_topics)}
        tmp.update(dict(ldamodel1fortsne[d]))
        vals = list(OrderedDict(tmp).values())
        top_dist += [vals]
        keys += [np.argmax(vals)]

    tsne = TSNE(n_components=2,perplexity=50,early_exaggeration=15,learning_rate=500,n_iter=2000)
    X_tsne = tsne.fit_transform(top_dist)

    plt.figure(figsize=(10, 10))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel('tsne - 1', fontsize=20)
    plt.ylabel('tsne - 2', fontsize=20)
    plt.title("t-sne for LDA Model 1", fontsize=20)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1],c=keys, cmap=plt.cm.tab20)
    plt.show()
    plt.savefig('tsne_ldamodel1.png')



    # ToDo: TSNE for LDA Model 2
    ldamodel2fortsne = ldamodel.load('ldamodel2.gensim')
    top_dist2 = []
    keys2 = []
    for d in bow_voc2_corpus:
        tmp = {i: 0 for i in range(no_of_topics)}
        tmp.update(dict(ldamodel2fortsne[d]))
        vals = list(OrderedDict(tmp).values())
        top_dist2 += [vals]
        keys2 += [np.argmax(vals)]

    tsne2 = TSNE(n_components=2, perplexity=50, early_exaggeration=15, learning_rate=500, n_iter=2000)
    X_tsne2 = tsne2.fit_transform(top_dist2)

    plt.figure(figsize=(10, 10))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel('tsne - 1', fontsize=20)
    plt.ylabel('tsne - 2', fontsize=20)
    plt.title("t-sne for LDA Model 2", fontsize=20)
    plt.scatter(X_tsne2[:, 0], X_tsne2[:, 1], c=keys2, cmap=plt.cm.tab20)
    plt.show()
    plt.savefig('tsne_ldamodel2.png')

    # ToDo: pyLDAVis for LDA Model 1
    ldamodel1forldavis = ldamodel.load('ldamodel1.gensim')
    bow_lda1_display = pyLDAvis.gensim.prepare(ldamodel1forldavis, bow_voc1_corpus, vocabulary1, sort_topics=False)
    pyLDAvis.save_html(bow_lda1_display, 'ldavis_ldamodel1.html')
    pyLDAvis.show(bow_lda1_display)


    # # ToDo: pyLDAVis for LDA Model 2
    # ldamodel2forldavis = ldamodel.load('ldamodel2.gensim')
    # bow_lda2_display = pyLDAvis.gensim.prepare(ldamodel2forldavis, bow_voc2_corpus, vocabulary2, sort_topics=False)
    # pyLDAvis.save_html(bow_lda2_display, 'ldavis_ldamodel2.html')
    # pyLDAvis.show(bow_lda2_display)


if __name__ == '__main__':
    main()