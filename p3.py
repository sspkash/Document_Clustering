from gensim.models.ldamodel import LdaModel as ldamodel
import pickle
import gensim
import pyLDAvis.gensim
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors
from collections import OrderedDict
import pandas as pd


# Bokeh
from bokeh.io import output_notebook
from bokeh.plotting import figure, show, save
from bokeh.models import HoverTool, CustomJS, ColumnDataSource, Slider
from bokeh.layouts import column
from bokeh.palettes import all_palettes
from bokeh.resources import INLINE
output_notebook(INLINE)


no_of_topics = 20

def main():
    global no_of_topics
    # ToDo: LDA for Voc1 using BOW
    vocabulary1 = gensim.corpora.Dictionary.load('vocabulary1.gensim')
    bow_voc1_corpus = pickle.load(open('bow_voc1_corpus.pkl', 'rb'))
    # ldamodel1 = ldamodel(bow_voc1_corpus, num_topics=no_of_topics, id2word=vocabulary1, passes=15)
    # ldamodel1.save('ldamodel1.gensim')

    # ToDO: LDA for Voc2 using BOW


    # ToDo: TSNE for LDA Model 1
    ldamodel1fortsne = ldamodel.load('bowldamodel.gensim')
    top_dist = []
    keys = []
    for d in bow_voc1_corpus:
        tmp = {i: 0 for i in range(no_of_topics)}
        tmp.update(dict(ldamodel1fortsne[d]))
        vals = list(OrderedDict(tmp).values())
        top_dist += [vals]
        keys += [np.argmax(vals)]

    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(top_dist)
    # cluster_colors = {0:'#e6194b', 1:'#3cb44b', 2:'#ffe119', 3:'#4363d8', 4:'#f58231', 5:'#911eb4',
    #                   6:'#46f0f0', 7:'#f032e6', 8:'#bcf60c', 9:'#fabebe', 10:'#008080',
    #                   11:'#e6beff', 12:'#9a6324', 13:'#fffac8', 14:'#800000', 15:'#aaffc3',
    #                   16:'#808000', 17:'#ffd8b1', 18:'#000075', 19:'#808080'}
    # my_colors = pd.DataFrame(keys).apply(lambda l: cluster_colors[l])
    plot = figure(title="t-SNE Clustering of {} LDA Topics".format(no_of_topics),
                   plot_width=900, plot_height=700)
    # ToDO: color
    plot.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1])
    show(plot)
    save(plot, '{}.html'.format("tSNE"))

    # hm = np.array([[y for (x, y) in ldamodel1fortsne[bow_voc1_corpus[i]]] for i in range(len(bow_voc1_corpus))])
    # tsne = TSNE(random_state=2017, perplexity=30, early_exaggeration=120)
    # tsne_lda = tsne.fit_transform(hm)
    # output_notebook()
    # mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
    # plot = figure(title="t-SNE Clustering of {} LDA Topics".format(no_of_topics),
    #               plot_width=900, plot_height=700)
    # plot.scatter(x=tsne_lda[:, 0], y=tsne_lda[:, 1])  # color=mycolors[topic_num]
    # show(plot)

    # ToDo: TSNE for LDA Model 2

    # ToDo: pyLDAVis for LDA Model 1
    # ldamodel1forldavis = ldamodel.load('ldamodel1.gensim')
    # bow_lda_display = pyLDAvis.gensim.prepare(ldamodel1forldavis, bow_voc1_corpus, vocabulary1, sort_topics=False)
    # pyLDAvis.show(bow_lda_display)

    # ToDo: pyLDAVis for LDA Model 2

if __name__ == '__main__':
    main()