import gensim
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
from gensim.models.ldamodel import LdaModel as ldamodel
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tabulate import tabulate
import docstatistics
import numpy as np
import pandas as pd
import pickle
from bioinfokit.visuz import cluster
import pyLDAvis.gensim

# Bokeh
from bokeh.io import output_notebook
from bokeh.plotting import figure, show
from bokeh.models import HoverTool, CustomJS, ColumnDataSource, Slider
from bokeh.layouts import column
from bokeh.palettes import all_palettes


tokens = []
tokenCount = {}

wordsCount = []
dictionary = gensim.corpora.Dictionary()
BoW_corpus = []
tfidf_corpus = []

def visualize_stats():
    global tokens
    global tokenCount
    for word in tokens:
        tokenCount[word] = tokenCount.get(word, 0) + 1

    plt.title('New Corpus Data Statistics')
    plt.xlabel('Word')
    plt.xticks(rotation=70)
    plt.ylabel('Count')
    plt.plot(*zip(*sorted(tokenCount.items(),key=lambda x: x[1], reverse=True)[:100]))
    plt.savefig('C:\\Users\\neeth\\Desktop\\ML\\Project\\statistics.png',dpi=300,bbox_inches='tight')
    # plt.show()


def preprocess(news_corpus, stoplist):
    global wordsCount
    global tokens
    global dictionary
    global BoW_corpus
    no_of_docs = 0
    dictionary_tokens = []
    for doc in news_corpus.data:
        # print(doc)
        no_of_docs = no_of_docs+1
        words = gensim.utils.simple_preprocess(doc, True, 3)
        wordsCount.append(len(words))
        words_list_per_doc = []
        for word in words:
            if word not in stoplist:
                tokens.append(word)
                words_list_per_doc.append(word)
        # print(words_list_per_doc)
        dictionary_tokens.append(words_list_per_doc)
        BoW_corpus.append(dictionary.doc2bow(words_list_per_doc, allow_update=True))
    dictionary = gensim.corpora.Dictionary(dictionary_tokens)
    dictionary.save('dictionary.gensim')

    stats = [[no_of_docs, len(tokens), len(set(tokens)),
              min(wordsCount), max(wordsCount), docstatistics.mean(wordsCount), docstatistics.stdev(wordsCount)]]
    print(tabulate(stats, headers=["# of docs",
                                        "# of words", "# of unique words",
                                        "min words", "max words", "mean words", "std of words"]))
    with open('pre_process_stats.txt', 'w') as outputfile:
        outputfile.write(
            tabulate(stats, headers=["# of docs",
                                     "# of words", "# of unique words",
                                     "min words", "max words", "mean words", "std of words"]))


def build_vocabulary():
    global BoW_corpus
    global dictionary
    global tfidf_corpus
    # Build vocabulary
    # 1. Bag of words
    # print(BoW_corpus)
    # bow_vocabulary = [[(dictionary[id], count) for id, count in doc] for doc in BoW_corpus]
    # print(bow_vocabulary)
    pickle.dump(BoW_corpus, open('bow_corpus.pkl', 'wb'))


    # 2. tf-idf
    tfidf = gensim.models.TfidfModel(BoW_corpus, smartirs='ntc')
    tfidf_corpus = tfidf[BoW_corpus]
    # for doc in tfidf_corpus:
    #     print(doc)
    # tfidf_vocabulary =[[(dictionary[id], np.around(freq,decimals=2)) for id, freq in doc]for doc in tfidf_corpus]
    # print (tfidf_vocabulary)
    pickle.dump(tfidf_corpus, open('tfidf_corpus.pkl', 'wb'))


def train_lda():
    global BoW_corpus
    global tfidf_corpus
    no_of_topics = 20
    bowldamodel = ldamodel(BoW_corpus,num_topics=no_of_topics, id2word=dictionary,passes=15)
    bowldamodel.save('bowldamodel.gensim')
    # for i in range(bowldamodel.num_topics):
    #     print (bowldamodel.print_topic(i))

    tfidfldamodel = ldamodel(tfidf_corpus, num_topics=no_of_topics, id2word=dictionary, passes=15)
    tfidfldamodel.save('tfidfldamodel.gensim')
    # for i in range(tfidfldamodel.num_topics):
    #     print(tfidfldamodel.print_topic(i))

    hm = np.array([[y for (x, y) in bowldamodel[BoW_corpus[i]]] for i in range(len(BoW_corpus))])
    tsne = TSNE(random_state=2017, perplexity=30, early_exaggeration=120)
    tsne_lda = tsne.fit_transform(hm)
    # #embedding = pd.DataFrame(embedding, columns=['x', 'y'])
    # # embedding['hue'] = hm.argmax(axis=1)
    # cluster.tsneplot(score=embedding)

    output_notebook()
    mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
    plot = figure(title="t-SNE Clustering of {} LDA Topics".format(no_of_topics),
                  plot_width=900, plot_height=700)
    plot.scatter(x=tsne_lda[:, 0], y=tsne_lda[:, 1]) #color=mycolors[topic_num]
    show(plot)




def main():
    news_corpus = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'), shuffle=False)
    nltk.download('stopwords')
    nltk.download('punkt')
    stoplist = stopwords.words('english')

    preprocess(news_corpus, stoplist)
    visualize_stats()
    build_vocabulary()
    train_lda()
    # visualize_topics()


if __name__ == '__main__':
    main()