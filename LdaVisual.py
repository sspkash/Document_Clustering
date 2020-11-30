from gensim.models.ldamodel import LdaModel as ldamodel
import pyLDAvis.gensim
import pickle
import gensim


if __name__ == '__main__':

    dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')

    # #BOW LDA LDAVis
    # BoW_corpus = pickle.load(open('bow_corpus.pkl', 'rb'))
    # bowldamodel = ldamodel.load('bowldamodel.gensim')
    # bow_lda_display = pyLDAvis.gensim.prepare(bowldamodel, BoW_corpus, dictionary, sort_topics=False)
    # pyLDAvis.show(bow_lda_display)

    # TFIDF LDA LDAVis
    tfidf_corpus = pickle.load(open('tfidf_corpus.pkl', 'rb'))
    tfidfldamodel = ldamodel.load('tfidfldamodel.gensim')
    tfidf_lda_display = pyLDAvis.gensim.prepare(tfidfldamodel, tfidf_corpus, dictionary, sort_topics=False)
    pyLDAvis.show(tfidf_lda_display)