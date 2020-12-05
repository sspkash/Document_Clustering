from gensim.corpora import Dictionary
from gensim.models import TfidfModel
import pandas as pd
import pickle

def main():
    news_df = pd.read_pickle("news_df.pkl")

    # Bag Of Words - Vocab 1
    dictionary:Dictionary = Dictionary.load('vocabulary1.gensim')
    bow_voc1_corpus = [dictionary.doc2bow(doc_tokens) for doc_tokens in news_df['DocTokens']]
    pickle.dump(bow_voc1_corpus, open('bow_voc1_corpus.pkl', 'wb'))

    # Bag Of Words - Vocab 2
    dictionary2: Dictionary = Dictionary.load('vocabulary2.gensim')
    bow_voc2_corpus = [dictionary2.doc2bow(doc_tokens) for doc_tokens in news_df['DocTokens']]
    pickle.dump(bow_voc2_corpus, open('bow_voc2_corpus.pkl', 'wb'))

    # TF-IDF - Vocab 1
    tfidf1 = TfidfModel(bow_voc1_corpus, smartirs='ntc')
    tfidf_voc1_corpus = tfidf1[bow_voc1_corpus]
    pickle.dump(tfidf_voc1_corpus, open('tfidf_voc1_corpus.pkl', 'wb'))

    # TF-IDF - Vocab 2
    tfidf2 = TfidfModel(bow_voc2_corpus, smartirs='ntc')
    tfidf_voc2_corpus = tfidf2[bow_voc2_corpus]
    pickle.dump(tfidf_voc2_corpus, open('tfidf_voc2_corpus.pkl', 'wb'))



if __name__ == '__main__':
    main()