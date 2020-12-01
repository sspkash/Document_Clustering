from sklearn.datasets import fetch_20newsgroups
import nltk
from nltk.corpus import stopwords
import gensim
import pickle


def main():
    news_corpus = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'), shuffle=False)
    nltk.download('stopwords')
    nltk.download('punkt')
    stoplist = stopwords.words('english')

    # Vocabulary 1
    tokens = []
    dictionary_tokens = []
    for doc in news_corpus.data:
        # print(doc)
        words = gensim.utils.simple_preprocess(doc, True, 3)
        words_list_per_doc = []
        for word in words:
            if word not in stoplist:
                tokens.append(word)
                words_list_per_doc.append(word)
        # print(words_list_per_doc)
        dictionary_tokens.append(words_list_per_doc)

    dictionary = gensim.corpora.Dictionary(dictionary_tokens)
    print(len(dictionary))
    dictionary.save('vocabulary1.gensim')


    #Vocabulary 2

    dicty = dictionary.cfs
    gensim.utils.trim_vocab_by_freq(dicty, 2000)
    dictionary.filter_tokens(good_ids=dicty.keys())
    print(len(dictionary))
    dictionary.save('vocabulary2.gensim')

    #Apply BOW on Voc 1
    vocabulary1 = gensim.corpora.Dictionary.load('vocabulary1.gensim')
    bow_voc1_corpus = [vocabulary1.doc2bow(doc_tokens) for doc_tokens in dictionary_tokens]
    pickle.dump(bow_voc1_corpus, open('bow_voc1_corpus.pkl', 'wb'))

    #  ToDo Apply BOW on Voc 2
    vocabulary2 = gensim.corpora.Dictionary.load('vocabulary2.gensim')
    bow_voc2_corpus = [vocabulary2.doc2bow(doc_tokens) for doc_tokens in dictionary_tokens]
    pickle.dump(bow_voc2_corpus, open('bow_voc2_corpus.pkl', 'wb'))

    #  Apply TFIDF on Voc 1
    tfidf1 = gensim.models.TfidfModel(bow_voc1_corpus, smartirs='ntc')
    tfidf_voc1_corpus = tfidf1[bow_voc1_corpus]
    pickle.dump(tfidf_voc1_corpus, open('tfidf_voc1_corpus.pkl', 'wb'))

    # ToDo Apply TFIDF on Voc 2
    tfidf2 = gensim.models.TfidfModel(bow_voc2_corpus, smartirs='ntc')
    tfidf_voc2_corpus = tfidf2[bow_voc2_corpus]
    pickle.dump(tfidf_voc2_corpus, open('tfidf_voc2_corpus.pkl', 'wb'))


if __name__ == '__main__':
    main()