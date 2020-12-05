from sklearn.datasets import fetch_20newsgroups
import nltk
from nltk.corpus import stopwords
import gensim
import pandas as pd
from nltk.tokenize.treebank import TreebankWordDetokenizer

stop_word_list = []
def tokenize(doc):
    global stop_word_list
    return [token for token in gensim.utils.simple_preprocess(doc, True, 3) if token not in stop_word_list]

def main():
    global stop_word_list
    nltk.download('stopwords')
    nltk.download('punkt')
    stop_word_list = stopwords.words('english')
    stop_word_list.extend(['etc', 'faq', 'bhj', 'com', 'meg','non','qax','tct',
                           'wwiz','nuy','giz','bxn','nrhj','bxn','ghj','fpl','pmf',
                           'nei'])
    #Fetch
    dataset = fetch_20newsgroups(subset='all',shuffle=False,
                            remove=('headers', 'footers', 'quotes'))

    #Tokenize, Lower, Remove
    tokens = [tokenize(doc) for doc in dataset.data]

    # detokens = [TreebankWordDetokenizer().detokenize(token_list) for token_list in tokens]
    #Build Vocabulary 1
    dictionary = gensim.corpora.Dictionary(tokens)
    vocab = dictionary.cfs
    gensim.utils.prune_vocab(vocab, min_reduce=3, trim_rule=None)
    dictionary.filter_tokens(good_ids=vocab.keys())
    dictionary.save('vocabulary1.gensim')
    # Filter out words that occur less than 10 documents, or more than 20% of the documents.
    #dictionary.filter_extremes(no_below=10, no_above=0.5)

    doc_tokens = []
    for token_list in tokens:
        temp_list = []
        for word in token_list:
            if word in dictionary.token2id:
                temp_list.append(word)
        doc_tokens.append(temp_list)

    news_df = pd.DataFrame()
    news_df['DocTokens'] = doc_tokens
    news_df['Topic'] = dataset.target

    print(news_df.shape)
    print(news_df.head())

    news_df.to_pickle("news_df.pkl")

    # Build Vocabulary 2
    gensim.utils.trim_vocab_by_freq(vocab, 2000)
    dictionary.filter_tokens(good_ids=vocab.keys())
    dictionary.save('vocabulary2.gensim')

    doc_tokens2 = []
    for token_list in tokens:
        temp_list = []
        for word in token_list:
            if word in dictionary.token2id:
                temp_list.append(word)
        doc_tokens2.append(temp_list)

    news_df2 = pd.DataFrame()
    news_df2['DocTokens'] = doc_tokens2
    news_df2['Topic'] = dataset.target

    print(news_df2.shape)
    print(news_df2.head())

    news_df2.to_pickle("news_df2.pkl")


if __name__ == '__main__':
    main()
