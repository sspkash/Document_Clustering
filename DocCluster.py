import gensim.downloader as api
import nltk
from nltk.corpus import stopwords


print(api.info('20-newsgroups'))
news_corpus = api.load('20-newsgroups')
nltk.download('stopwords')
nltk.download('punkt')
stoplist = stopwords.words('english')

tokens = []


def pre_process(document):
    sentences = document["data"]
    for sentence in nltk.sent_tokenize(sentences.lower()):
        # tokenizer = nltk.RegexpTokenizer(r"\w+")
        # for word in tokenizer.tokenize(sentence):
        for word in nltk.word_tokenize(sentence):
            if word not in [x.lower() for x in stoplist]:
                tokens.append(word)
                # ToDo: remove duplicate words
    print(tokens)
    return


for document in news_corpus:
    print(document["data"])
    pre_process(document)
    break
