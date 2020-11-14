import gensim
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt
from tabulate import tabulate
import statistics

news_corpus = fetch_20newsgroups(subset='all',remove=('headers', 'footers', 'quotes'),shuffle=False)

nltk.download('stopwords')
nltk.download('punkt')
stoplist = stopwords.words('english')


sentences = []
tokens = []
tokenCount = {}

no_of_docs = 0
wordsCount = []
dictionary = gensim.corpora.Dictionary()
BoW_corpus = []


for doc in news_corpus.data:
    no_of_docs = no_of_docs+1
    for sentence in nltk.sent_tokenize(doc):
        sentences.append(sentence)
        words = gensim.utils.simple_preprocess(sentence,True,3)
        if(len(words) != 0):
            wordsCount.append(len(words))
        words_list = []
        for word in words:
            if word not in stoplist:
                tokens.append(word)
                words_list.append(word)
        # print(words_list)
        BoW_corpus.append(dictionary.doc2bow(words_list, allow_update=True))
    if no_of_docs == 2:
        break


statistics = [[no_of_docs,len(sentences),len(tokens),len(set(tokens)),
              min(wordsCount),max(wordsCount), statistics.mean(wordsCount),statistics.stdev(wordsCount)]]
print(tabulate(statistics, headers=["# of docs", "# of sentences",
                               "# of words", "# of unique words",
                                     "min words", "max words","mean words","std of words"]))



for word in tokens:
    tokenCount[word] = tokenCount.get(word, 0) + 1

plt.title('statistics')
plt.xlabel('word')
plt.xticks(rotation=70)
plt.ylabel('count')
plt.plot(*zip(*sorted(tokenCount.items(),key=lambda x: x[1], reverse=True)[:100]))
plt.savefig('C:\\Users\\neeth\\Desktop\\ML\\Project\\statistics.png',dpi=300,bbox_inches='tight')
# plt.show()


# Build vocabulary
print(BoW_corpus)
id_words = [[(dictionary[id], count) for id, count in line] for line in BoW_corpus]
print(id_words)
