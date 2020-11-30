from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import gensim
from tabulate import tabulate
import statistics

def main():
    news_corpus = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'), shuffle=False)
    nltk.download('stopwords')
    nltk.download('punkt')
    stoplist = stopwords.words('english')

    tokens = []
    tokenCount = {}
    wordsCount = []
    no_of_docs = 0

    for doc in news_corpus.data:
        # print(doc)
        no_of_docs = no_of_docs + 1
        words = gensim.utils.simple_preprocess(doc, True, 3)
        wordsCount.append(len(words))
        words_list_per_doc = []
        for word in words:
            if word not in stoplist:
                tokens.append(word)
                words_list_per_doc.append(word)
        # print(words_list_per_doc)

    stats = [[no_of_docs, len(tokens), len(set(tokens)),
              min(wordsCount), max(wordsCount), statistics.mean(wordsCount), statistics.stdev(wordsCount)]]
    print(tabulate(stats, headers=["# of docs",
                                   "# of words", "# of unique words",
                                   "min words", "max words", "mean words", "std of words"]))
    with open('pre_process_stats.txt', 'w') as outputfile:
        outputfile.write(
            tabulate(stats, headers=["# of docs",
                                     "# of words", "# of unique words",
                                     "min words", "max words", "mean words", "std of words"]))

    for word in tokens:
        tokenCount[word] = tokenCount.get(word, 0) + 1

    plt.title('New Corpus Data Statistics')
    plt.xlabel('Word')
    plt.xticks(rotation=70)
    plt.ylabel('Count')
    plt.plot(*zip(*sorted(tokenCount.items(),key=lambda x: x[1], reverse=True)[:100]))
    plt.savefig('C:\\Users\\neeth\\Desktop\\ML\\Project\\statistics.png',dpi=300,bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()