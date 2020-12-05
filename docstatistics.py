from tabulate import tabulate
import statistics
from gensim.corpora import Dictionary
import matplotlib.pyplot as plt

def main():
    dictionary:Dictionary = Dictionary.load('vocabulary1.gensim')
    vocab = dictionary.cfs
    stats = [[dictionary.num_docs, sum(vocab.values()), len(vocab.keys()),
              min(vocab.values()), max(vocab.values()), statistics.mean(vocab.values()), statistics.stdev(vocab.values())]]
    print(tabulate(stats, headers=["# of docs",
                                   "# of words", "# of unique words",
                                   "min words", "max words", "mean words", "std of words"]))
    with open('doc_statistics.txt', 'w') as outputfile:
        outputfile.write(
            tabulate(stats, headers=["# of docs",
                                     "# of words", "# of unique words",
                                     "min words", "max words", "mean words", "std of words"]))


    new_dict = sorted(vocab.items(), key=lambda x: x[1], reverse=True)[:100]

    # print(new_dict)
    new_vocab = {}
    for k,v in new_dict:
        new_vocab[dictionary[k]]=v

    # print(new_vocab)
    plt.title('Data Statistics')
    plt.xlabel('Word')
    plt.xticks(rotation=70)
    plt.ylabel('Count')
    plt.plot(list(new_vocab.keys()), list(new_vocab.values()))
    plt.savefig('statistics.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()
