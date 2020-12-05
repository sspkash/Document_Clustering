from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.cluster import normalized_mutual_info_score
import nltk
from nltk.corpus import stopwords
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from gensim.models import Word2Vec
# from bokeh.io import output_notebook
# from bokeh.plotting import figure, show, save
#
# output_notebook()

def main():
    news_corpus = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'), shuffle=False)
    nltk.download('stopwords')
    nltk.download('punkt')
    stoplist = stopwords.words('english')

    # Vocabulary 1
    tokens = []
    i = 0
    documents = []
    cleaned_documents = []
    for doc in news_corpus.data:
        # print(doc)
        words = gensim.utils.simple_preprocess(doc, True, 3)
        words_list_per_doc = []
        for word in words:
            if word not in stoplist:
                tokens.append(word)
                words_list_per_doc.append(word)
        documents.append(TaggedDocument(words_list_per_doc, [i]))
        cleaned_documents.append(words_list_per_doc)
        i = i+1
    # print(documents)
    # vocabulary1 = gensim.corpora.Dictionary.load('vocabulary1.gensim')
    model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40, window=10)
    model.build_vocab(documents)
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
    nmi_score = normalized_mutual_info_score(news_corpus.target,model)
    print(nmi_score)
    # model = Doc2Vec(documents)
    model.save('doc2vec_voc1.model')
    # model.save_word2vec_format()
    # model.save_word2vec_format('word2vec_voc1.bin')
    # print(model.docvecs.vector_size)

    tsne_model = TSNE(n_components=2,perplexity=10,random_state=2017,learning_rate=40)
    tsne_d2v = tsne_model.fit_transform(model.docvecs.vectors_docs)
    print(tsne_d2v.shape)
    plt.figure(figsize=(10, 10))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel('tsne - 1', fontsize=20)
    plt.ylabel('tsne - 2', fontsize=20)
    plt.title("t-sne for Doc2Vec Model 1", fontsize=20)
    plt.scatter(tsne_d2v[:, 0], tsne_d2v[:, 1], c=news_corpus.target[:], cmap=plt.cm.tab20)
    plt.savefig('tsne_doc2vecmodel1.png')
    plt.show()


    # for doc in cleaned_documents:
    #     vector = model.infer_vector(doc)
    #     print(vector)

    # ######################################
    # # doing word2vec for word embeddings
    # word2vecmodel_voc1 = Word2Vec(cleaned_documents,
    #                               min_count=100,  # Ignore words that appear less than this
    #                               # size=200,      # Dimensionality of word embeddings
    #                               workers=4,  # Number of processors (parallelisation)
    #                               window=20,  # Context window for words during training
    #                               iter=30)  # Number of epochs training over corpus
    #
    # word2vecmodel_voc1.save('word2vecmodel_voc1')  # saving word2vec for voc1
    # tsne_model = TSNE(n_components=2,
    #                   n_jobs=4,
    #                   random_state=2018)
    #
    # tsne_w2v = word2vecmodel_voc1.wv[word2vecmodel_voc1.wv.vocab]
    # X_tsne = tsne_model.fit_transform(tsne_w2v)
    #
    # tsne_w2v_df = pd.DataFrame(data=X_tsne, columns=["x", "y"])
    # plot = figure(title="t-SNE  of {} W2V Topics".format(20),
    #               plot_width=900, plot_height=700)
    # plot.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1])
    # show(plot)
    # save(plot, '{}.html'.format("tSNEw2v"))



if __name__ == '__main__':
    main()