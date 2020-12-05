import pandas as pd
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from gensim.models.word2vec import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def main():
    news_df = pd.read_pickle("news_df.pkl")
    news_df2 = pd.read_pickle("news_df2.pkl")
    # # Doc2Vec Model - Vocab 1
    # documents = []
    # i = 0
    # for doc_rokens in news_df['DocTokens']:
    #     documents.append(TaggedDocument(doc_rokens, [i]))
    #     i = i+1
    #
    # doc2vec_model_1 = Doc2Vec(vector_size=50, min_count=2, epochs=40,workers=4)
    # doc2vec_model_1.build_vocab(documents)
    # doc2vec_model_1.train(documents, total_examples=doc2vec_model_1.corpus_count, epochs=doc2vec_model_1.epochs)
    # doc2vec_model_1.save('doc2vec_voc1.model')
    #
    # tsne_model = TSNE(n_components=2, perplexity=50, early_exaggeration=15, learning_rate=500, n_iter=2000)
    # tsne_d2v = tsne_model.fit_transform(doc2vec_model_1.docvecs.vectors_docs)
    # print(tsne_d2v.shape)
    # plt.figure(figsize=(10, 10))
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=14)
    # plt.xlabel('tsne - 1', fontsize=20)
    # plt.ylabel('tsne - 2', fontsize=20)
    # plt.title("t-sne for Doc2Vec Model 1", fontsize=20)
    # plt.scatter(tsne_d2v[:, 0], tsne_d2v[:, 1], c=news_df['Topic'], cmap=plt.cm.tab20)
    # plt.savefig('tsne_doc2vecmodel_1.png')
    # plt.show()

    # Doc2Vec Model - Vocab 2
    # news_df2 = pd.read_pickle("news_df2.pkl")
    # documents2 = []
    # i = 0
    # for doc_tokens in news_df2['DocTokens']:
    #     documents2.append(TaggedDocument(doc_tokens, [i]))
    #     i = i + 1
    #
    # doc2vec_model_2 = Doc2Vec(vector_size=50, min_count=2, epochs=40, workers=4)
    # doc2vec_model_2.build_vocab(documents2)
    # doc2vec_model_2.train(documents2, total_examples=doc2vec_model_2.corpus_count,
    #                       epochs=doc2vec_model_2.epochs)
    # doc2vec_model_2.save('doc2vec_voc2.model')
    #
    # tsne_model2 = TSNE(n_components=2, perplexity=50, early_exaggeration=15, learning_rate=500, n_iter=2000)
    # tsne_d2v_2 = tsne_model2.fit_transform(doc2vec_model_2.docvecs.vectors_docs)
    # print(tsne_d2v_2.shape)
    # plt.figure(figsize=(10, 10))
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=14)
    # plt.xlabel('tsne - 1', fontsize=20)
    # plt.ylabel('tsne - 2', fontsize=20)
    # plt.title("t-sne for Doc2Vec Model 2", fontsize=20)
    # plt.scatter(tsne_d2v_2[:, 0], tsne_d2v_2[:, 1], c=news_df2['Topic'], cmap=plt.cm.tab20)
    # plt.savefig('tsne_doc2vecmodel_2.png')
    # plt.show()

    # ######################################
    # # doing word2vec for word embeddings
    # word2vecmodel_voc1 = Word2Vec(news_df['DocTokens'],
    #                               min_count=100,  # Ignore words that appear less than this
    #                               # size=200,      # Dimensionality of word embeddings
    #                               workers=4,  # Number of processors (parallelisation)
    #                               window=20,  # Context window for words during training
    #                               iter=30)  # Number of epochs training over corpus
    #
    # word2vecmodel_voc1.save('word2vecmodel_voc1.model')  # saving word2vec for voc1
    # tsne_model = TSNE(n_components=2,
    #                   n_jobs=4,
    #                   random_state=2018)
    #
    # tsne_w2v = word2vecmodel_voc1.wv[word2vecmodel_voc1.wv.vocab]
    # X_tsne_w2v_1 = tsne_model.fit_transform(word2vecmodel_voc1.wv.vectors)
    # plt.figure(figsize=(10, 10))
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=14)
    # plt.xlabel('tsne - 1', fontsize=20)
    # plt.ylabel('tsne - 2', fontsize=20)
    # plt.title("t-sne for Word2Vec Model 1", fontsize=20)
    # plt.scatter(X_tsne_w2v_1[:, 0], X_tsne_w2v_1[:, 1])
    # plt.savefig('tsne_word2vecmodel_1.png')
    # plt.show()

    # word2vecmodel_voc2 = Word2Vec(news_df2['DocTokens'],
    #                               min_count=100,  # Ignore words that appear less than this
    #                               # size=200,      # Dimensionality of word embeddings
    #                               workers=4,  # Number of processors (parallelisation)
    #                               window=20,  # Context window for words during training
    #                               iter=30)  # Number of epochs training over corpus
    #
    # word2vecmodel_voc2.save('word2vecmodel_voc2.model')
    # tsne_model2 = TSNE(n_components=2,
    #                   n_jobs=4,
    #                   random_state=2018)
    #
    # # tsne_w2v_2 = word2vecmodel_voc2.wv[word2vecmodel_voc2.wv.vocab]
    # X_tsne_w2v_2 = tsne_model2.fit_transform(word2vecmodel_voc2.wv.vectors)
    # plt.figure(figsize=(10, 10))
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=14)
    # plt.xlabel('tsne - 1', fontsize=20)
    # plt.ylabel('tsne - 2', fontsize=20)
    # plt.title("t-sne for Word2Vec Model 2", fontsize=20)
    # plt.scatter(X_tsne_w2v_2[:, 0], X_tsne_w2v_2[:, 1])
    # plt.savefig('tsne_word2vecmodel_2.png')
    # plt.show()

if __name__ == '__main__':
    main()