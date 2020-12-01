from sklearn.datasets import fetch_20newsgroups
import nltk
from nltk.corpus import stopwords
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.manifold import TSNE
import pandas as pd

from bokeh.io import output_notebook
from bokeh.plotting import figure, show, save

output_notebook()

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
        # if i==10:
        #     break
    print(documents)
    vocabulary1 = gensim.corpora.Dictionary.load('vocabulary1.gensim')
    model = Doc2Vec(documents)
    model.save('doc2vec_voc1.model')
    model.save_word2vec_format()
    #print(model.docvecs.vector_size)

    tsne_model = TSNE(n_jobs=4,
                      early_exaggeration=4,
                      n_components=2,
                      verbose=1,
                      random_state=2018)
    tsne_d2v = tsne_model.fit_transform(model.docvecs.vectors_docs)

    # Putting the tsne information into sq
    tsne_d2v_df = pd.DataFrame(data=tsne_d2v, columns=["x", "y"])
    plot = figure(title="t-SNE  of {} D2V Topics".format(20),
                  plot_width=900, plot_height=700)
    # ToDO: color
    plot.scatter(x=tsne_d2v[:, 0], y=tsne_d2v[:, 1])
    show(plot)
    save(plot, '{}.html'.format("tSNEd2v"))

    # for doc in cleaned_documents:
    #     vector = model.infer_vector(doc)
    #     print(vector)




if __name__ == '__main__':
    main()