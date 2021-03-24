import nltk
import spacy
import pandas as pd
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class Word2VecClass:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_lg')

    def downloadfiles(self):
        file = pd.read_excel('Posts from 2019-10-01 to 2019-10-31.xls')
        # file = pd.read_excel('./5_Word2Vec/Camera/data/Posts from 2019-10-01 to 2019-10-31.xls')
        file = file[["Unnamed: 1", "Unnamed: 2", "Unnamed: 3"]]
        file.columns = file.iloc[0]
        file = file.iloc[1:]
        file = file.reset_index(drop=True)
        file.Contents = file.Contents.apply(lambda x: str(x))
        contents = file['Contents'].str.lower().str.strip().values.tolist()
        return contents

    def getlemmapos(self, contents):

        pjt_list_lemma = []
        pjt_list_pos = []

        for idx, doc in enumerate(self.nlp.pipe(contents)):

            print(idx, "/", len(contents))

            nlp_text = self.nlp(doc.text)
            temp_list_lemma = []
            temp_list_pos = []

            for word in nlp_text:
                temp_list_lemma.append(word.lemma_)
                temp_list_pos.append(word.pos_)

            pjt_list_lemma.append(temp_list_lemma)
            pjt_list_pos.append(temp_list_pos)

        return pjt_list_lemma, pjt_list_pos

    def cleandata(self, pjt_list_lemma, pjt_list_pos):

        pos_list = ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'CCONJ', 'DET', 'INTJ',
                    'PART', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
        index_list = []

        for index1, value1 in enumerate(pjt_list_pos):
            # print(index1, value1)
            temp_index_list = []
            for index2, value2 in enumerate(pjt_list_pos[index1]):
                if value2 not in pos_list:
                    temp_index_list.append(index2)
            index_list.append(temp_index_list)

        pjt_list_lemma_update = []

        for index, (index_value, lemma_value) in enumerate(zip(index_list, pjt_list_lemma)):
            # print(index, index_value, lemma_value)
            temp_lemma_update = []
            for noun_index in index_value:
                # print(noun_index)
                # print(lemma_value[noun_index])
                temp_lemma_update.append(lemma_value[noun_index])
            pjt_list_lemma_update.append(temp_lemma_update)

        item_list = ['-PRON-', '.', ',', '!', '?', '$', '#', '=', '*', '/', '[', ']',
                     '（', '(', ')', '）', '-', '–', ':', '️', '  ', 'rt', '\n', '\n ',
                     ';', '…', '...', '....', '️', '_', '>', '<', '|', '"', "'"]

        pjt_list = []
        for lst in pjt_list_lemma_update:
            temp_list = []
            for element in lst:
                if element not in item_list:
                    if 'https://' not in element and 'http://' not in element and '@' not in element:
                        temp_list.append(element)
            pjt_list.append(temp_list)

        return pjt_list

    def list_data(self, pjt_list):

        for index in range(len(pjt_list)):
            # print(pjt_list1_copy[index])
            print(pjt_list[index])
            print("##########################################")

    def trainandplot(self, pjt_list, num_voca):

        # 각 문장별로, spacy lemma 이용해서 다 원형으로 바꿔주고
        # 바뀐 문장 (list형태)를 가지고 embedding 실행 (tf-idf 방법으로 계산)
        # 이후에 2차원 [x,y]로 만들어서 맵핑하기

        ######################
        #stopword 목록 다운로드
        nltk.download('stopwords')
        stop_words = set(nltk.corpus.stopwords.words('english'))

        ######################
        # train
        sentences = pjt_list
        model = Word2Vec(size=150, window=5, min_count=2, workers=10, iter=10) # Word2Vec(size=150, window=10, min_count=2, workers=10, iter=10)
        model.build_vocab(sentences)

        # model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)  # train word vectors # >> corpus_count >> tf-idf
        # __main__:78: DeprecationWarning: Call to deprecated `iter` (Attribute will be removed in 4.0.0, use self.epochs instead).
        model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)  # train word vectors # >> corpus_count >> tf-idf

        ######################
        #test
        # w1 = 'p30'
        # model.wv.most_similar(positive=w1)

        ######################
        # plot
        # len(list(model.wv.vocab))
        vocab = list(model.wv.vocab)[:num_voca]
        X = model[vocab]
        tsne = TSNE(n_components=2)
        X_tsne = tsne.fit_transform(X)
        df = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y'])

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(df['x'], df['y'])

        for word, pos in df.iterrows():
            ax.annotate(word, pos)

        plt.show()
        ######################

    def main(self, num_voca):
        contents = self.downloadfiles()
        pjt_list_lemma, pjt_list_pos = self.getlemmapos(contents)
        pjt_list = self.cleandata(pjt_list_lemma, pjt_list_pos)
        # self.list_data(pjt_list)
        self.trainandplot(pjt_list, num_voca=100) # num_voca, tested 100


if __name__ == '__main__':
    w2v = Word2VecClass()
    w2v.main(num_voca=10000)