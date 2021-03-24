from __future__ import unicode_literals, print_function
import re
import spacy
import random
import pandas as pd
from spacy.util import minibatch, compounding
from nltk.tokenize.treebank import TreebankWordDetokenizer

class Train_NER:

    def trim_entity_spans(self, data: list) -> list:
        """Removes leading and trailing white spaces from entity spans.
        Args:
            data (list): The data to be cleaned in spaCy JSON format.
        Returns:
            list: The cleaned data.
        """
        invalid_span_tokens = re.compile(r'\s')

        cleaned_data = []
        for text, annotations in data:
            entities = annotations['entities']
            valid_entities = []
            for start, end, label in entities:
                valid_start = start
                valid_end = end
                while valid_start < len(text) and invalid_span_tokens.match(
                        text[valid_start]):
                    valid_start += 1
                while valid_end > 1 and invalid_span_tokens.match(
                        text[valid_end - 1]):
                    valid_end -= 1
                valid_entities.append([valid_start, valid_end, label])
            cleaned_data.append([text, {'entities': valid_entities}])

        return cleaned_data


    def create_train_data(self, df, tag_list, word_list):
        print("start create train data")
        leng = 0
        temp = ""

        for i in range(len(tag_list)):
            if tag_list[i][0] == 'B' and (
                    tag_list[i - 1][0] == 'O' or tag_list[i - 1][0] == 'B' or len(tag_list[i - 1]) > 1) and tag_list[i - 1][0] != 'I':
                tag_list[i] = tag_list[i][2:]
                leng = 1
                temp = word_list[i]
            elif tag_list[i][0] == 'I':
                leng = leng + 1
                temp = temp + " " + word_list[i]
            elif tag_list[i][0] == 'O' or (tag_list[i][0] == 'B' and tag_list[i - 1][0] == 'I'):
                if leng > 1:
                    word_list[i - leng] = temp
                    for k in range(1, leng):
                        word_list[i - k] = ""
                        tag_list[i - k] = "O"
                    leng = 0
                    temp = ""
                if tag_list[i][0] == 'B':
                    tag_list[i] = tag_list[i][2:]
                    leng = 1

        WORD = pd.DataFrame(word_list)
        TAG = pd.DataFrame(tag_list)
        WORD.columns = ["Word"]
        TAG.columns = ["Tag"]

        new_df = pd.concat([df[["Sentence #"]], WORD, TAG], axis=1)
        new_df = new_df[new_df.Word != ""]

        word_series = new_df.groupby("Sentence #").apply(lambda s: [w for w in s["Word"].values.tolist()])
        word_series = word_series.tolist()

        tag_series = new_df.groupby("Sentence #").apply(lambda s: [t for t in s["Tag"].values.tolist()])
        tag_series = tag_series.tolist()

        tag_list = []
        for i in range(len(word_series)):
            temp = []
            for j in range(len(word_series[i])):
                if tag_series[i][j] != "O":
                    temp.append(j)
            ttemp = []
            for j in temp:
                ttemp.append([word_series[i][j], tag_series[i][j]])
            tag_list.append(ttemp)

        tag_list_check = []
        train_data = []

        print("start trim entity spans")

        for i in range(len(tag_list)):
            sentence = TreebankWordDetokenizer().detokenize(word_series[i])
            entities = []
            for j in range(len(tag_list[i])):
                start = sentence.find(tag_list[i][j][0])
                end = start + len(tag_list[i][j][0])
                tag = tag_list[i][j][1]
                tag_list_check.append(tag)
                entities.append((start, end, tag))
                # if (start, end, tag) not in entities:  # it was (start, end, tag)
                #     entities.append((start, end, tag))
            entities_dict = {"entities": entities}
            sentence_data = (sentence, entities_dict)
            train_data.append(sentence_data)

        train_data = self.trim_entity_spans(train_data)

        return train_data, tag_list_check

    def train_nlp_ner(self, train_data, tag_list_check):

        print("start train nlp ner")

        n_iter = 1
        nlp = spacy.blank("en")

        if "ner" not in nlp.pipe_names:
            ner = nlp.create_pipe("ner")
            nlp.add_pipe(ner)
        # otherwise, get it, so we can add labels to it
        else:
            ner = nlp.get_pipe("ner")

        for label in set(tag_list_check):
            # print(label)
            ner.add_label(label)

        optimizer = nlp.begin_training()
        # move_names = list(ner.move_names)

        sizes = compounding(1.0, 4.0, 1.001)
        # batch up the examples using spaCy's minibatch

        # error_list = []

        for itn in range(n_iter):
            random.shuffle(train_data)
            batches = minibatch(train_data, size=sizes)
            losses = {}
            i = 0
            for batch in batches:
                texts, annotations = zip(*batch)
                i = i + 1 * len(batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.35, losses=losses)
                # try: # overlap issue
                #     nlp.update(texts, annotations, sgd=optimizer, drop=0.35, losses=losses)
                # except ValueError:
                #     error_list.append([texts, annotations])

            print("Losses", losses)

        return nlp # , error_list



    def test_nlp_ner(self, nlp, test_text):
        print("start test nlp ner")

        doc = nlp(test_text)
        for ent in doc.ents:
            print(ent.label_, ent.text)

        ########### in jupyter notebook ###########

        # from spacy import displacy
        #
        # text = "When Sebastian Thrun started working on self-driving cars at Google in 2007, few people outside of the company took him seriously."
        #
        # nlp = spacy.load("en_core_web_sm")
        # doc = nlp(text)
        # spacy.displacy.render(doc, style="ent", page="true")
        # displacy.serve(doc, style="ent", page="true")

    def main(self):

        # nlp = spacy.load("en_core_web_lg")

        df = pd.read_csv('ner_dataset.txt', encoding="latin1")
        df = df.drop(['POS'], axis =1)
        df = df.fillna(method="ffill")
        df["Sentence #"] = df["Sentence #"].apply(lambda x: int(x.split(": ")[1]))

        tag_list = df["Tag"].tolist()
        word_list = df["Word"].tolist()

        train_data, tag_list_check = self.create_train_data(df, tag_list, word_list)

        nlp = self.train_nlp_ner(train_data, tag_list_check)

        # nlp, error_list = train_nlp_ner(train_data, tag_list_check)

        test_text = "Over the past few weeks, rumors of the Samsung Galaxy S10 Lite have ramped up."

        self.test_nlp_ner(nlp, test_text)

        nlp.to_disk("Trained_NER")

        # # save model to output directory
        # output_dir = Path('/data')
        # if output_dir is not None:
        #     output_dir = Path('/data')
        #     if not output_dir.exists():
        #         output_dir.mkdir()
        #     nlp.to_disk(output_dir)
        #     print("Saved model to", output_dir)
        #
        #     # test the saved model
        #     print("Loading from", output_dir)
        #     nlp2 = spacy.load(output_dir)
        #     for text, _ in TRAIN_DATA:
        #         doc = nlp2(text)
        #         print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
        #         print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])
        return nlp

if __name__ == '__main__':
    n = Train_NER()
    nlp = n.main()