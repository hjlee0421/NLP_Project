import os
import spacy
import neuralcoref
import pandas as pd

# en-core-web-lg==2.1.0
# python -m spacy download en_core_web_lg
# pip install -r requirements.txt
# pip install spacy==2.1.0
# pkg-resources==0.0.0

# NEED TO UPDATE WITH DEPENDENCY BASED SEARCHING ALGORITHM

class Adjective_Analysis_with_Depth:

    def __init__(self):
        self.nlp = spacy.load('en_core_web_lg', disable=['ner', 'textcat', 'entity_ruler', 'sentencizer'])
        neuralcoref.add_to_pipe(self.nlp)
        #self.nlp.add_pipe(self.nlp.create_pipe("merge_noun_chunks"))

    @staticmethod
    def data_pre_processing(data_frame):
        contents = data_frame['Contents'].str.lower().str.strip().values.tolist()
        date = data_frame['Date (KST)'].values.tolist()
        url = data_frame['URL'].values.tolist()
        return contents, date, url

    # check ########################
    @staticmethod
    def post_processing(result_list):
        print("post_processing", pd.Timestamp.now())
        if result_list:
            tmp_df = pd.DataFrame(result_list)
            tmp_df[1] = tmp_df[1].apply(lambda x: str(x))
            tmp_df_modi = tmp_df.stack().apply(pd.Series).stack().unstack(1)
            tmp_df_modi = tmp_df_modi.reset_index()
            date_url_copy_df = tmp_df_modi.loc[tmp_df_modi['level_1'] == 0, ['level_0', 1, 2]]
            tmp_df_modi = tmp_df_modi.drop([1, 2], axis=1)
            tmp_df_modi = tmp_df_modi.set_index(['level_0'])
            tmp_df_modi2 = tmp_df_modi.join(date_url_copy_df.set_index(['level_0']))
            tmp_df_modi2 = tmp_df_modi2.reset_index()
            tmp_df_modi2 = tmp_df_modi2.set_index(['level_0', 'level_1'])

            tmp_df_modi2[['brand', 'adjective', 'dependency_type', 'contents']] = pd.DataFrame(tmp_df_modi2[0].values.tolist(), index=tmp_df_modi2.index)

            analysis_result_df = tmp_df_modi2[['adjective', 'contents', 1, 2]]
            analysis_result_df = analysis_result_df.drop_duplicates(['adjective', 'contents'])
            analysis_result_df.columns = ['Adjective', 'Contents', 'Date (KST)', 'URL']
            print("post_processing", pd.Timestamp.now())
            return analysis_result_df

    def ADJ_Analysis(self, DF, keyword, POS, DEPTH):

        self.nlp.vocab[keyword]
        POS = POS.upper()
        f_token = keyword.lower()

        if not isinstance(DF, pd.DataFrame):
            print("error, check your input datatype, must be pandas dataframe with 'Contents', 'Date (KST)', 'URL'")

        else:
            contents, date, url = self.data_pre_processing(DF) # , guid
            adj_list = []

            for idx, doc in enumerate(self.nlp.pipe(contents)):

                print(idx, "/", len(contents))
                doc = self.nlp(doc._.coref_resolved)
                adj_noun_pair = []

                for sent in doc.sents:

                    if f_token in sent.text:

                        for token in sent:

                            if token.text == f_token:

                                ########################################################################################
                                ########################################################################################

                                if token.head:
                                    if DEPTH >= 1:
                                        # depth 1 head
                                        if token.head.text != f_token and token.head.pos_ == POS:
                                            adj_noun_pair.append([f_token, token.head.text, 'head', sent])

                                        if DEPTH >= 2:
                                            # depth 2 head head
                                            if token.head.head:
                                                if token.head.head.text != f_token and token.head.head.pos_ == POS:
                                                    adj_noun_pair.append([f_token, token.head.head.text, 'head_head', sent])

                                                if DEPTH >= 3:
                                                    # dpeth 3 head head head
                                                    if token.head.head.head:
                                                        if token.head.head.head.text != f_token and token.head.head.head.pos_ == POS:
                                                            adj_noun_pair.append([f_token, token.head.head.head.text, 'head_head_head', sent])

                                                    # dpeth 3 head head child
                                                    if token.head.head.children:
                                                        for head_head_child in token.head.head.children:
                                                            if head_head_child.text != f_token and head_head_child.pos_ == POS:
                                                                adj_noun_pair.append([f_token, head_head_child.text, 'head_head_child', sent])

                                            #depth 2 head child
                                            if token.head.children:
                                                for head_child in token.head.children:
                                                    if head_child.text != f_token and head_child.pos_ == POS:
                                                        adj_noun_pair.append([f_token, head_child.text, 'head_child', sent])

                                                    if DEPTH >= 3:
                                                        # dpeth 3 head child head
                                                        if head_child.head:
                                                            if head_child.head.text != f_token and head_child.head.pos_ == POS:
                                                                adj_noun_pair.append([f_token, head_child.head.text, 'head_child_head', sent])

                                                        # dpeth 3 head child child
                                                        for head_child_child in head_child.children:
                                                            if head_child_child.text != f_token and head_child_child.pos_ == POS:
                                                                adj_noun_pair.append([f_token, head_child_child.text, 'head_child_child', sent])

                                ########################################################################################

                                if token.children:
                                    if DEPTH >= 1:
                                        # depth 1 child
                                        for child in token.children:
                                            if child.text != f_token and child.pos_ == POS:
                                                adj_noun_pair.append([f_token, child.text, 'child', sent])

                                            if DEPTH >= 2:
                                                # depth 2 child head
                                                if child.head:
                                                    if child.head.text != f_token and child.head.pos_ == POS:
                                                        adj_noun_pair.append([f_token, child.head.text, 'child_head', sent])

                                                    if DEPTH >= 3:
                                                        # dpeth 3 child head head
                                                        if child.head.head:
                                                            if child.head.head.text != f_token and child.head.head.pos_ == POS:
                                                                adj_noun_pair.append([f_token, child.head.head.text, 'child_head_head', sent])

                                                        # dpeth 3 child head child
                                                        if child.head.children:
                                                            for child_head_child in child.head.children:
                                                                if child_head_child.text != f_token and child_head_child.pos_ == POS:
                                                                    adj_noun_pair.append([f_token, child_head_child.text, 'child_head_child', sent])

                                                # depth 2 child child
                                                if child.children:
                                                    for child_child in child.children:
                                                        if child_child.text != f_token and child_child.pos_ == POS:
                                                            adj_noun_pair.append([f_token, child_child.text, 'child_child', sent])

                                                        if DEPTH >= 3:
                                                            # dpeth 3 child child head
                                                            if child_child.head:
                                                                if child_child.head.text != f_token and child_child.head.pos_ == POS:
                                                                    adj_noun_pair.append([f_token, child_child.head.text, 'child_child_head', sent])

                                                            # dpeth 3 child child child
                                                            if child_child.children:
                                                                for child_child_child in child_child.children:
                                                                    if child_child_child.text != f_token and child_child_child.pos_ == POS:
                                                                        adj_noun_pair.append([f_token, child_child_child.text, 'child_child_child', sent])

                                ########################################################################################
                                ########################################################################################

                if adj_noun_pair:
                    adj_list.append([adj_noun_pair, date[idx], url[idx]]) # , guid[idx]

        adj_df = self.post_processing(adj_list)

        return adj_df

if __name__ == '__main__':

    file = pd.read_excel("C:\\work\\8_AOM\\SLCC_Alert\\data\\Posts from 2019-01-01 to 2019-09-30.xls")
    file = file[["Unnamed: 1", "Unnamed: 2", "Unnamed: 3"]]
    file.columns = file.iloc[0]
    file = file.iloc[1:]
    file = file.reset_index(drop=True)
    file.Contents = file.Contents.apply(lambda x: str(x))
    DATA = pd.DataFrame(file)

    DATA = DATA[['Contents', 'Date (KST)', 'URL']]
    DATA = DATA.dropna()
    DATA = DATA[:1000]

    d = Adjective_Analysis_with_Depth()
    print("start1")
    result_df_1 = d.ADJ_Analysis(DATA, "samsung", "ADJ", 1)
    print("start2")
    result_df_2 = d.ADJ_Analysis(DATA, "samsung", "ADJ", 2)
    print("start3")
    result_df_3 = d.ADJ_Analysis(DATA, "samsung", "ADJ", 3)

    #result_df_1.to_csv("C:\\work\\depth_11.csv", index=False, encoding="utf-8")
    #result_df_2.to_csv("C:\\work\\depth_22.csv", index=False, encoding="utf-8")
    #result_df_3.to_csv("C:\\work\\depth_33.csv", index=False, encoding="utf-8")