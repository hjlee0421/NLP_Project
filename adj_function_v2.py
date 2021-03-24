import spacy
import pandas as pd


class Adjective_Analysis:

    def __init__(self, model_name='en_core_web_lg',
                 disable_list=['ner', 'textcat', 'entity_ruler', 'sentencizer']):

        print(f'model initialize:{model_name}', pd.Timestamp.now())
        self.nlp = spacy.load(model_name, disable=disable_list)
        print('model loaded', pd.Timestamp.now(), f'disabled:{disable_list}')

    def recursive_search(self, sent, token_find, token_ind, count, flag):
        if token_find == token_ind.string:
            # print(0, token_ind)
            return True
        else:
            # print("recursive_search start:", pd.Timestamp.now())
            for tok in sent:
                if tok.i == count:
                    # print(tok, token_ind)
                    if flag <= 1:
                        if tok.head:
                            # print(1, tok, tok.head.i, count, tok.head)
                            if tok.head.i != count:
                                return self.recursive_search(sent, token_find, tok.head, tok.head.i, 1)

                    if flag <= 2:
                        for child in tok.children:
                            # print(2, tok, child.i, count, child)
                            if child.i != count:
                                return self.recursive_search(sent, token_find, child, child.i, 2)

                    if flag <= 3:
                        for ancestor in tok.ancestors:
                            # print(3, tok, ancestor.i, count, ancestor)
                            if ancestor.i != count:
                                return self.recursive_search(sent, token_find, ancestor, ancestor.i, 3)
            # print("recursive_search end:", pd.Timestamp.now())
            return False

    @staticmethod
    def data_pre_processing(data_frame):
        print("data_pre_processing", pd.Timestamp.now())
        data_frame = data_frame[['Contents', 'Date (KST)', 'URL']] # 'GUID',
        data_frame = data_frame.dropna()
        data_frame['Contents'] = data_frame['Contents'].str.lower()
        data_frame['Contents'] = data_frame['Contents'].str.strip()
        contents = data_frame["Contents"].values.tolist()
        date = data_frame['Date (KST)'].values.tolist()
        url = data_frame['URL'].values.tolist()
        #guid = data_frame['GUID'].values.tolist()
        print("data_pre_processing done", pd.Timestamp.now())
        return contents, date, url#, guid

    @staticmethod
    def post_processing(result_list):
        print("post_processing", pd.Timestamp.now())
        if result_list:
            tmp_df = pd.DataFrame(result_list)
            tmp_df_modi = tmp_df.stack().apply(pd.Series).stack().unstack(1)
            tmp_df_modi = tmp_df_modi.reset_index()
            date_url_copy_df = tmp_df_modi.loc[tmp_df_modi['level_1'] == 0,
                                               ['level_0', 1, 2]]
            tmp_df_modi = tmp_df_modi.drop([1, 2], axis=1)
            tmp_df_modi = tmp_df_modi.set_index(['level_0'])
            tmp_df_modi2 = tmp_df_modi.join(date_url_copy_df.set_index(['level_0']))
            tmp_df_modi2 = tmp_df_modi2.reset_index()
            tmp_df_modi2 = tmp_df_modi2.set_index(['level_0', 'level_1'])

            tmp_df_modi2[['brand', 'adjective', 'dependency_type', 'contents']] = \
                pd.DataFrame(tmp_df_modi2[0].values.tolist(), index=tmp_df_modi2.index)

            analysis_result_df = tmp_df_modi2[['adjective', 'contents', 1, 2]]
            analysis_result_df = analysis_result_df.drop_duplicates(['adjective', 'contents'])
            analysis_result_df.columns = ['Adjective', 'Contents', 'Date (KST)', 'URL']
            print("post_processing", pd.Timestamp.now())
            return analysis_result_df

    def ADJ_Analysis(self, df, p_token, p_tag, brand):

        p_tag = p_tag.upper()

        if not isinstance(df, pd.DataFrame):
            print("error, check your input datatype, must be pandas dataframe with 'Contents', 'Date (KST)', 'URL'")
        else:
            print("adj_analysis", pd.Timestamp.now())

            contents, date, url = self.data_pre_processing(df) # , guid

            adj_list = []

            print("NLP Start", pd.Timestamp.now())

            # for doc, i in zip(self.nlp.pipe(contents), range(len(contents))):
            for idx, doc in enumerate(self.nlp.pipe(contents)):
                print(idx, "/", len(contents), brand, p_token)
                adj_noun_pair = []
                f_token = p_token.lower()

                for sent in doc.sents:
                    subject_flag = 0
                    potential_token = set()

                    for tok in sent:

                        token_flag = 0
                        # direct ###################
                        if tok.head:
                            if tok.head.string == f_token:
                                if tok.pos_ == p_tag and tok.string != f_token:
                                    adj_noun_pair.append([f_token, tok.string, 'head_a', sent])

                        if tok.string == f_token:
                            subject_flag = 1
                            # print("h ", tok.head, " ", tok.head.pos_)
                            if tok.head.pos_ == p_tag \
                                    and tok.head.string != f_token \
                                    and token_flag == 0:
                                # print("h ", tok.head)
                                adj_noun_pair.append([f_token, tok.head.string, 'head_b', sent])
                                token_flag = 1
                            for child in tok.children:
                                # print("c ", child)
                                if child.pos_ == p_tag and child.string != f_token and token_flag == 0:
                                    adj_noun_pair.append([f_token, child.string, 'child', sent])
                                    token_flag = 1
                            for ancestor in tok.ancestors:
                                # print("a ", ancestor)
                                if ancestor.pos_ == p_tag \
                                        and ancestor.string != f_token \
                                        and token_flag == 0:
                                    adj_noun_pair.append([f_token, ancestor.string, 'ancestor', sent])
                                    token_flag = 1
                            for desc in tok.subtree:
                                # print("d ", dec)
                                if desc.pos_ == p_tag and desc.string != f_token and token_flag == 0:
                                    adj_noun_pair.append([f_token, desc.string, 'descendent', sent])
                                    token_flag = 1
                        if token_flag == 1:
                            potential_token.add(tok.string)

                    if subject_flag == 1:
                        # indirect ###################
                        for r_token in sent:
                            if r_token.pos_ == p_tag \
                                    and r_token.string != f_token \
                                    and r_token.string not in potential_token:
                                if self.recursive_search(sent, f_token, r_token, r_token.i, 0):
                                    adj_noun_pair.append([f_token, r_token.string, 'far', sent])

                if adj_noun_pair:
                    adj_list.append([adj_noun_pair, date[idx], url[idx]]) # , guid[idx]

            print("NLP Done", pd.Timestamp.now())

        return adj_list