import spacy
import pandas as pd

class Adjective_Analysis:

    def __init__(self, model_name='en_core_web_lg',
                 disable_list=['ner', 'textcat', 'entity_ruler', 'sentencizer']):

        print(f'model initialize:{model_name}', pd.Timestamp.now())
        self.nlp = spacy.load(model_name, disable=disable_list)
        print('model loaded', pd.Timestamp.now(), f'disabled:{disable_list}')

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
            tmp_df[1] = tmp_df[1].apply(lambda x: str(x))
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
                            if tok.head.text == f_token:
                                if tok.pos_ == p_tag and tok.text != f_token:
                                    adj_noun_pair.append([f_token, tok.text, 'head_a', sent])
                                    token_flag = 1

                        # match based on noun ###################
                        if tok.text == f_token:

                            subject_flag = 1

                            # print("h ", tok.head, " ", tok.head.pos_)
                            if tok.head.pos_ == p_tag  and tok.head.text != f_token and token_flag == 0:
                                # print("h ", tok.head)
                                adj_noun_pair.append([f_token, tok.head.text, 'head_b', sent])
                                token_flag = 1
                            if token_flag == 0:
                                for child in tok.children:
                                    # print("c ", child)
                                    if child.pos_ == p_tag and child.text != f_token:
                                        adj_noun_pair.append([f_token, child.text, 'child', sent])
                                        token_flag = 1
                            if token_flag == 0:
                                for ancestor in tok.ancestors:
                                    # print("a ", ancestor)
                                    if ancestor.pos_ == p_tag and ancestor.text != f_token:
                                        adj_noun_pair.append([f_token, ancestor.text, 'ancestor', sent])
                                        token_flag = 1
                            if token_flag == 0:
                                for desc in tok.subtree:
                                    # print("d ", dec)
                                    if desc.pos_ == p_tag and desc.text != f_token:
                                        adj_noun_pair.append([f_token, desc.text, 'descendent', sent])
                                        token_flag = 1

                        if token_flag == 1:
                            potential_token.add(tok.text)


                    # look for connection if subject is in sentence ###################
                    if subject_flag == 1:
                        # indirect ###################
                        for r_token in sent:

                            find_token_flag = 0
                            find_pos_flag = 0
                            find_pos_token = ""
                            token_flag = 0

                            if r_token.text != f_token and r_token.text not in potential_token:

                                if token_flag == 0:
                                    for child in r_token.children:
                                        if child.text not in potential_token:
                                            if f_token == child.text:
                                                find_token_flag = 1
                                            if p_tag == child.pos_:
                                                find_pos_flag = 1
                                                find_pos_token = child.text

                                    if find_token_flag == 1 and find_pos_flag == 1:
                                        adj_noun_pair.append([f_token, find_pos_token, 'child_v2', sent])
                                        potential_token.add(find_pos_token)
                                        token_flag = 1

                                if token_flag == 0:
                                    for ancestor in r_token.ancestors:
                                        if ancestor.text not in potential_token:
                                            if f_token == ancestor.text:
                                                find_token_flag = 1
                                            if p_tag == ancestor.pos_:
                                                find_pos_flag = 1
                                                find_pos_token = ancestor.text

                                    if find_token_flag == 1 and find_pos_flag == 1:
                                        adj_noun_pair.append([f_token, find_pos_token, 'anchestor_v2', sent])
                                        potential_token.add(find_pos_token)
                                        token_flag = 1

                                if token_flag == 0:
                                    for desc in r_token.subtree:
                                        if desc.text not in potential_token:
                                            if f_token == desc.text:
                                                find_token_flag = 1
                                            if p_tag == desc.pos_:
                                                find_pos_flag = 1
                                                find_pos_token = desc.text

                                    if find_token_flag == 1 and find_pos_flag == 1:
                                        adj_noun_pair.append([f_token, find_pos_token, 'descendent_v2', sent])
                                        token_flag = 1
                                        potential_token.add(find_pos_token)

                if adj_noun_pair:
                    adj_list.append([adj_noun_pair, date[idx], url[idx]]) # , guid[idx]

            print("NLP Done", pd.Timestamp.now())

        return adj_list