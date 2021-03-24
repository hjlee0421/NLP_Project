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
                        if tok.pos_ == "ADJ":
                            adj_noun_pair.append([f_token, tok.text, 'head_a', sent])

                if adj_noun_pair:
                    adj_list.append([adj_noun_pair, date[idx], url[idx]]) # , guid[idx]

            print("NLP Done", pd.Timestamp.now())

        return adj_list