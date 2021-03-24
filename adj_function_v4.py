import spacy
import neuralcoref
import pandas as pd

class Adjective_Analysis:

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

    def ADJ_Analysis(self, DF, keyword, POS):

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

                    negation_tokens = []
                    negation_tokens_heads = []

                    if f_token in sent.text:

                        #print("sentence = ", sent)

                        for token in sent:

                            if token.dep_ == "neg":
                                negation_tokens.append(token)
                                negation_tokens_heads.append(token.head)

                        for token in sent:

                            if token.text == f_token:

                                token_not_used = True
                                adj_set = set()

                                # checking ancestors
                                if token_not_used:

                                    for ancestor in token.ancestors:
                                        if ancestor.text != f_token and ancestor.pos_ == POS:
                                            if ancestor in negation_tokens_heads and "~" + ancestor.text not in adj_set:
                                                adj_noun_pair.append([f_token, '~' + ancestor.text, 'ancestor', sent])
                                                adj_set.add('~' + ancestor.text)
                                                #print('~' + ancestor.text, "ancestor")
                                                token_not_used = False
                                            elif ancestor in negation_tokens_heads and ancestor.text not in adj_set:
                                                adj_noun_pair.append([f_token, ancestor.text, 'ancestor', sent])
                                                adj_set.add(ancestor.text)
                                                #print(ancestor.text, "ancestor")
                                                token_not_used = False

                                            if ancestor.n_rights > 1 and list(ancestor.rights)[0].pos_ == "CCONJ" and list(ancestor.rights)[1].text != f_token and list(ancestor.rights)[
                                                1].pos_ == POS:
                                                if list(ancestor.rights)[1] in negation_tokens_heads:
                                                    adj_noun_pair.append([f_token, "~" + list(ancestor.rights)[1].text, 'ancestor_v2', sent])
                                                    adj_set.add("~" + list(ancestor.rights)[1].text)
                                                    #print("~" + list(ancestor.rights)[1].text, "ancestor_v2")
                                                else:
                                                    adj_noun_pair.append([f_token, list(ancestor.rights)[1].text, 'ancestor_v2', sent])
                                                    adj_set.add(list(ancestor.rights)[1].text)
                                                    #print(list(ancestor.rights)[1].text, "ancestor_v2")

                                # checking descendants
                                if token_not_used:

                                    for descendant in token.subtree:
                                        if descendant.text != f_token and descendant.pos_ == POS:
                                            if descendant in negation_tokens_heads and "~" + descendant.text not in adj_set:
                                                adj_noun_pair.append([f_token, "~" + descendant.text, 'descendant', sent])
                                                adj_set.add("~" + descendant.text)
                                                #print("~" + descendant.text, "descendant")
                                                token_not_used = False
                                            elif descendant in negation_tokens_heads and descendant.text not in adj_set:
                                                adj_noun_pair.append([f_token, descendant.text, 'descendant', sent])
                                                adj_set.add(descendant.text)
                                                #print(descendant.text, "descendant")
                                                token_not_used = False

                                            if descendant.n_rights > 1 and list(descendant.rights)[0].pos_ == "CCONJ" and list(descendant.rights)[1].text != f_token and list(descendant.rights)[
                                                1].pos_ == POS:
                                                if list(descendant.rights)[1] in negation_tokens_heads:
                                                    adj_noun_pair.append([f_token, "~" + list(descendant.rights)[1].text, 'descendant_v2', sent])
                                                    adj_set.add("~" + list(descendant.rights)[1].text)
                                                    #print("~" + list(descendant.rights)[1].text, "descendant_v2")
                                                else:
                                                    adj_noun_pair.append([f_token, list(descendant.rights)[1].text, 'descendant_v2', sent])
                                                    adj_set.add(list(descendant.rights)[1].text)
                                                    #print(list(descendant.rights)[1].text, "descendant_v2")

                                ############################################################################################################

                                # checking verb connected adjectives
                                if token_not_used:

                                    for verb in token.ancestors:
                                        if verb.text != f_token and verb.pos_ == "VERB":

                                            if verb in negation_tokens_heads:

                                                for verb_connected_ajd in verb.children: # subtree for more adjs

                                                    if verb_connected_ajd.text != f_token and verb_connected_ajd.pos_ == POS:

                                                        if "~" + verb_connected_ajd.text not in adj_set:
                                                            adj_noun_pair.append([f_token, "~" + verb_connected_ajd.text, 'verb_connected_ajd', sent])
                                                            adj_set.add("~" + verb_connected_ajd.text)
                                                            #print("~" + verb_connected_ajd.text, "verb_connected_ajd")
                                                            token_not_used = False
                                                            if verb_connected_ajd.n_rights > 1 and \
                                                                    list(verb_connected_ajd.rights)[0].pos_ == "CCONJ" and \
                                                                    list(verb_connected_ajd.rights)[1].text != f_token and \
                                                                    list(verb_connected_ajd.rights)[1].pos_ == POS:
                                                                adj_noun_pair.append(
                                                                    [f_token, "~" + list(verb_connected_ajd.rights)[1].text,
                                                                     'verb_connected_ajd_v2', sent])
                                                                adj_set.add("~" + list(verb_connected_ajd.rights)[1].text)
                                                                #print("~" + list(verb_connected_ajd.rights)[1].text,
                                                                #      "verb_connected_ajd_v2")

                                            else:

                                                for verb_connected_ajd in verb.children:  # subtree for more adjs

                                                    if verb_connected_ajd.text != f_token and verb_connected_ajd.pos_ == POS:

                                                        if verb_connected_ajd.text not in adj_set:
                                                            adj_noun_pair.append(
                                                                [f_token, verb_connected_ajd.text, 'verb_connected_ajd',
                                                                 sent])
                                                            adj_set.add(verb_connected_ajd.text)
                                                            #print(verb_connected_ajd.text, "verb_connected_ajd")
                                                            token_not_used = False
                                                            if verb_connected_ajd.n_rights > 1 and \
                                                                    list(verb_connected_ajd.rights)[0].pos_ == "CCONJ" and \
                                                                    list(verb_connected_ajd.rights)[1].text != f_token and \
                                                                    list(verb_connected_ajd.rights)[1].pos_ == POS:
                                                                adj_noun_pair.append(
                                                                    [f_token, list(verb_connected_ajd.rights)[1].text,
                                                                     'verb_connected_ajd_v2', sent])
                                                                adj_set.add(list(verb_connected_ajd.rights)[1].text)
                                                                #print(list(verb_connected_ajd.rights)[1].text,
                                                                #      "verb_connected_ajd_v2")

                if adj_noun_pair:
                    adj_list.append([adj_noun_pair, date[idx], url[idx]]) # , guid[idx]

        return adj_list