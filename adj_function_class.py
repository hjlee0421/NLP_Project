import re
import os
import crimson_scraper as cs
import crimson_dataproc as dp
import gbq_function
import datetime
import pandas as pd
import dev.adj_adhoc.adj_function_v3 as adj
import spacy
import sys
from spacy.tokenizer import Tokenizer

senti_score = pd.read_csv('./dev/adj_adhoc/SentiWords_2.0.rawdata')

#%%

class AdjectiveAnalysis:
    """
    형용사 분석 Standalone 버전
    """
    def __init__(self, monitor_id: int, from_date: str, to_date: str, keywords: list):
        """
        Args:
            monitor_id: cimson monitor id
            from_date: start date ex) 2019-01-01
            to_date:  last date ex) 2019-01-10
            keywords:  search keywods in list
        """

        print("init work")
        self.monitor_id = monitor_id
        self.from_date = from_date
        self.to_date = to_date
        self.keywords = keywords


    def crimson_data_dump(self):
        """
        step1. crimson data dump
        Returns: aggregated df for analysis, xls list to remove xls files later
        """

        print("crimson_data_dump work")
        f = cs.Crimson()
        f.crimson_login()
        f.download_data(monitor_id=self.monitor_id, start_date=self.from_date, end_date=self.to_date)
        g = dp.DataProcessor()
        result_df, xls_lst = g.main(self.monitor_id)
        return result_df, xls_lst


    def upload_to_gbq(self, result_df, xls_lst):
        """
        step2. upload to gbq
        Args:
            result_df: upload aggregated df to gbq
            xls_lst: remove all files
        Returns: nothing
        """

        print("upload_to_gbq work")
        ggg = gbq_function.Gbq()
        ggg.insert_into_gbq(result_df, destination_table="adj_adhoc", if_exist="append", project_id='slcc-250008')

        for xls in xls_lst:
            try:
                os.remove(xls)
            except OSError:
                print(f'Error while deleting file:{xls}')


    def select_from_gbq(self, sql, p):
        """
        step3. select from gbq(uploaded)
        Args:
            table_id: download df from gbq with given table id
        Returns: aggregated df for analysis
        """

        print("select_from_gbq work")
        ggg = gbq_function.Gbq()
        result_df = ggg.select_from_gbq(sql, p)

        #def select_from_gbq(self, sql, p):
        #    query_str = sql.substitute(p)
        #    result_df = self.client.query(query_str).to_dataframe()
        #    return result_df

        return result_df


    def analyze(self, df):
        """
        step 4. analyze
        Args:
            df: contains contents date url info
        Returns: adj from contents, contents, date, url, sentiment score
        """

        print("analyze work")
        df = df.iloc[:,1:4]
        df.columns = ["Date (KST)", "URL", "Contents"]

        adj_cls = adj.AdjectiveAnalysis()

        nlp = spacy.load("en_core_web_lg", disable_list=['ner', 'textcat', 'entity_ruler'])
        tokenizer = Tokenizer(nlp.vocab)
        document = df['Contents'].apply(lambda x: re.sub(r'[^a-zA-Z0-9+ ]', r' ', x)).values.tolist()

        result_lst = []

        for idx, doc in enumerate(tokenizer.pipe(document, n_threads=-1)):
            if idx % 10000 == 0:
                print(idx, "/", len(document))
            tmp_tok = [t.text for t in doc]

            for token in tmp_tok:
                if token.lower() in self.keywords:
                    result_lst.append(idx)

        result_set = set(result_lst)
        df = df.loc[df.index.isin(result_set), ['Contents', 'Date (KST)', 'URL']]

        output = pd.DataFrame()

        for keyword in self.keywords:

            analysis_result_lst_flag = adj_cls.adj_analysis(df, keyword, 'adj', "brand or product")
            # brand or product = temp

            if analysis_result_lst_flag:
                analysis_result_df_flag = adj_cls.post_processing(analysis_result_lst_flag)
                if not type(analysis_result_df_flag) == type(None):
                    analysis_result_df_flag = analysis_result_df_flag.reset_index(drop=True)
                    analysis_result_df_flag['product'] = pd.DataFrame([keyword] * len(analysis_result_df_flag))
                    output = pd.concat([output, analysis_result_df_flag], axis=0)

        output['Adjective'] = output['Adjective'].apply(lambda x: x[:-1] if x[-1:] == " " else x)
        join_df = pd.merge(output, senti_score, on="Adjective")

        return join_df


    def upload_result_to_gbq(self, df):
        """
        step5. upload to gbq
        Args:
            result_df: upload result_df to gbq
        Returns: nothing
        """

        print("upload_to_gbq work")
        ggg = gbq_function.Gbq()
        ggg.insert_into_gbq(df, destination_table="adj_adhoc", if_exist="append", project_id='slcc-250008') # if_exist="replace"


    def main(self):
        """
        전체 흐름을 관리하는 main function
        Returns: null
        """
        print("main work")

        result_df, xls_lst = self.crimson_data_dump()

        self.upload_to_gbq(result_df, xls_lst)

        #result_df = self.select_from_gbq(sql, p)

        result_df = self.analyze(result_df)

        self.upload_result_to_gbq(result_df)

        with pd.ExcelWriter('./data/test_output.xlsx') as writer:
            result_df.to_excel(writer, index=False, encoding="utf-8")

'''
    """ step7. save CA plot """
    def save_ca_plot(self, join_df):
        pass
    """ step8. send report """
    def send_report(self):
        pass
'''


if __name__ == "__main__":

    #pass

    monitor_id = sys.argv[1]
    from_date = sys.argv[2]
    to_date = sys.argv[3]
    #keywords = sys.argv[4] # multiple
    keywords = sys.argv[4].split(',')
    """
        input multiple keywords with comma (,)
    """

    print(f'from:{from_date},to:{to_date}')
    aa = AdjectiveAnalysis(monitor_id, from_date, to_date, keywords)
    aa.main()