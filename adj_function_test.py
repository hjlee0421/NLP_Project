# terminal >> pip install -r requirements.txt
# terminal >> python -m spacy download en_core_web_lg
# UPDATE VERSIONS
# terminal >> pip freeze > requirements.txt

import os
import pandas as pd
from Spacy_ADJ import adj_function_all as adj_all, adj_function_v3 as adj3

if __name__ == '__main__':

    senti_score = pd.read_csv('./data/SentiWords_2.0.rawdata')

    path = './data/bulk_export/'
    files = os.listdir(path)

    DATA = pd.DataFrame()

    for f in files:
        file = pd.read_excel(path + f)
        file = file[["Unnamed: 1", "Unnamed: 2", "Unnamed: 3"]]
        file.columns = file.iloc[0]
        file = file.iloc[1:]
        file = file.reset_index(drop=True)
        file.Contents = file.Contents.apply(lambda x: str(x))
        DATA = pd.concat([DATA, pd.DataFrame(file)], axis=0)

    DATA = DATA.reset_index(drop=True)
    print(DATA.shape)

    ADJ_CLS_ALL = adj_all.Adjective_Analysis()
    result_list_all = ADJ_CLS_ALL.ADJ_Analysis(DATA, "Samsung", 'ADJ', 'brand')
    result_df_all = ADJ_CLS_ALL.post_processing(result_list_all)
    result_df_all = result_df_all.reset_index(drop=True)
    result_df_all['Adjective'] = result_df_all['Adjective'].apply(lambda x: x[:-1] if x[-1:] == " " else x)
    join_df = pd.merge(result_df_all, senti_score, on="Adjective")

    with pd.ExcelWriter('./data/GoPro_output_all_adj.xlsx') as writer:
        join_df.to_excel(writer, index=False, encoding="utf-8")


    #ADJ_CLS_V2 = adj2.Adjective_Analysis()
    ADJ_CLS_V3 = adj3.Adjective_Analysis()
    #ADJ_CLS_V4 = adj4.Adjective_Analysis()

    #result_list_v2 = ADJ_CLS_V2.ADJ_Analysis(DATA, "Samsung", 'ADJ', 'brand')
    #result_df_v2 = ADJ_CLS_V2.post_processing(result_list_v2)

    "Hypersmooth"
    "Stabilization"
    "GoPro"

    RESULT = pd.DataFrame()

    #result_list_v3 = ADJ_CLS_V3.ADJ_Analysis(DATA, "Samsung", 'ADJ', 'GoPro')
    result_list_v3 = ADJ_CLS_V3.ADJ_Analysis(DATA, "Hypersmooth", 'ADJ', 'GoPro')
    result_df_v3 = ADJ_CLS_V3.post_processing(result_list_v3)

    RESULT = pd.concat([RESULT, result_df_v3], axis = 0)

    result_list_v3 = ADJ_CLS_V3.ADJ_Analysis(DATA, "Stabilization", 'ADJ', 'GoPro')
    result_df_v3 = ADJ_CLS_V3.post_processing(result_list_v3)

    RESULT = pd.concat([RESULT, result_df_v3], axis = 0)

    result_list_v3 = ADJ_CLS_V3.ADJ_Analysis(DATA, "GoPro", 'ADJ', 'GoPro')
    result_df_v3 = ADJ_CLS_V3.post_processing(result_list_v3)

    RESULT = pd.concat([RESULT, result_df_v3], axis = 0)





    #result_list_v4 = ADJ_CLS_V4.ADJ_Analysis(DATA, "Samsung", 'ADJ')
    #result_df_v4 = ADJ_CLS_V4.post_processing(result_list_v4)

    #print(result_df_v2.shape)
    print(result_df_v3.shape)
    #print(result_df_v4.shape)


    output = pd.DataFrame()

    for keyword in ["Hypersmooth", "Stabilization", "GoPro"]:

        analysis_result_lst_flag = ADJ_CLS_V3.ADJ_Analysis(DATA, keyword, 'adj', "brand or product")
        # brand or product = temp

        if analysis_result_lst_flag:
            analysis_result_df_flag = ADJ_CLS_V3.post_processing(analysis_result_lst_flag)
            if not type(analysis_result_df_flag) == type(None):
                analysis_result_df_flag = analysis_result_df_flag.reset_index(drop=True)
                analysis_result_df_flag['product'] = pd.DataFrame([keyword] * len(analysis_result_df_flag))
                output = pd.concat([output, analysis_result_df_flag], axis=0)

    output['Adjective'] = output['Adjective'].apply(lambda x: x[:-1] if x[-1:] == " " else x)
    join_df = pd.merge(output, senti_score, on="Adjective")

    with pd.ExcelWriter('./data/GoPro_output.xlsx') as writer:
        join_df.to_excel(writer, index=False, encoding="utf-8")