"""
预处理NL2SQL(TableQA)数据集, 构造查询语句, 使用单引号而非双引号包裹条件值
"""

import json
from tqdm import tqdm

op_sql_dict = {0:">", 1:"<", 2:"==", 3:"!="}
agg_sql_dict = {0:"", 1:"AVG", 2:"MAX", 3:"MIN", 4:"COUNT", 5:"SUM"}
conn_sql_dict = {0:"", 1:"and", 2:"or"}

def construct_query_tableqa(file_name: str, schema_file_name: str):
    with open(file_name, 'r', encoding='utf8') as f:
        data_list = json.load(f)
    
    with open(schema_file_name, 'r', encoding='utf8') as f:
        db_schema = json.load(f)

    # 解析出schema字典
    schema_dict = {}
    for db in db_schema:
        schema_dict[db['db_id']] = db

    for data in tqdm(data_list):
        current_schema = schema_dict[data['db_id']]
        current_column = [x[1] for x in current_schema['column_names_original']]

        select_list = []
        select_list_with_backquote = []
        conn = conn_sql_dict[data['sql']['cond_conn_op']]
        cond_list = []
        cond_list_with_backquote = []
        double_quote_cond_list = []

        # process select
        for index, column in enumerate(data['sql']['sel']):
            column_name = current_column[column]
            if data['sql']['agg'][index] != 0:
                agg = agg_sql_dict[data['sql']['agg'][index]]
                column_name_with_backquote = f"{agg} ( `{column_name}` )"
                column_name = f"{agg} ( {column_name} )"
            else:
                column_name_with_backquote = f"`{column_name}`"
                column_name = f"{column_name}"
            select_list.append(column_name)
            select_list_with_backquote.append(column_name_with_backquote)
        
        select_clause = ' , '.join(select_list)
        select_clause_with_backquote = ' , '.join(select_list_with_backquote)

        # process where
        for column, op, value in data['sql']['conds']:
            column_name = current_column[column]
            op_name = op_sql_dict[op]
            # manually fix error
            # if '\t ' in value:
            #     value = value.replace('\t ', '')
            cond_str = f"{column_name} {op_name} '{value}'"
            cond_str_double_quote = f"{column_name} {op_name} \"{value}\""
            cond_str_with_backquote = f"`{column_name}` {op_name} '{value}'"
            cond_list.append(cond_str)
            double_quote_cond_list.append(cond_str_double_quote)
            cond_list_with_backquote.append(cond_str_with_backquote)
        
        where_clause = f" {conn} ".join(cond_list)
        double_quote_where_clause = f" {conn} ".join(double_quote_cond_list)
        where_clause_with_backquote = f" {conn} ".join(cond_list_with_backquote)

        sql_str = f"SELECT {select_clause} WHERE {where_clause}"
        double_quote_sql_str = f"SELECT {select_clause} WHERE {double_quote_where_clause}"
        query_with_backquote = f"SELECT {select_clause_with_backquote} WHERE {where_clause_with_backquote}"

        # assert double_quote_sql_str == data['query']

        data['query'] = sql_str
        data['query_for_parse'] = query_with_backquote
    
    with open(file_name, 'w') as f:
        json.dump(data_list, f, ensure_ascii=False)
    


if __name__ == '__main__':
    # construct_query_tableqa('nl2sql-dbg/debug.json', 'nl2sql/db_schema.json')
    construct_query_tableqa('nl2sql/train.json', 'nl2sql/db_schema.json')
    construct_query_tableqa('nl2sql/dev.json', 'nl2sql/db_schema.json')
        
        
