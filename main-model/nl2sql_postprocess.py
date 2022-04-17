import json

def nl2sql_postprocess(pred_file, schema_file):
    # 先处理schema
    schema_dict = {}
    with open(schema_file, 'r') as f:
        schema_list = json.load(f)
        for schema in schema_list:
            temp = {}
            table_name = f"Table_{schema['db_id']}"
            temp['table'] = table_name
            temp['column'] = schema['tables'][table_name]['header']
            schema_dict[schema['db_id']] = temp
    
    with open(pred_file, 'r') as f:
        with open('new_' + pred_file, 'w') as ff:
            for line in f.readlines():
                line = line.replace('\n', '')
                qid, sql, db_id = line.split('\t')
                if 'SELECT' not in sql and 'select' not in sql:
                    table_name = schema_dict[db_id]['table']
                    default_column = schema_dict[db_id]['column'][0]
                    new_sql = f"SELECT {default_column} FROM {table_name} WHERE {default_column} = '1'"
                    line = '\t'.join([qid, new_sql, db_id])
                ff.write(line + '\n')

if __name__ == '__main__':
    nl2sql_postprocess('preds_test_nl2sql.sql', 'nl2sql/db_content.json')
