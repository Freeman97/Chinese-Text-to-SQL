"""
dusql中部分数据存在列重复问题，需要删去重复列
"""

import json
from tqdm import tqdm
from src.utils.generate_query_toks import generate_query_toks

def dusql_dbschema_preprocess(db_schema_file):
    schema_list = []
    with open(db_schema_file, 'r') as f:
        schema_list = json.load(f)

    # duplicate_dbid_list = []
    
    for schema in tqdm(schema_list):
        column_name_dict = {}
        column_name_original_dict = {}
        new_column_name = []
        new_column_name_original = []
        new_types = []
        schema['column_names_original'] = schema['column_names']
        schema['table_names_original'] = schema['table_names']
        for i_, column_name in enumerate(schema['column_names']):
            column_name_serial = str(column_name)
            if column_name_serial not in column_name_dict:
                new_column_name.append(column_name)
                new_types.append(schema['column_types'][i_])
                column_name_dict[column_name_serial] = column_name
            else:
                print(f"duplicate name occured in db_id: {schema['db_id']}, column_name: {str(column_name)}")
        
        for column_name_original in schema['column_names_original']:
            column_name_original_serial = str(column_name_original)
            if column_name_original_serial not in column_name_original_dict:
                new_column_name_original.append(column_name_original)
                column_name_original_dict[column_name_original_serial] = column_name_original
            else:
                print(f"duplicate name occured in db_id: {schema['db_id']}, column_name_original: {str(column_name_original)}")
        schema['column_types'] = new_types
        schema['column_names'] = new_column_name
        schema['column_names_original'] = new_column_name_original

    with open(db_schema_file, 'w') as f:
        json.dump(schema_list, f, ensure_ascii=False)

if __name__ == '__main__':
    dusql_dbschema_preprocess('data/dusql/db_schema.json')
    generate_query_toks('data/dusql/train.json', use_back_quote=True)
    generate_query_toks('data/dusql/dev.json', use_back_quote=True)