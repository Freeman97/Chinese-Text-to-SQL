import json

def dusql_fix(dataset_path):
    with open(dataset_path + 'db_schema.json', 'r') as f:
        schema_list = json.load(f)
        for schema in schema_list:
            if 'table_names_original' not in schema:
                schema['table_names_original'] = schema['table_names']
            if 'column_names_original' not in schema:
                schema['column_names_original'] = schema['column_names']
    
    with open(dataset_path + 'db_schema.json', 'w') as f:
        json.dump(schema_list, f, ensure_ascii=False)