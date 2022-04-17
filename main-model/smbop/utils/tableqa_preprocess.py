"""
1. 根据TableQA数据集的query字段还原出目标SQL语句 -> 需要添加FROM
2. 根据TableQA数据集的tables.json还原出db_schema.json和db_content.json
"""
"""
    处理NL2SQL的方法:
    1. 强行指定一个节点__Table__, 在选择DB_CONSTANT时无论是训练还是推理都将其分数置为最大
    2. 修改规则，与R相关的计算全部去掉，新增特殊的投影运算: C X P -> R
        2较难实现，因为需要实现一套特殊语法
"""
import json

def fix_query(data_path, file_name, new_file_name=None):
    if new_file_name is None:
        old_name, ext = file_name.split('.')
        new_file_name = f"{old_name}_new.{ext}"

    with open(data_path + file_name, 'r') as f:
        data_list = json.load(f)

    for data in data_list:
        query = data['query']
        query_for_parse = data['query_for_parse']
        # 人工添加统一的FROM子句，构成完整的SQL
        left, right = query.split('WHERE')
        left_for_parse, right_for_parse = query_for_parse.split('WHERE')
        new_query = f"{left}FROM _TABLE_ WHERE{right}"
        new_query_for_parse = f"{left_for_parse}FROM _TABLE_ WHERE{right_for_parse}"
        data['query'] = new_query
        data['query_for_parse'] = new_query_for_parse
    
    with open(data_path + new_file_name, 'w') as f:
        json.dump(data_list, f, ensure_ascii=False)

if __name__ == '__main__':
    fix_query('nl2sql/', 'train.json')
    fix_query('nl2sql/', 'dev.json')
