
# 输入总db_schema和选表结果, 分割出单独的schema
def schema_split(db_schema, candidate_table, db_id):
    splitted_schema = {}
    splitted_schema['db_id'] = db_id
    # 训练阶段candidate_table中会有一样的id，需要进行去重
    candidate_table = list(set(candidate_table))
    table_names_original = []
    table_names = []

    table_id_map = {}

    for i, table_id in enumerate(candidate_table):
        table_names.append(db_schema['table_names'][table_id])
        table_names_original.append(db_schema['table_names_original'][table_id])
        table_id_map[db_schema['table_names'][table_id]] = i
    
    splitted_schema['table_names'] = table_names
    splitted_schema['table_names_original'] = table_names_original

    column_names_original = [[-1, '*']]
    column_names = [[-1, '*']]
    # 硬塞一个type进行对齐
    column_types = ['number']
    column_counter = 0
    primary_keys = []
    column_id_map = {}
    column_id_map_reverse = {}

    # 为了处理方便, 对原数据进行一定的转换
    primary_keys_set = set(db_schema['primary_keys'])
    forgien_keys_map = {}
    for key_pair in db_schema['foreign_keys']:
        forgien_keys_map[key_pair[0]] = key_pair[1]

    # 处理列
    for i, column in enumerate(db_schema['column_names']):
        current_column_names_original = []
        current_column_names = []
        if column[0] in candidate_table:
            column_counter += 1
            current_table_name = db_schema['table_names'][column[0]]
            current_table_name_original = db_schema['table_names_original'][column[0]]
            current_table_id = table_id_map[current_table_name]
            current_column_names.append(current_table_id)
            current_column_names.append(column[1])
            current_column_names_original.append(current_table_id)
            current_column_names_original.append(db_schema['column_names_original'][i][1])
            column_names_original.append(current_column_names_original)
            column_names.append(current_column_names)
            # 处理类型
            column_types.append(db_schema['column_types'][i - 1])

            # 处理主键
            if i in primary_keys_set:
                primary_keys.append(column_counter)

            # 构建列ID映射
            # 原本的列ID -> 新的列ID
            column_id_map[i] = column_counter
            # 新的列ID -> 原本的列ID
            column_id_map_reverse[column_counter] = i
    
    splitted_schema['column_names_original'] = column_names_original
    splitted_schema['column_names'] = column_names
    splitted_schema['column_types'] = column_types

    splitted_schema['primary_keys'] = primary_keys

    # 处理外键
    foreign_keys = []
    for i, column in enumerate(column_names):
        if i == 0:
            continue
        # 原本的列ID
        original_id_key = column_id_map_reverse[i]
        if original_id_key in forgien_keys_map:
            current_pair = [i]
            original_id_val = forgien_keys_map[original_id_key]
            if original_id_val in column_id_map:
                current_pair.append(column_id_map[ original_id_val ])
                foreign_keys.append(current_pair)

    splitted_schema['foreign_keys'] = foreign_keys
    
    return splitted_schema
