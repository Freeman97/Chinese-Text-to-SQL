"""
直接删除DuSQL训练集、测试集中语法错误的SQL语句
"""

import json
import os

def remove_corrupted_case(dir_name, data_file_name, log_file):
    data_file = os.path.join(dir_name, data_file_name)
    data_list = []
    with open(data_file, 'r') as f:
        data_list = json.load(f)
    
    corrupted_sql = set()
    
    with open(log_file, 'r') as f:
        for line in f.readlines():
            corrupted_sql.add(line.replace('\n', ''))

    new_data_list = []
    for data in data_list:
        if data['query'] not in corrupted_sql:
            new_data_list.append(data)

    with open(os.path.join(dir_name, f'new_{data_file_name}'), 'w') as f:
        json.dump(new_data_list, f, ensure_ascii=False)


if __name__ == '__main__':
    remove_corrupted_case('data/dusql', 'train.json', 'error_log.log')
    remove_corrupted_case('data/dusql', 'dev.json', 'error_log.log')