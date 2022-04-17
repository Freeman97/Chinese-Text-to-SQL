import json
from smbop.utils import moz_sql_parser as msp
import smbop.utils.ra_preproc as ra_preproc
from anytree import Node
from smbop.utils.generate_query_toks import tokenize_dusql
from smbop.utils.tableqa_preprocess import fix_query
import logging
from tqdm import tqdm

import matplotlib as mpl
import numpy as np
mpl.use('Agg')

import matplotlib.pyplot as plt

def sanitize(query):
    query = query.replace('"', "'")
    if query.endswith(";"):
        query = query[:-1]
    for i in [1, 2, 3, 4, 5]:
        query = query.replace(f"t{i}", f"T{i}")
    for agg in ["count", "min", "max", "sum", "avg"]:
        query = query.replace(f"{agg} (", f"{agg}(")
    for agg in ["COUNT", "MIN", "MAX", "SUM", "AVG"]:
        query = query.replace(f"{agg} (", f"{agg}(")
    # 将双等号替换为等号
    query.replace("==", "=")
    return query

def tree_height(train_file, dev_file):
    train_data = []
    dev_data = []
    with open(train_file, 'r') as f:
        train_data = json.load(f)

    with open(dev_file, 'r') as f:
        dev_data = json.load(f)

    train_height = []
    dev_height = []

    for ex in tqdm(dev_data):
        sql = ''
        if 'nl2sql' in dev_file:
            sql = ex['query_for_parse']
        else:
            sql = ex['query']
        sql = sanitize(sql).strip()

        try:
            if 'dusql' in dev_file or 'debug' in dev_file:
                # sql = fix_date(sql)
                sql_tokens_with_backquote = tokenize_dusql(sql, use_back_quote=True)
                sql = ' '.join(sql_tokens_with_backquote)
            elif 'cspider' in dev_file:
                sql_tokens = tokenize_dusql(sql)
                sql = ' '.join(sql_tokens)
            tree_dict = msp.parse(sql)
            tree_dict_values = msp.parse(sql)
        except msp.ParseException as e:
            # DUSQL的带括号的集合操作会出现解析问题 -> 是msp有问题...
            # 尝试进行针对性处理
            set_operations = ['EXCEPT', 'except', 'UNION', 'union', 'INTERSECT', 'intersect']
            set_op_flag = False
            for set_op in set_operations:
                if set_op in sql:
                    set_op_flag = True
                    left_sql, right_sql = sql.split(set_op)
                    left_sql = left_sql.strip()
                    right_sql = right_sql.strip()
                    # 去掉多余括号
                    if left_sql.startswith('(') and left_sql.endswith(')'):
                        left_sql = left_sql[1:-1].strip()
                    if right_sql.startswith('(') and right_sql.endswith(')'):
                        right_sql = right_sql[1:-1].strip()
                    try:
                        # 手动完成集合操作的解析
                        left_dict = msp.parse(left_sql)
                        right_dict = msp.parse(right_sql)
                        tree_dict = {
                            'query': {
                                'op': {
                                    'type': set_op.lower(),
                                    'query1': left_dict['query'],
                                    'query2': right_dict['query']
                                }
                            }
                        }
                        tree_dict_values = tree_dict
                        break
                    except msp.ParseException as e:
                        logging.exception(e)
                        print(f"could'nt create AST for:  {sql}") # 带中文的SQL语句无法创建AST...-> 在中文列名周围用反引号进行包裹可破
                        with open('error_sql.log', 'a') as f:
                            f.write(f"could'nt create AST for:  {sql}\n")
                        continue
                    except Exception as e:
                        logging.exception(e)
                        with open('error_sql.log', 'a') as f:
                            f.write(f"could'nt create AST for:  {sql}\n")
                        continue
            if not set_op_flag:
                logging.exception(e) # 有4个数据出错了，LIMIT和数字没分开，不管
                print(f"could'nt create AST for:  {sql}") # 带中文的SQL语句无法创建AST...-> 在中文列名周围用反引号进行包裹可破
                with open('error_sql.log', 'a') as f:
                    f.write(f"could'nt create AST for:  {sql}\n")
                continue
        except Exception as e:
            logging.exception(e) # 有部分数据存在其它错
            print(f"could'nt create AST for:  {sql}")
            with open('error_sql.log', 'a') as f:
                f.write(f"could'nt create AST for:  {sql}\n")
            continue
        tree_obj = ra_preproc.ast_to_ra(tree_dict["query"])
        dev_height.append(tree_obj.height)

    print(f"Max height of dev set: {max(dev_height)}")
    plt.hist(dev_height, bins='auto')
    plt.savefig('nl2sql_dev_height_stat.png')    

    for ex in tqdm(train_data):
        sql = ''
        if 'nl2sql' in train_file:
            sql = ex['query_for_parse']
        else:
            sql = ex['query']
        sql = sanitize(sql).strip()

        try:
            if 'dusql' in train_file or 'debug' in train_file:
                # sql = fix_date(sql)
                sql_tokens_with_backquote = tokenize_dusql(sql, use_back_quote=True)
                sql = ' '.join(sql_tokens_with_backquote)
            elif 'cspider' in train_file:
                sql_tokens = tokenize_dusql(sql)
                sql = ' '.join(sql_tokens)
            tree_dict = msp.parse(sql)
            tree_dict_values = msp.parse(sql)
        except msp.ParseException as e:
            # DUSQL的带括号的集合操作会出现解析问题 -> 是msp有问题...
            # 尝试进行针对性处理
            set_operations = ['EXCEPT', 'except', 'UNION', 'union', 'INTERSECT', 'intersect']
            set_op_flag = False
            for set_op in set_operations:
                if set_op in sql:
                    set_op_flag = True
                    left_sql, right_sql = sql.split(set_op)
                    left_sql = left_sql.strip()
                    right_sql = right_sql.strip()
                    # 去掉多余括号
                    if left_sql.startswith('(') and left_sql.endswith(')'):
                        left_sql = left_sql[1:-1].strip()
                    if right_sql.startswith('(') and right_sql.endswith(')'):
                        right_sql = right_sql[1:-1].strip()
                    try:
                        # 手动完成集合操作的解析
                        left_dict = msp.parse(left_sql)
                        right_dict = msp.parse(right_sql)
                        tree_dict = {
                            'query': {
                                'op': {
                                    'type': set_op.lower(),
                                    'query1': left_dict['query'],
                                    'query2': right_dict['query']
                                }
                            }
                        }
                        tree_dict_values = tree_dict
                        break
                    except msp.ParseException as e:
                        logging.exception(e)
                        print(f"could'nt create AST for:  {sql}") # 带中文的SQL语句无法创建AST...-> 在中文列名周围用反引号进行包裹可破
                        with open('error_sql.log', 'a') as f:
                            f.write(f"could'nt create AST for:  {sql}\n")
                        continue
                    except Exception as e:
                        logging.exception(e)
                        with open('error_sql.log', 'a') as f:
                            f.write(f"could'nt create AST for:  {sql}\n")
                        continue
            if not set_op_flag:
                logging.exception(e) # 有4个数据出错了，LIMIT和数字没分开，不管
                print(f"could'nt create AST for:  {sql}") # 带中文的SQL语句无法创建AST...-> 在中文列名周围用反引号进行包裹可破
                with open('error_sql.log', 'a') as f:
                    f.write(f"could'nt create AST for:  {sql}\n")
                continue
        except Exception as e:
            logging.exception(e) # 有部分数据存在其它错
            print(f"could'nt create AST for:  {sql}")
            with open('error_sql.log', 'a') as f:
                f.write(f"could'nt create AST for:  {sql}\n")
            continue
        tree_obj = ra_preproc.ast_to_ra(tree_dict["query"])
        train_height.append(tree_obj.height)

    print(f"Max height of train set: {max(train_height)}")
    plt.hist(train_height, bins='auto')
    plt.savefig('nl2sql_train_height_stat.png')

if __name__ == '__main__':
    # fix_query('nl2sql/', 'train.json')
    # fix_query('nl2sql/', 'dev.json')
    tree_height('dusql/dev.json', 'dusql/train.json')