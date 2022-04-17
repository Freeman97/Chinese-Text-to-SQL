import json
from smbop.utils import moz_sql_parser as msp
from smbop.utils import ra_preproc
from smbop.utils import generate_query_toks
from anytree import search
import logging

def analysis():
    with open('dev_err_log.json', 'r') as f:
        data = json.load(f)
    
    leaf_err = []
    row_calculation = []
    col_calculation = []
    others = []
    for unit in data:
        query = unit['query']
        toks = generate_query_toks.tokenize_dusql(query, use_back_quote=True)
        new_query = ' '.join(toks)
        try:
            parsed = msp.parse(new_query)
        except Exception as e:
            set_operations = ['EXCEPT', 'except', 'UNION', 'union', 'INTERSECT', 'intersect']
            set_op_flag = False
            for set_op in set_operations:
                if set_op in new_query:
                    set_op_flag = True
                    left_sql, right_sql = new_query.split(set_op)
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
                        continue
        
        
        tree_obj = ra_preproc.ast_to_ra(parsed['query'])
        leaves = tree_obj.leaves

        math_ops = search.findall(tree_obj, filter_=lambda x: x.name in ['add', 'sub', 'mul', 'div'])

        if len(unit['leaf_list']) < len(leaves):
            unit['type'] = 'leaf'
            leaf_err.append(unit)
        if len(math_ops) > 0:
            for math_op in math_ops:
                if hasattr(math_op.children[0], 'val') and hasattr(math_op.children[1], 'val') and math_op.children[0].val == math_op.children[1].val:
                    unit['type'] = 'row_calculation'
                    row_calculation.append(unit)
                    break
                else:
                    unit['type'] = 'col_calculation'
                    col_calculation.append(unit)
                    break
        else:
            unit['type'] = 'others'
            others.append(unit)

    with open('leaf_err.json', 'w') as f:
        json.dump(leaf_err, f, ensure_ascii=False)
        
    with open('row_calculation.json', 'w') as f:
        json.dump(row_calculation, f, ensure_ascii=False)

    with open('col_calculation.json', 'w') as f:
        json.dump(col_calculation, f, ensure_ascii=False)
    
    with open('others.json', 'w') as f:
        json.dump(others, f, ensure_ascii=False)

if __name__ == '__main__':
    analysis()
    