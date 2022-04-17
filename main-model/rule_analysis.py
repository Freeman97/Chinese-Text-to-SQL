import json
from smbop.utils import moz_sql_parser as msp
from smbop.utils import ra_preproc
from smbop.utils import generate_query_toks
from smbop.utils import node_util
import logging

# 分析train和dev中的AST, 以寻找缺失的结构，顺便看看树高分布

old_rules = []

missing_rules = []

depth_dict = {}

def get_ra(query: str):
    try:
        tree_dict = msp.parse(query)
    except Exception as e:
        set_operations = ['EXCEPT', 'except', 'UNION', 'union', 'INTERSECT', 'intersect']
        set_op_flag = False
        for set_op in set_operations:
            if set_op in query:
                set_op_flag = True
                left_sql, right_sql = query.split(set_op)
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
                    break
                except msp.ParseException as e:
                    logging.exception(e)
                    return None
            else:
                return None
    ra = ra_preproc.ast_to_ra(tree_dict['query'])
    return ra

def traverse_tree(ra_tree):
    if not hasattr(ra_tree, 'children') or len(ra_tree.children) == 0:
        return 1
    elif len(ra_tree.children) == 1:
        cur_rule = f'["{ra_tree.name}", ["{ra_tree.children[0].name}"]]'
        if cur_rule not in old_rules and cur_rule not in missing_rules:
            missing_rules.append(json.loads(cur_rule))
        return traverse_tree(ra_tree.children[0]) + 1
    elif len(ra_tree.children) == 2:
        cur_rule = f'["{ra_tree.name}", ["{ra_tree.children[0].name}", "{ra_tree.children[1].name}"]]'
        if cur_rule not in old_rules and cur_rule not in missing_rules:
            missing_rules.append(json.loads(cur_rule))
        left = traverse_tree(ra_tree.children[0])
        right = traverse_tree(ra_tree.children[1])
        return max(left, right) + 1

def rule_analysis():
    global old_rules
    old_rules = node_util.RULES_values
    with open('dusql/train.json', 'r') as f:
        train_data = json.load(f)
    
    with open('dusql/dev.json', 'r') as f:
        dev_data = json.load(f)
    
    for train in train_data:
        query = train['query']
        toks = generate_query_toks.tokenize_dusql(query, use_back_quote=True)
        query = ' '.join(toks)
        ra = get_ra(query)
        if ra is not None:
            depth = traverse_tree(ra)
            if depth in depth_dict:
                count = depth_dict[depth]
                depth_dict[depth] = count + 1
            else:
                depth_dict[depth] = 1
    
    for dev in dev_data:
        query = train['query']
        toks = generate_query_toks.tokenize_dusql(query, use_back_quote=True)
        query = ' '.join(toks)
        ra = get_ra(query)
        if ra is not None:
            depth = traverse_tree(ra)
            if depth in depth_dict:
                count = depth_dict[depth]
                depth_dict[depth] = count + 1
            else:
                depth_dict[depth] = 1
            
    with open('missing_rules.json', 'w') as f:
        json.dump(missing_rules, f)

    with open('depth_statistics.json', 'w') as f:
        json.dump(depth_dict, f)

if __name__ == '__main__':
    rule_analysis()