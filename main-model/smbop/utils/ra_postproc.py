from functools import reduce
from itertools import *
from anytree import Node
import copy
from anytree.search import *
import re
import smbop.utils.node_util as node_util
import logging


else_dict = {
    "Selection": " WHERE ",
    "SELECTION": " WHERE ",
    "Groupby": " GROUP BY ",
    "GROUPBY": " GROUP BY ",
    "Limit": " LIMIT ",
    "LIMIT": " LIMIT ",
    "Having": " HAVING ",
    "HAVING": " HAVING ",
}

value_binary_op_dict = {
    'add': ' + ',
    'sub': ' - ',
    'mul': ' * ',
    'div': ' / ',
}

pred_dict = {
    "eq": " = ",
    "EQ": " = ",
    "like": " LIKE ",
    "LIKE": " LIKE ",
    "nlike": " NOT LIKE ",
    "NLIKE": " NOT LIKE ",
    "nin": " NOT IN ",
    "NIN": " NOT IN ",
    "lte": " <= ",
    "LTE": " <= ",
    "lt": " < ",
    "LT": " < ",
    "neq": " != ",
    "NEQ": " != ",
    "in": " IN ",
    "IN": " IN ",
    "gte": " >= ",
    "GTE": " >= ",
    "gt": " > ",
    "GT": " > ",
    "And": " AND ",
    "AND": " AND ",
    "Or": " OR ",
    "OR": " OR ",
    "except": " EXCEPT ",
    "EXCEPT": " EXCEPT ",
    "union": " UNION ",
    "UNION": " UNION ",
    "intersect": " INTERSECT ",
    "INTERCECT": " INTERSECT ",
    "Val_list": " , ",
    "VAL_LIST": " , ",
    "Product": " , ",
    "PRODUCT": " , "
}


def wrap_and(x):
    return [Node("And", children=x)] if len(x) > 1 else x


def fix_between(inp):
    inp = re.sub(r"([\s|\S]+) >= (\d*) AND \1 <= (\d*)", r"\1 BETWEEN \2 and \3", inp)
    # DuSQL中的LIKE条件值没有规范化为% %的形式
    # inp = re.sub(r"LIKE '([\s|\S]+?)'", r"LIKE '%\1%'", inp)
    return inp


class Item:
    def __init__(self, curr_type, l_child_idx, r_child_idx, mask):
        self.curr_type = curr_type
        self.l_child_idx = l_child_idx
        self.r_child_idx = r_child_idx
        self.mask = mask


class ZeroItem:
    def __init__(
        self,
        curr_type,
        final_leaf_indices,
        span_start_indices,
        span_end_indices,
        entities,
        enc_original,
        tokenizer,
        db_id,
    ):
        self.curr_type = curr_type
        self.final_leaf_indices = final_leaf_indices
        self.span_start_indices = span_start_indices
        self.span_end_indices = span_end_indices
        self.entities = entities
        self.enc_original = enc_original
        self.tokenizer = tokenizer
        self.db_id = db_id


def reconstruct_tree(
    op_names, binary_op_count, batch_el, idx, items, cnt, num_schema_leafs
):
    type_data = int(items[cnt].curr_type[batch_el][idx])
    tuple_el = Node(op_names[type_data])
    if cnt > 0:
        if type_data < binary_op_count:
            l_idx = items[cnt].l_child_idx[batch_el][idx]
            r_idx = items[cnt].r_child_idx[batch_el][idx]

            l_child = reconstruct_tree(
                op_names,
                binary_op_count,
                batch_el,
                l_idx,
                items,
                cnt - 1,
                num_schema_leafs,
            )
            r_child = reconstruct_tree(
                op_names,
                binary_op_count,
                batch_el,
                r_idx,
                items,
                cnt - 1,
                num_schema_leafs,
            )
            tuple_el.children = [l_child, r_child]
        else:
            idx = items[cnt].l_child_idx[batch_el][idx]
            child = reconstruct_tree(
                op_names,
                binary_op_count,
                batch_el,
                idx,
                items,
                cnt - 1,
                num_schema_leafs,
            )
            tuple_el.children = [child]
    else:
        num_schema_leafs = min(num_schema_leafs, items[cnt].final_leaf_indices[batch_el].shape[0])
        if idx < num_schema_leafs:
            entities = items[cnt].entities[batch_el]
            entity_idx = items[cnt].final_leaf_indices[batch_el][idx]
            tuple_el.val = entities[entity_idx]
            if '_table_' in entities:
                table_name = f'Table_{items[cnt].db_id[batch_el]}'
                if tuple_el.val == '_table_':
                    tuple_el.val = table_name
                elif entity_idx >= entities.index('*'):
                    tuple_el.val = f'{table_name}.{tuple_el.val}'
        else:
            span_idx = idx - num_schema_leafs
            enc_tokens = items[cnt].enc_original[batch_el][1:].tolist() # 把开头去掉了
            enc_tokens = [x.text for x in enc_tokens]
            start_id = items[cnt].span_start_indices[batch_el][span_idx]
            end_id = items[cnt].span_end_indices[batch_el][span_idx]
            # BERT系列的tokenizer在decode的时候会自动加空格
            if 'bert' in items[cnt].tokenizer.name_or_path:
                token_list = []
                for token in enc_tokens[start_id : end_id + 1]:
                    # 将token中的## 去掉
                    token = token.replace('##', '').replace('▁', '')
                    token_list.append(token)
                tuple_el.val = (
                    ''.join(token_list)
                )
            else:
                tuple_el.val = (
                    ''.join(enc_tokens[start_id : end_id + 1])
                )
    return tuple_el


def remove_keep(node: Node):
    if node.name == "keep":
        node = remove_keep(node.children[0])
    node.children = [remove_keep(child) for child in node.children]
    return node


def promote(node, root=False):
    children = node.children
    if node.name in ["Having"]:
        while True:
            if not node.is_root and node.parent.name not in [
                "union",
                "intersect",
                "Subquery",
                "except",
            ]:
                prev_parent = node.parent
                grandparent = (
                    prev_parent.parent if not prev_parent.is_root else prev_parent
                )
                node.parent = grandparent
            else:
                break
        node.siblings[0].parent = node
    for child in children:
        promote(child)


def flatten_cnf(in_node):
    if in_node.name in ["And", "Or", "Val_list", "Product"]:
        return flatten_cnf_recurse(in_node, in_node.name, is_root=True)
    else:
        children_list = []
        for child in in_node.children:
            child.parent = None
            child = flatten_cnf(child)
            children_list.append(child)
        in_node.children = children_list
        return in_node


def flatten_cnf_recurse(in_node, n_type, is_root=False):
    other_op = "And" if n_type == "Or" else "Or"
    if in_node.name == n_type:
        res = []
        for child in in_node.children:
            child.parent = None
            res += flatten_cnf_recurse(child, n_type)
        if is_root:
            in_node.children = res
            return in_node
        else:
            return res
    elif in_node.name == other_op:
        return [flatten_cnf_recurse(in_node, other_op, True)]
    else:
        if not is_root:
            children_list = []
            for child in in_node.children:
                child.parent = None
                child = flatten_cnf(child)
                children_list.append(child)
            in_node.children = children_list
        return [in_node]


def irra_to_sql(tree, peren=True, is_row_calc=False, alias=None):
    if len(tree.children) == 0:
        if tree.name == "Table" and isinstance(tree.val, dict):
            return tree.val["value"] + " AS " + tree.val["name"]
        if hasattr(tree, "val"):
            if is_row_calc and '.' in str(tree.val) and alias is not None:
                _, column_name = str(tree.val).split('.')
                column_name = alias + '.' + column_name
                return column_name
            else:
                return str(tree.val)
        else:
            print(tree)
            return ""
    if len(tree.children) == 1:
        if tree.name in [
            "min",
            "count",
            "max",
            "avg",
            "sum",
        ]:
            return "".join(
                [tree.name.upper(), " ( ", irra_to_sql(tree.children[0], is_row_calc=is_row_calc), " )"]
            )
        elif tree.name == "distinct":
            return "DISTINCT " + irra_to_sql(tree.children[0], is_row_calc=is_row_calc)
        elif tree.name == "literal":
            return """\'""" + str(irra_to_sql(tree.children[0], is_row_calc=is_row_calc)) + """\'"""
        elif tree.name == "Subquery":
            if peren:
                parsed = "".join(["(", irra_to_sql(tree.children[0], is_row_calc=is_row_calc), ")"])
                if is_row_calc and alias is not None:
                    parsed = parsed + ' ' + alias
                return parsed
            else:
                parsed = irra_to_sql(tree.children[0], is_row_calc=is_row_calc)
                if is_row_calc and alias is not None:
                    parsed = parsed + ' ' + alias
                return parsed
        elif tree.name == "Join_on":
            tree = tree.children[0]
            if tree.name == "eq":
                first_table_name = tree.children[0].val.split(".")[0]
                second_table_name = tree.children[1].val.split(".")[0]
                return f"{first_table_name} JOIN {second_table_name} ON {tree.children[0].val} = {tree.children[1].val}"
            else:
                if len(tree.children) > 0:
                    try:
                        t_Res = ", ".join([child.val for child in tree.children])
                    except Exception as e:
                        logging.exception(e)
                        raise e
                    return t_Res
                else:
                    return tree.val
        else:  # Predicate or Table or 'literal' or Agg
            return irra_to_sql(tree.children[0], is_row_calc=is_row_calc, alias=alias)
    else:
        if tree.name in [
            "eq",
            "like",
            "nin",
            "lte",
            "lt",
            "neq",
            "in",
            "gte",
            "gt",
            "And",
            "Or",
            "except",
            "union",
            "intersect",
            "Product",
            "Val_list",
        ]:
            pren_t = tree.name in [
                "eq",
                "like",
                "nin",
                "lte",
                "lt",
                "neq",
                "in",
                "gte",
                "gt",
            ]
            if tree.name == 'Product':
                if len(tree.children[0].children) == 1 and len(tree.children[1].children) == 1 and tree.children[0].children[0].name == 'Subquery' and tree.children[1].children[0].name == 'Subquery':
                    return (
                        pred_dict[tree.name].upper()
                        .join([
                            irra_to_sql(tree.children[0], pren_t, is_row_calc=is_row_calc, alias='a'),
                            irra_to_sql(tree.children[1], pren_t, is_row_calc=is_row_calc, alias='b'),
                        ])
                    )
            return (
                pred_dict[tree.name]
                .upper()
                .join([irra_to_sql(child, pren_t, is_row_calc=is_row_calc) for child in tree.children])
            )
        elif tree.name in value_binary_op_dict.keys():
            # 此时出现列 或 常量的四则运算
            if hasattr(tree.children[0], 'val') and hasattr(tree.children[1], 'val') and tree.children[0].val == tree.children[1].val:
                return (
                    irra_to_sql(tree.children[0], is_row_calc=is_row_calc, alias='a')
                    + value_binary_op_dict[tree.name]
                    + irra_to_sql(tree.children[1], is_row_calc=is_row_calc, alias='b')
                )
            else:
                return (
                    irra_to_sql(tree.children[0], is_row_calc=is_row_calc)
                    + value_binary_op_dict[tree.name]
                    + irra_to_sql(tree.children[1], is_row_calc=is_row_calc)
                )
        elif tree.name == "Orderby_desc":
            return (
                irra_to_sql(tree.children[1], is_row_calc=is_row_calc)
                + " ORDER BY "
                + irra_to_sql(tree.children[0], is_row_calc=is_row_calc)
                + " DESC"
            )
        elif tree.name == "Orderby_asc":
            return (
                irra_to_sql(tree.children[1], is_row_calc=is_row_calc)
                + " ORDER BY "
                + irra_to_sql(tree.children[0], is_row_calc=is_row_calc)
                + " ASC"
            )
        elif tree.name == "Project":
            return (
                "SELECT "
                + irra_to_sql(tree.children[0], is_row_calc=is_row_calc)
                + " FROM "
                + irra_to_sql(tree.children[1], is_row_calc=is_row_calc)
            )
        elif tree.name == "Join_on":
            # tree
            def table_name(x):
                return x.val.split(".")[0]

            table_tups = [
                (table_name(child.children[0]), table_name(child.children[1]))
                for child in tree.children
            ]
            res = table_tups[0][0]
            seen_tables = set(res)
            for (first, sec), child in zip(table_tups, tree.children):
                tab = first if sec in seen_tables else sec
                res += (
                    f" JOIN {tab} ON {child.children[0].val} = {child.children[1].val}"
                )
                seen_tables.add(tab)

            return res
        elif tree.name == "Selection":
            if len(tree.children) == 1:
                return irra_to_sql(tree.children[0], is_row_calc=is_row_calc)
            return (
                irra_to_sql(tree.children[1], is_row_calc=is_row_calc)
                + " WHERE "
                + irra_to_sql(tree.children[0], is_row_calc=is_row_calc)
            )
        else:  # 'Selection'/'Groupby'/'Limit'/Having'
            return (
                irra_to_sql(tree.children[1], is_row_calc=is_row_calc)
                + else_dict[tree.name]
                + irra_to_sql(tree.children[0], is_row_calc=is_row_calc)
            )


def ra_to_irra(tree):
    flat_tree = flatten_cnf(copy.deepcopy(tree))
    for node in findall(flat_tree, filter_=lambda x: x.name == "Selection"):
        table_node = node.children[1]
        join_list = []
        where_list = []
        having_list = []
        if node.children[0].name == "And":
            for predicate in node.children[0].children:
                # Join list的判断存在问题
                if (
                    all(node_util.is_field(child) for child in predicate.children)
                    and predicate.name == "eq"
                ):
                    join_list.append(predicate)
                else:
                    if predicate.name == "Or" or all(
                        child.name in ["literal", "Subquery", "Value", "Or", 'add', 'sub', 'div', 'mul', 'Val_list']
                        for child in predicate.children
                    ):
                        where_list.append(predicate)
                    else:
                        having_list.append(predicate)
                predicate.parent = None
        else:
            if node.children[0].name == "eq" and all(
                node_util.is_field(child) for child in node.children[0].children
            ):
                join_list = [node.children[0]]
            elif node.children[0].name == "Or":
                where_list = [node.children[0]]
            else:
                if all(
                    child.name in ["literal", "Subquery", "Value", "Or", 'add', 'sub', 'div', 'mul', 'Val_list']
                    for child in node.children[0].children
                ):
                    where_list = [node.children[0]]
                else:
                    having_list = [node.children[0]]
            node.children[0].parent = None
        having_node = (
            [Node("Having", children=wrap_and(having_list))] if having_list else []
        )
        join_on = Node("Join_on", children=join_list)
        if len(join_on.children) == 0:
            join_on.children = [table_node]
        node.children = having_node + wrap_and(where_list) + [join_on]
    flat_tree = Node("Subquery", children=[flat_tree])
    promote(flat_tree)
    return flat_tree.children[0]

def if_row_calc(tree):
    if tree.name == 'Product':
        l_child = tree.children[0]
        r_child = tree.children[1]
        if len(l_child.children) == 1 and len(r_child.children) == 1 and l_child.children[0].name == 'Subquery' and r_child.children[0].name == 'Subquery':
            return True
    
    if len(tree.children) == 0:
        return False
    return any([if_row_calc(x) for x in tree.children])

def ra_to_sql(tree):
    if tree:
        tree = remove_keep(tree)
        is_row_calc = if_row_calc(tree)
        irra = ra_to_irra(tree)
        sql = irra_to_sql(irra, is_row_calc=is_row_calc)
        # No Need in dusql
        # sql = fix_between(sql)
        sql = sql.replace("LIMIT value", "LIMIT 1")
        return sql
    else:
        return ""
