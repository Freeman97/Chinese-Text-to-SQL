# 补充C X C -> C的四则运算规则
# 补充P X P -> P的运算规则
import json

yield_c_ops = ["max", "min", "sum", "avg", "count", "Value", "Val_list", "literal", "distinct"]

yield_p_ops = ["And", "Or", "lte", "lt", "gte", "gt", "eq", "neq", "like", "nlike", "in", "nin"]

binary_ops = ["add", "sub", "mul", "div"]


# 二元运算符规则的标注方式: [op, [op1, op2]]
def _generate_rules_binary(op, op1, op2) -> list:
    rule_list = []
    if op1 in yield_c_ops:
        for binary_op in binary_ops:
            temp = [op, [binary_op, op2]]
            rule_list.append(temp)
    
    if op2 in yield_c_ops:
        for binary_op in binary_ops:
            temp = [op, [op1, binary_op]]
            rule_list.append(temp)
    return rule_list

# [op, [op1]]
def _generate_rules_unary(op, op1) -> list:
    rule_list = []
    if op1 in yield_c_ops:
        for binary_op in binary_ops:
            temp = [op, [binary_op]]
            rule_list.append(temp)
    return rule_list


def generate_binary_column_op_rules():
    with open('rule_values.json', 'r') as f:
        rules = json.load(f)
    
    addtional_rules = []
    for rule in rules:
        if len(rule[1]) == 2:
            temp = _generate_rules_binary(rule[0], rule[1][0], rule[1][1])
            if len(temp) > 0:
                addtional_rules.extend(temp)
        elif len(rule[1]) == 1:
            temp = _generate_rules_unary(rule[0], rule[1][0])
            if len(temp) > 0:
                addtional_rules.extend(temp)
    
    rules.extend(addtional_rules)
    with open('new_rule_values.json', 'w') as f:
        json.dump(rules, f)

def _generate_rules_predicate(op, op1, op2, rules, addtional_rules):
    rule_list = []
    if op1 in yield_p_ops:
        op1_idx = yield_p_ops.index(op1)
        for idx, yield_p_op in enumerate(yield_p_ops):
            if idx != op1_idx:
                temp = [op, [yield_p_op, op2]]
                if temp not in rules and temp not in addtional_rules and temp not in rule_list:
                    rule_list.append(temp)
    
    if op2 in yield_p_ops:
        op2_idx = yield_p_ops.index(op2)
        for idx, yield_p_op in enumerate(yield_p_ops):
            if idx != op2_idx:
                temp = [op, [op1, yield_p_op]]
                if temp not in rules and temp not in addtional_rules and temp not in rule_list:
                    rule_list.append(temp)
    return rule_list


def generate_predicate_rules():
    with open('new_rule_values.json', 'r') as f:
        rules = json.load(f)
    
    addtional_rules = []

    for rule in rules:
        if len(rule[1]) == 2:
            temp = _generate_rules_predicate(rule[0], rule[1][0], rule[1][1], rules, addtional_rules)
            if len(temp) > 0:
                addtional_rules.extend(temp)
    
    rules.extend(addtional_rules)
    with open('new_rule_values.json', 'w') as f:
        json.dump(rules, f)


if __name__ == "__main__":
    # generate_binary_column_op_rules()
    generate_predicate_rules()