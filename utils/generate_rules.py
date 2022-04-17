import json

r_r_r = ["union", "intersect", "except", "Product"]

r_r = ["Subquery"]

p_r_r = ["Selection"]

cset_r_r = ["Project", "Groupby"]

c_r_r = ["Orderby_desc", "Orderby_asc", "Limit"]

c_r_r.extend(cset_r_r)

p_p_p = ["And", "Or"]

c_c_p = ["lt", "gt", "lte", "gte", "eq", "neq", "like", "nlike"]

c_r_p = ["in", "nin"]

# 特殊运算符: 考虑到 子查询的特殊情况
special_c_r_p = ["lt", "gt", "lte", "gte", "eq", "neq"]

c_r_p.extend(special_c_r_p)

c_c = ["sum", "max", "min", "count", "avg", "distinct"]

c_c_c = ["add", "sub", "mul", "div"]

cset_cset_cset = ["Val_list"]

c_c_cset = cset_cset_cset

cset_c_cset = cset_cset_cset

basic_type_c = ["Value", "literal", "Table"]

basic_type_r = ["Table"]

type_convert = [
    ["literal", ["Value"]],
    ["Table", ["Subquery"]]
]


yield_r = []
yield_r.extend(basic_type_r)
yield_r.extend(r_r_r)
yield_r.extend(r_r)
yield_r.extend(p_r_r)
yield_r.extend(c_r_r)

yield_p = []
yield_p.extend(p_p_p)
yield_p.extend(c_c_p)
yield_p.extend(c_r_p)

yield_c = []
yield_c.extend(c_c_c)
yield_c.extend(c_c)
yield_c.extend(basic_type_c)

yield_cset = []
yield_cset.extend(cset_cset_cset)
yield_cset.extend(yield_c)



def generate_rules():
    rules = []
    # r x r -> r
    for op in r_r_r:
        for op1 in yield_r:
            for op2 in yield_r:
                rules.append([op, [op1, op2]])
                rules.append([op, [op2, op1]])
    
    # r -> r
    for op in r_r:
        for op1 in yield_r:
            rules.append([op, [op1]])
    
    # p x r -> r
    for op in p_r_r:
        for op1 in yield_p:
            for op2 in yield_r:
                rules.append([op, [op1, op2]])
                rules.append([op, [op2, op1]])
    
    # cset x r -> r
    for op in cset_r_r:
        for op1 in yield_cset:
            for op2 in yield_r:
                rules.append([op, [op1, op2]])
                rules.append([op, [op2, op1]])
    
    # c x r -> r
    for op in c_r_r:
        for op1 in yield_c:
            for op2 in yield_r:
                rules.append([op, [op1, op2]])
                rules.append([op, [op2, op1]])
    
    # p x p -> p
    for op in p_p_p:
        for op1 in yield_p:
            for op2 in yield_p:
                rules.append([op, [op1, op2]])
                rules.append([op, [op2, op1]])

    # c x c -> p
    for op in c_c_p:
        for op1 in yield_c:
            for op2 in yield_c:
                rules.append([op, [op1, op2]])
                rules.append([op, [op2, op1]])

    # c x r -> p
    for op in c_r_p:
        for op1 in yield_c:
            for op2 in yield_r:
                rules.append([op, [op1, op2]])
                rules.append([op, [op2, op1]])
    
    # c -> c
    for op in c_c:
        for op1 in yield_c:
            # 排除一下literal
            if op1 != 'literal':
                rules.append([op, [op1]])
    
    # c x c -> c
    for op in c_c_c:
        for op1 in yield_c:
            for op2 in yield_c:
                rules.append([op, [op1, op2]])
                rules.append([op, [op2, op1]])

    # cset x cset -> cset
    for op in cset_cset_cset:
        for op1 in yield_cset:
            for op2 in yield_cset:
                rules.append([op, [op1, op2]])
                rules.append([op, [op2, op1]])

    rules.extend(type_convert)

    # TODO:不放心的话求个并集?

    with open('new_rules.json', 'w', encoding='utf8') as f:
        json.dump(rules, f)

if __name__ == '__main__':
    generate_rules()