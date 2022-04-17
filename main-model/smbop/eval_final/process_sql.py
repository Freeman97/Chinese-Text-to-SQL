################################
# Assumptions:
#   1. sql is correct
#   2. only table name has alias
#   3. only one intersect/union/except
#
# val: number(float)/string(str)/sql(dict)
# col_unit: (agg_id, col_id, isDistinct(bool))
# val_unit: (unit_op, col_unit1, col_unit2)
# table_unit: (table_type, col_unit/sql)
# cond_unit: (not_op, op_id, val_unit, val1, val2)
# condition: [cond_unit1, 'and'/'or', cond_unit2, ...]
# sql {
#   'select': (isDistinct(bool), [(agg_id, val_unit), (agg_id, val_unit), ...])
#   'from': {'table_units': [table_unit1, table_unit2, ...], 'conds': condition}
#   'where': condition
#   'groupBy': [col_unit1, col_unit2, ...]
#   'orderBy': ('asc'/'desc', [val_unit1, val_unit2, ...])
#   'having': condition
#   'limit': None/limit value
#   'intersect': None/sql
#   'except': None/sql
#   'union': None/sql
# }
################################

import json
import sqlite3
from nltk import word_tokenize
from smbop.utils.generate_query_toks import tokenize_query, tokenize_dusql
import logging

CLAUSE_KEYWORDS = (
    "select",
    "from",
    "where",
    "group",
    "order",
    "limit",
    "intersect",
    "union",
    "except",
)

CLAUSE_KEYWORDS_CAPITAL = (
    "SELECT",
    "FROM",
    "WHERE",
    "GROUP",
    "ORDER",
    "LIMIT",
    "INTERSECT",
    "UNION",
    "EXCEPT",
)

JOIN_KEYWORDS = ("join", "on", "as")

JOIN_KEYWORDS_CAPITAL = ("JOIN", "ON", "AS")

WHERE_OPS = (
    "not",
    "between",
    "=",
    ">",
    "<",
    ">=",
    "<=",
    "!=",
    "in",
    "like",
    "is",
    "exists",
)

WHERE_OPS_CAPITAL = (
    "NOT",
    "BETWEEN",
    "=",
    ">",
    "<",
    ">=",
    "<=",
    "!=",
    "IN",
    "LIKE",
    "IS",
    "EXISTS",
)

UNIT_OPS = ("none", "-", "+", "*", "/")

UNIT_OPS_CAPITAL = ("NONE", "-", "+", "*", "/")

AGG_OPS = ("none", "max", "min", "count", "sum", "avg")

AGG_OPS_CAPITAL = ("NONE", "MAX", "MIN", "COUNT", "SUM", "AVG")

TABLE_TYPE = {
    "sql": "sql",
    "table_unit": "table_unit",
}

COND_OPS = ("and", "or")
COND_OPS_CAPITAL = ("AND", "OR")
SQL_OPS = ("intersect", "union", "except")
SQL_OPS_CAPITAL = ("INTERSECT", "UNION", "EXCEPT")
ORDER_OPS = ("desc", "asc")
ORDER_OPS_CAPITAL = ("DESC", "ASC")
mapped_entities = []

class Schema:
    """
    Simple schema which maps table&column to a unique identifier
    """

    def __init__(self, schema):
        self._schema = schema
        self._idMap = self._map(self._schema)

    @property
    def schema(self):
        return self._schema

    @property
    def idMap(self):
        return self._idMap

    def _map(self, schema):
        idMap = {"*": "__all__"}
        id = 1
        for key, vals in schema.items():
            for val in vals:
                idMap[key.lower() + "." + val.lower()] = (
                    "__" + key.lower() + "." + val.lower() + "__"
                )
                # idMap[key.upper() + "." + val.upper()] = (
                #     "__" + key.upper() + "." + val.upper() + "__"
                # )
                id += 1

        for key in schema:
            # idMap[key.upper()] = "__" + key.upper() + "__"
            idMap[key.lower()] = "__" + key.lower() + "__"
            id += 1

        return idMap


def get_schema(db):
    """
    Get database's schema, which is a dict with table name as key
    and list of column names as value
    :param db: database path
    :return: schema dict
    """

    schema = {}
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    # fetch table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [str(table[0].lower()) for table in cursor.fetchall()]

    # fetch table info
    for table in tables:
        cursor.execute("PRAGMA table_info({})".format(table))
        schema[table] = [str(col[1].lower()) for col in cursor.fetchall()]

    return schema


def get_schema_from_json(fpath, db_id):
    # 传入db_content.json的路径
    with open(fpath) as f:
        data = json.load(f)

    table_data = None
    for current_db in data:
        if current_db['db_id'] == db_id:
            table_data = current_db['tables']
    
    if table_data is None:
        table_data = data[0]['tables']

    schema = {}
    for table_name in table_data.keys():
        cols = table_data[table_name]['header']
        schema[table_name] = cols

    return schema


def tokenize(string):
    string = str(string)
    string = string.replace(
        "'", '"'
    )  # ensures all string values wrapped by "" problem??
    quote_idxs = [idx for idx, char in enumerate(string) if char == '"']
    assert len(quote_idxs) % 2 == 0, "Unexpected quote"

    # keep string value as token
    vals = {}
    for i in range(len(quote_idxs) - 1, -1, -2):
        qidx1 = quote_idxs[i - 1]
        qidx2 = quote_idxs[i]
        val = string[qidx1 : qidx2 + 1]
        key = "__val_{}_{}__".format(qidx1, qidx2)
        string = string[:qidx1] + key + string[qidx2 + 1 :]
        vals[key] = val

    toks = [word.lower() for word in word_tokenize(string)]
    # replace with string value token
    for i in range(len(toks)):
        if toks[i] in vals:
            toks[i] = vals[toks[i]]

    # find if there exists !=, >=, <=
    eq_idxs = [idx for idx, tok in enumerate(toks) if tok == "="]
    eq_idxs.reverse()
    prefix = ("!", ">", "<")
    for eq_idx in eq_idxs:
        pre_tok = toks[eq_idx - 1]
        if pre_tok in prefix:
            toks = toks[: eq_idx - 1] + [pre_tok + "="] + toks[eq_idx + 1 :]

    return toks

# 因为大小写的原因没有匹配成功，这个程序写得也太脆弱了...
def scan_alias(toks):
    """Scan the index of 'as' and build the map for all alias"""
    as_idxs = [idx for idx, tok in enumerate(toks) if tok.lower() == "as"]
    alias = {}
    for idx in as_idxs:
        alias[toks[idx + 1]] = toks[idx - 1]
    return alias


def get_tables_with_alias(schema, toks):
    tables = scan_alias(toks)
    for key in schema:
        assert key not in tables, "Alias {} has the same name in table".format(key)
        tables[key] = key
    return tables


def parse_col(toks, start_idx, tables_with_alias, schema, default_tables=None, use_capital=False):
    """
    :returns next idx, column id
    """
    global mapped_entities
    tok = toks[start_idx]
    if tok == "*":
        return start_idx + 1, schema.idMap[tok]

    if "." in tok:  # if token is a composite -> 意思是表名.列名形式的token不该被拆开？
        alias, col = tok.split(".")
        key = tables_with_alias[alias] + "." + col
        mapped_entities.append((start_idx, tables_with_alias[alias] + "@" + col))
        return start_idx + 1, schema.idMap[key]

    assert (
        default_tables is not None and len(default_tables) > 0
    ), "Default tables should not be None or empty"

    for alias in default_tables:
        table = tables_with_alias[alias]
        if tok in schema.schema[table]:
            key = table + "." + tok
            mapped_entities.append((start_idx, table + "@" + tok))
            return start_idx + 1, schema.idMap[key]
    # 处理TIME_NOW关键字
    if tok.upper() == 'TIME_NOW':
        return start_idx + 1, tok
    assert False, "Error col: {}".format(tok)


def parse_col_unit(toks, start_idx, tables_with_alias, schema, default_tables=None, use_capital=False):
    """
    :returns next idx, (agg_op id, col_id)
    """
    idx = start_idx
    len_ = len(toks)
    isBlock = False
    isDistinct = False
    if toks[idx] == "(":
        isBlock = True
        idx += 1

    if toks[idx] in AGG_OPS:
        agg_id = AGG_OPS.index(toks[idx])
        idx += 1
        assert idx < len_ and toks[idx] == "("
        idx += 1
        if toks[idx] == "distinct":
            idx += 1
            isDistinct = True
        idx, col_id = parse_col(toks, idx, tables_with_alias, schema, default_tables)
        assert idx < len_ and toks[idx] == ")"
        idx += 1
        return idx, (agg_id, col_id, isDistinct)

    if toks[idx] == "distinct":
        idx += 1
        isDistinct = True
    agg_id = AGG_OPS.index("none")
    idx, col_id = parse_col(toks, idx, tables_with_alias, schema, default_tables)

    if isBlock:
        assert toks[idx] == ")"
        idx += 1  # skip ')'

    return idx, (agg_id, col_id, isDistinct)


def parse_val_unit(toks, start_idx, tables_with_alias, schema, default_tables=None, use_capital=False):
    idx = start_idx
    len_ = len(toks)
    isBlock = False
    if toks[idx] == "(":
        isBlock = True
        idx += 1

    col_unit1 = None
    col_unit2 = None
    unit_op = UNIT_OPS.index("none")

    idx, col_unit1 = parse_col_unit(
        toks, idx, tables_with_alias, schema, default_tables
    )
    if idx < len_ and toks[idx] in UNIT_OPS:
        unit_op = UNIT_OPS.index(toks[idx])
        idx += 1
        idx, col_unit2 = parse_col_unit(
            toks, idx, tables_with_alias, schema, default_tables
        )

    if isBlock:
        assert toks[idx] == ")"
        idx += 1  # skip ')'
    if unit_op in (UNIT_OPS.index('+'), UNIT_OPS.index('*')):
        col_unit1, col_unit2 = sorted([col_unit1, col_unit2])

    return idx, (unit_op, col_unit1, col_unit2)


def parse_table_unit(toks, start_idx, tables_with_alias, schema, use_capital=False):
    """
    :returns next idx, table id, table name
    """
    idx = start_idx
    len_ = len(toks)
    key = tables_with_alias[toks[idx]]

    if idx + 1 < len_ and toks[idx + 1] == "as":
        idx += 3
    else:
        idx += 1

    return idx, schema.idMap[key], key


def parse_value(toks, start_idx, tables_with_alias, schema, default_tables=None, use_capital=False):
    idx = start_idx
    len_ = len(toks)

    isBlock = False
    if toks[idx] == "(":
        isBlock = True
        idx += 1

    if toks[idx] == "select":
        idx, val = parse_sql(toks, idx, tables_with_alias, schema)
    elif (toks[idx].startswith('"') and toks[idx].endswith('"')) or toks[idx].startswith("'") and toks[idx].endswith("'"):  # token is a string value
        val = toks[idx]
        idx += 1
    else:
        try:
            val = float(toks[idx])
            idx += 1
        except:
            end_idx = idx
            while (
                end_idx < len_
                and toks[end_idx] != ","
                and toks[end_idx] != ")"
                and toks[end_idx] != "and"
                and toks[end_idx] not in CLAUSE_KEYWORDS
                and toks[end_idx] not in JOIN_KEYWORDS
            ):
                end_idx += 1

            idx, val = parse_col_unit(
                toks[start_idx:end_idx], 0, tables_with_alias, schema, default_tables
            )
            idx = end_idx

    if isBlock:
        assert toks[idx] == ")"
        idx += 1

    return idx, val


def parse_condition(toks, start_idx, tables_with_alias, schema, default_tables=None, use_capital=False):
    idx = start_idx
    len_ = len(toks)
    conds = []

    while idx < len_:
        agg_id = 0
        if idx < len_ and toks[idx] in AGG_OPS:
            agg_id = AGG_OPS.index(toks[idx])
            idx += 1
            
        idx, val_unit = parse_val_unit(
            toks, idx, tables_with_alias, schema, default_tables
        )
        not_op = False
        if toks[idx] == "not":
            not_op = True
            idx += 1

        assert (
            idx < len_ and toks[idx] in WHERE_OPS
        ), "Error condition: idx: {}, tok: {}".format(idx, toks[idx])
        op_id = WHERE_OPS.index(toks[idx])
        idx += 1
        val1 = val2 = None
        if op_id == WHERE_OPS.index(
            "between"
        ):  # between..and... special case: dual values
            idx, val1 = parse_value(
                toks, idx, tables_with_alias, schema, default_tables
            )
            assert toks[idx] == "and"
            idx += 1
            idx, val2 = parse_value(
                toks, idx, tables_with_alias, schema, default_tables
            )
        else:  # normal case: single value
            idx, val1 = parse_value(
                toks, idx, tables_with_alias, schema, default_tables
            )
            val2 = None

        conds.append((not_op, op_id, val_unit, val1, val2))

        if idx < len_ and (
            toks[idx] in CLAUSE_KEYWORDS
            or toks[idx] in (")", ";")
            or toks[idx] in JOIN_KEYWORDS
        ):
            break

        if idx < len_ and toks[idx] in COND_OPS:
            conds.append(toks[idx])
            idx += 1  # skip and/or

    return idx, conds


def parse_select(toks, start_idx, tables_with_alias, schema, default_tables=None, use_capital=False):
    idx = start_idx
    len_ = len(toks)

    assert toks[idx] == "select", "'select' not found"
    idx += 1
    isDistinct = False
    if idx < len_ and toks[idx] == "distinct":
        idx += 1
        isDistinct = True
    val_units = []

    while idx < len_ and toks[idx] not in CLAUSE_KEYWORDS:
        agg_id = AGG_OPS.index("none")
        if toks[idx] in AGG_OPS:
            agg_id = AGG_OPS.index(toks[idx])
            idx += 1
        idx, val_unit = parse_val_unit(
            toks, idx, tables_with_alias, schema, default_tables
        )
        val_units.append((agg_id, val_unit))
        if idx < len_ and toks[idx] == ",":
            idx += 1  # skip ','

    return idx, (isDistinct, val_units)


def parse_from(toks, start_idx, tables_with_alias, schema, use_capital=False):
    """
    Assume in the from clause, all table units are combined with join
    """
    assert "from" in toks[start_idx:], "'from' not found"

    len_ = len(toks)
    idx = toks.index("from", start_idx) + 1
    default_tables = []
    table_units = []
    conds = []
    last_table = None

    while idx < len_:
        isBlock = False
        if toks[idx] == "(":
            isBlock = True
            idx += 1

        if toks[idx] == "select":
            idx, sql = parse_sql(toks, idx, tables_with_alias, schema)
            table_units.append((TABLE_TYPE["sql"], sql))
            last_table = sql['from']['table_units'][0][1].strip('_')
        else:
            if idx < len_ and toks[idx] == "join":
                idx += 1  # skip join
            idx, table_unit, table_name = parse_table_unit(
                toks, idx, tables_with_alias, schema
            )
            table_units.append((TABLE_TYPE["table_unit"], table_unit))
            default_tables.append(table_name)
        if idx < len_ and toks[idx] == "on":
            idx += 1  # skip on
            idx, this_conds = parse_condition(
                toks, idx, tables_with_alias, schema, default_tables
            )
            if len(conds) > 0:
                conds.append("and")
            conds.extend(this_conds)

        if isBlock:
            assert toks[idx] == ")"
            idx += 1
        if idx < len_ and toks[idx].lower() == 'a':
            # 对于不适用as 的a、b别名，进行特殊处理
            assert last_table is not None, 'last_table should be a table name strin, not None'
            tables_with_alias['a'] = last_table
            idx += 2
        elif idx < len_ and toks[idx].lower() == 'b':
            assert last_table is not None, 'last_table should be a table name strin, not None'
            tables_with_alias['b'] = last_table
            idx += 1
        if idx < len_ and (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
            break

    return idx, table_units, conds, default_tables


def parse_where(toks, start_idx, tables_with_alias, schema, default_tables, use_capital=False):
    idx = start_idx
    len_ = len(toks)

    if idx >= len_ or toks[idx] != "where":
        return idx, []

    idx += 1
    idx, conds = parse_condition(toks, idx, tables_with_alias, schema, default_tables)
    return idx, conds


def parse_group_by(toks, start_idx, tables_with_alias, schema, default_tables, use_capital=False):
    idx = start_idx
    len_ = len(toks)
    col_units = []

    if idx >= len_ or toks[idx] != "group":
        return idx, col_units

    idx += 1
    assert toks[idx] == "by"
    idx += 1

    while idx < len_ and not (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
        idx, col_unit = parse_col_unit(
            toks, idx, tables_with_alias, schema, default_tables
        )
        col_units.append(col_unit)
        if idx < len_ and toks[idx] == ",":
            idx += 1  # skip ','
        else:
            break

    return idx, col_units


def parse_order_by(toks, start_idx, tables_with_alias, schema, default_tables, use_capital=False):
    idx = start_idx
    len_ = len(toks)
    val_units = []
    order_type = "ASC"  # default type is 'asc'

    if idx >= len_ or toks[idx] != ORDER:
        return idx, val_units

    idx += 1
    assert toks[idx] == BY
    idx += 1

    while idx < len_ and not (toks[idx] in CURRENT_CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
        agg_id = CURRENT_AGG_OPS.index(NONE)
        if toks[idx] in CURRENT_AGG_OPS:
            agg_id = CURRENT_AGG_OPS.index(toks[idx])
            idx += 1
        idx, val_unit = parse_val_unit(
            toks, idx, tables_with_alias, schema, default_tables, use_capital=use_capital
        )
        val_units.append(val_unit)
        if idx < len_ and toks[idx] in CURRENT_ORDER_OPS:
            order_type = toks[idx]
            idx += 1
        if idx < len_ and toks[idx] == ",":
            idx += 1  # skip ','
        else:
            break

    return idx, (order_type, val_units)


def parse_having(toks, start_idx, tables_with_alias, schema, default_tables, use_capital=False):
    idx = start_idx
    len_ = len(toks)
    HAVING = "HAVING"
    if not use_capital:
        HAVING = "having"

    if idx >= len_ or toks[idx] != HAVING:
        return idx, []

    idx += 1
    idx, conds = parse_condition(toks, idx, tables_with_alias, schema, default_tables, use_capital=use_capital)
    return idx, conds


def parse_limit(toks, start_idx, use_capital=False):
    idx = start_idx
    len_ = len(toks)

    LIMIT = "LIMIT"
    VALUE = '"VALUE"'
    if not use_capital:
        LIMIT = "limit"
        VALUE = '"value"'

    if idx < len_ and toks[idx] == LIMIT:
        idx += 2
        try:
            limit_val = int(toks[idx - 1])
        except Exception:
            limit_val = VALUE
        return idx, limit_val

    return idx, None


def parse_sql(toks, start_idx, tables_with_alias, schema, mapped_entities_fn=None, use_capital=False):
    global mapped_entities

    if mapped_entities_fn is not None:
        mapped_entities = mapped_entities_fn()
    isBlock = False  # indicate whether this is a block of sql/sub-sql
    len_ = len(toks)
    idx = start_idx

    sql = {}
    if toks[idx] == "(":
        isBlock = True
        idx += 1

    # parse from clause in order to get default tables
    from_end_idx, table_units, conds, default_tables = parse_from(
        toks, start_idx, tables_with_alias, schema, use_capital=use_capital
    )
    sql["from"] = {"table_units": table_units, "conds": conds}
    # select clause
    _, select_col_units = parse_select(
        toks, idx, tables_with_alias, schema, default_tables, use_capital=use_capital
    )
    idx = from_end_idx
    sql["select"] = select_col_units
    # where clause
    idx, where_conds = parse_where(toks, idx, tables_with_alias, schema, default_tables, use_capital=use_capital)
    sql["where"] = where_conds
    # group by clause
    idx, group_col_units = parse_group_by(
        toks, idx, tables_with_alias, schema, default_tables, use_capital=use_capital
    )
    sql["groupBy"] = group_col_units
    # having clause
    idx, having_conds = parse_having(
        toks, idx, tables_with_alias, schema, default_tables, use_capital=use_capital
    )
    sql["having"] = having_conds
    # order by clause
    idx, order_col_units = parse_order_by(
        toks, idx, tables_with_alias, schema, default_tables, use_capital=use_capital
    )
    sql["orderBy"] = order_col_units
    # limit clause
    idx, limit_val = parse_limit(toks, idx, use_capital=use_capital)
    sql["limit"] = limit_val

    idx = skip_semicolon(toks, idx)
    if isBlock:
        assert toks[idx] == ")"
        idx += 1  # skip ')'
    idx = skip_semicolon(toks, idx)

    # intersect/union/except clause
    for op in SQL_OPS:  # initialize IUE
        sql[op] = None
    if idx < len_ and toks[idx] in SQL_OPS:
        sql_op = toks[idx]
        idx += 1
        idx, IUE_sql = parse_sql(toks, idx, tables_with_alias, schema, use_capital=use_capital)
        sql[sql_op] = IUE_sql

    if mapped_entities_fn is not None:
        return idx, sql, mapped_entities
    else:
        return idx, sql

def get_sql(schema, query):
    toks = tokenize(query)
    tables_with_alias = get_tables_with_alias(schema.schema, toks)
    _, sql = parse_sql(toks, 0, tables_with_alias, schema)

    return sql

def get_sql_using_dusql_tokenizer(schema, query, silent=False):
    toks = tokenize_dusql(query)
    tables_with_alias = get_tables_with_alias(schema.schema, toks)
    try:
        _, sql = parse_sql(toks, 0, tables_with_alias, schema, use_capital=False)
    except Exception as e:
        if not silent:
            logging.exception(e)
        raise e
    return sql

def get_sql_using_sqlparse(schema, query):
    toks = tokenize_query(query)
    tables_with_alias = get_tables_with_alias(schema.schema, toks)
    _, sql = parse_sql(toks, 0, tables_with_alias, schema)
    return sql


def skip_semicolon(toks, start_idx):
    idx = start_idx
    while idx < len(toks) and toks[idx] == ";":
        idx += 1
    return idx