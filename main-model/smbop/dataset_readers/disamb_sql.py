"""
Utility functions for reading the standardised text2sql datasets presented in
`"Improving Text to SQL Evaluation Methodology" <https://arxiv.org/abs/1806.09029>`_
"""
import json
import os
import sqlite3
from collections import defaultdict
from typing import List, Dict, Optional
import json
import sqlite3
from nltk import word_tokenize
import logging

from allennlp.common import JsonDict

# from spider_evaluation.process_sql import get_tables_with_alias, parse_sql


class TableColumn:
    def __init__(
        self,
        name: str,
        text: str,
        column_type: str,
        is_primary_key: bool,
        foreign_key: Optional[str],
    ):
        self.name = name
        self.text = text
        self.column_type = column_type
        self.is_primary_key = is_primary_key
        self.foreign_key = foreign_key


class Table:
    def __init__(self, name: str, text: str, columns: List[TableColumn]):
        self.name = name
        self.text = text
        self.columns = columns


def read_dataset_schema(schema_path: str) -> Dict[str, List[Table]]:
    schemas: Dict[str, Dict[str, Table]] = defaultdict(dict)
    with open(schema_path, "r") as f:
        dbs_json_blob = json.load(f)
        for db in dbs_json_blob:
            db_id = db["db_id"]

            column_id_to_table = {}
            column_id_to_column = {}

            for i, (column, text, column_type) in enumerate(
                zip(db["column_names_original"], db["column_names"], db["column_types"])
            ):
                table_id, column_name = column
                _, column_text = text

                table_name = db["table_names_original"][table_id].lower()
                # table_name = db["table_names_original"][table_id].upper()

                if table_name not in schemas[db_id]:
                    table_text = db["table_names"][table_id]
                    schemas[db_id][table_name] = Table(table_name, table_text, [])

                if column_name == "*":
                    continue

                is_primary_key = i in db["primary_keys"]
                table_column = TableColumn(
                    column_name.lower(), column_text, column_type, is_primary_key, None
                    # column_name.upper(), column_text, column_type, is_primary_key, None
                )
                schemas[db_id][table_name].columns.append(table_column)
                column_id_to_table[i] = table_name
                column_id_to_column[i] = table_column

            for (c1, c2) in db["foreign_keys"]:
                foreign_key = (
                    column_id_to_table[c2] + ":" + column_id_to_column[c2].name
                )
                column_id_to_column[c1].foreign_key = foreign_key

    return {**schemas}


def read_dataset_values(db_id: str, dataset_path: str, tables: List[str]):
    db = os.path.join(dataset_path, db_id, db_id + ".sqlite") # FIXME: 无用方法
    try:
        conn = sqlite3.connect(db)
    except Exception as e:
        raise Exception(f"Can't connect to SQL: {e} in path {db}")
    conn.text_factory = str
    cursor = conn.cursor()

    values = {}
    c = "1"  # TODO: fixme
    if False:  # fixme
        for table in tables:
            try:
                cursor.execute(f"SELECT * FROM {table.name} LIMIT {c}")
                values[table] = cursor.fetchall()
            except:
                conn.text_factory = lambda x: str(x, "latin1")
                cursor = conn.cursor()
                cursor.execute(f"SELECT * FROM {table.name} LIMIT {c}")
                values[table] = cursor.fetchall()

    return values


def ent_key_to_name(key):
    parts = key.split(":")
    if parts[0] == "table":
        return parts[1]
    elif parts[0] == "column":
        _, _, table_name, column_name = parts
        return f"{table_name}@{column_name}"
    else:
        return parts[1]


def fix_number_value(ex: JsonDict):
    """
    There is something weird in the dataset files - the `query_toks_no_value` field anonymizes all values,
    which is good since the evaluator doesn't check for the values. But it also anonymizes numbers that
    should not be anonymized: e.g. LIMIT 3 becomes LIMIT 'value', while the evaluator fails if it is not a number.
    """

    def split_and_keep(s, sep):
        if not s:
            return [""]  # consistent with string.split()

        # Find replacement character that is not used in string
        # i.e. just use the highest available character plus one
        # Note: This fails if ord(max(s)) = 0x10FFFF (ValueError)
        p = chr(ord(max(s)) + 1)

        return s.replace(sep, p + sep + p).split(p)

    # input is tokenized in different ways... so first try to make splits equal
    query_toks = ex["query_toks"]
    ex["query_toks"] = []
    for q in query_toks:
        ex["query_toks"] += split_and_keep(q, ".")

    i_val, i_no_val = 0, 0
    while i_val < len(ex["query_toks"]) and i_no_val < len(ex["query_toks_no_value"]):
        if ex["query_toks_no_value"][i_no_val] != "value":
            i_val += 1
            i_no_val += 1
            continue

        i_val_end = i_val
        while (
            i_val + 1 < len(ex["query_toks"])
            and i_no_val + 1 < len(ex["query_toks_no_value"])
            and ex["query_toks"][i_val_end + 1].lower()
            != ex["query_toks_no_value"][i_no_val + 1].lower()
        ):
            i_val_end += 1

        if (
            i_val == i_val_end
            and ex["query_toks"][i_val] in ["1", "2", "3", "4", "5"]
            and ex["query_toks"][i_val - 1].lower() == "limit"
        ):
            ex["query_toks_no_value"][i_no_val] = ex["query_toks"][i_val]
        i_val = i_val_end

        i_val += 1
        i_no_val += 1

    return ex


_schemas_cache = None


def get_schema_from_db_id(db_id: str, tables_file: str, use_capital=False):
    # class Schema:
    #     """
    #     Simple schema which maps table&column to a unique identifier
    #     """

    #     def __init__(self, schema, table):
    #         self._schema = schema
    #         self._table = table
    #         self._idMap = self._map(self._schema, self._table)

    #     @property
    #     def schema(self):
    #         return self._schema

    #     @property
    #     def idMap(self):
    #         return self._idMap

    #     def _map(self, schema, table):
    #         column_names_original = table["column_names_original"]
    #         table_names_original = table["table_names_original"]
    #         # print 'column_names_original: ', column_names_original
    #         # print 'table_names_original: ', table_names_original
    #         for i, (tab_id, col) in enumerate(column_names_original):
    #             if tab_id == -1:
    #                 idMap = {"*": i}
    #             else:
    #                 # key = table_names_original[tab_id].lower()
    #                 # val = col.lower()
    #                 key = table_names_original[tab_id].upper()
    #                 val = col.upper()
    #                 idMap[key + "." + val] = i

    #         for i, tab in enumerate(table_names_original):
    #             # key = tab.lower()
    #             key = tab.upper()
    #             idMap[key] = i

    #         return idMap

    def get_schemas_from_json(fpath, use_capital=False):
        global _schemas_cache

        if _schemas_cache is not None:
            return _schemas_cache

        with open(fpath) as f:
            data = json.load(f)
        db_names = [db["db_id"] for db in data]

        tables = {}
        schemas = {}
        for db in data:
            db_id = db["db_id"]
            schema = {}  # {'table': [col.lower, ..., ]} * -> __all__
            column_names_original = db["column_names_original"]
            table_names_original = db["table_names_original"]
            tables[db_id] = {
                "column_names_original": column_names_original,
                "table_names_original": table_names_original,
            }
            for i, tabn in enumerate(table_names_original):
                # table = str(tabn.lower())
                if use_capital:
                    table = str(tabn.upper())
                else:
                    table = str(tabn.lower())
                if use_capital:
                    cols = [str(col.upper()) for td, col in column_names_original if td == i]
                else:
                    cols = [str(col.lower()) for td, col in column_names_original if td == i]
                schema[table] = cols
            schemas[db_id] = schema

        _schemas_cache = schemas, db_names, tables
        return _schemas_cache

    schemas, db_names, tables = get_schemas_from_json(tables_file, use_capital=use_capital)
    # schema = Schema(schemas[db_id], tables[db_id])
    schema = Schema(schemas[db_id])
    return schema


def sanitize_toks(query_toks: list):
    for index in range(len(query_toks)):
        if query_toks[index] == '==':
            query_toks[index] = '='



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


# 需要跑一个Spider数据看看这个地方怎么处理的
def disambiguate_items(
    db_id: str, query_toks: List[str], tables_file: str, allow_aliases: bool
) -> List[str]:
    """
    we want the query tokens to be non-ambiguous - so we can change each column name to explicitly
    tell which table it belongs to

    parsed sql to sql clause is based on supermodel.gensql from syntaxsql
    """
    # TODO: 需要使用Replacer？
    schema = get_schema_from_db_id(db_id, tables_file, use_capital=False)
    fixed_toks = []
    i = 0
    while i < len(query_toks):
        tok = query_toks[i]
        if tok == "value" or tok == "'value'":
            # TODO: value should alawys be between '/" (remove first if clause)
            new_tok = f'"{tok}"'
        elif tok in ["!", "<", ">"] and query_toks[i + 1] == "=":
            new_tok = tok + "="
            i += 1
        elif i + 1 < len(query_toks) and query_toks[i + 1] == ".":
            new_tok = "".join(query_toks[i : i + 3])
            i += 2
        else:
            new_tok = tok
        fixed_toks.append(new_tok)
        i += 1

    toks = fixed_toks

    tables_with_alias = get_tables_with_alias(schema.schema, toks)

    # if 'dusql' in tables_file:
    #     use_capital = False
    # else:
    #     use_capital = True

    _, sql, mapped_entities = parse_sql(
        toks, 0, tables_with_alias, schema, mapped_entities_fn=lambda: [], use_capital=False
    )

    for i, new_name in mapped_entities:
        curr_tok = toks[i]
        if "." in curr_tok and allow_aliases:
            parts = curr_tok.split(".")
            assert len(parts) == 2
            toks[i] = parts[0] + "." + new_name
        else:
            toks[i] = new_name.replace("@", ".")

    if not allow_aliases:
        # 只删除AS和AS后面的别名，其它别名如果能够替换则替换，不能替换则保留。即只删除别名的定义，其它别名如果可以替换则替换，出现无法替换的情况则不替换
        for index in range(len(toks)):
            if toks[index].lower() in ["t1", "t2", "t3", "t4", "a", "b"]:
                if index - 1 >= 0 and toks[index - 1].lower() == 'as':
                    # 删除这个别名定义
                    toks[index] = ''
                elif toks[index] in tables_with_alias:
                    # 查看是否可以替换
                        toks[index] = tables_with_alias[toks[index]]
                    
        toks = [tok for tok in toks if tok.lower() not in ["as", ""]]

    toks = [f"'value'" if tok == '"value"' else tok for tok in toks]
    query_tokens = " ".join(toks)
    # query_tokens = query_tokens
    for agg in ["count", "min", "max", "sum", "avg", "COUNT", "MIN", "MAX", "SUM", "AVG"]:
        query_tokens = query_tokens.replace(f"{agg} (", f"{agg}(")
    return query_tokens




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
            idMap[key.lower()] = "__" + key.lower() + "__"
            # idMap[key.upper()] = "__" + key.upper() + "__"
            id += 1

        return idMap


def get_schema(db):
    """
    Get database's schema, which is a dict with table name as key
    and list of column names as value
    :param db: database path
    :return: schema dict
    """
    # FIXME: 无用方法
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


# def get_schema_from_json(fpath):
#     with open(fpath) as f:
#         data = json.load(f)

#     schema = {}
#     for entry in data:
#         # table = str(entry["table"].lower())
#         table = str(entry["table"].upper())
#         # cols = [str(col["column_name"].lower()) for col in entry["col_data"]]
#         cols = [str(col["column_name"].upper()) for col in entry["col_data"]]
#         schema[table] = cols

#     return schema


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
        # return start_idx + 1, schema.idMap[tok]
        return start_idx + 1, tok

    if "." in tok:  # if token is a composite
        alias, col = tok.split(".")
        key = tables_with_alias[alias] + "." + col
        mapped_entities.append((start_idx, tables_with_alias[alias] + "@" + col))
        # return start_idx + 1, schema.idMap[key]
        return start_idx + 1, tok

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
    DISTINCT = "DISTINCT"
    CURRENT_AGG_OPS = AGG_OPS_CAPITAL
    NONE = "NONE"
    if not use_capital:
        DISTINCT = DISTINCT.lower()
        NONE = NONE.lower()
        CURRENT_AGG_OPS = AGG_OPS

    idx = start_idx
    len_ = len(toks)
    isBlock = False
    isDistinct = False
    if toks[idx] == "(":
        isBlock = True
        idx += 1

    if toks[idx] in CURRENT_AGG_OPS:
        agg_id = CURRENT_AGG_OPS.index(toks[idx])
        idx += 1
        assert idx < len_ and toks[idx] == "("
        idx += 1
        if toks[idx] == DISTINCT:
            idx += 1
            isDistinct = True
        idx, col_id = parse_col(toks, idx, tables_with_alias, schema, default_tables, use_capital=use_capital)
        assert idx < len_ and toks[idx] == ")"
        idx += 1
        return idx, (agg_id, col_id, isDistinct)

    if toks[idx] == DISTINCT:
        idx += 1
        isDistinct = True
    agg_id = CURRENT_AGG_OPS.index(NONE)
    idx, col_id = parse_col(toks, idx, tables_with_alias, schema, default_tables, use_capital=use_capital)

    if isBlock:
        assert toks[idx] == ")"
        idx += 1  # skip ')'

    return idx, (agg_id, col_id, isDistinct)


def parse_val_unit(toks, start_idx, tables_with_alias, schema, default_tables=None, use_capital=False):
    NONE = "NONE"
    CURRENT_UNIT_OPS = UNIT_OPS_CAPITAL
    if not use_capital:
        NONE = NONE.lower()
        CURRENT_UNIT_OPS = UNIT_OPS
    idx = start_idx
    len_ = len(toks)
    isBlock = False

    if toks[idx] == "(":
        isBlock = True
        idx += 1

    col_unit1 = None
    col_unit2 = None
    try:
        unit_op = CURRENT_UNIT_OPS.index(NONE)
    except Exception as e:
        logging.exception(e)

    idx, col_unit1 = parse_col_unit(
        toks, idx, tables_with_alias, schema, default_tables, use_capital=use_capital
    )
    if idx < len_ and toks[idx] in CURRENT_UNIT_OPS:
        unit_op = CURRENT_UNIT_OPS.index(toks[idx])
        idx += 1
        idx, col_unit2 = parse_col_unit(
            toks, idx, tables_with_alias, schema, default_tables, use_capital=use_capital
        )

    if isBlock:
        assert toks[idx] == ")"
        idx += 1  # skip ')'

    return idx, (unit_op, col_unit1, col_unit2)


def parse_table_unit(toks, start_idx, tables_with_alias, schema, use_capital=False):
    """
    :returns next idx, table id, table name
    """
    AS = "AS"
    if not use_capital:
        AS = AS.lower()
    idx = start_idx
    len_ = len(toks)
    try:
        key = tables_with_alias[toks[idx]]
    except Exception as e:
        raise e

    if idx + 1 < len_ and toks[idx + 1] == AS:
        idx += 3
    else:
        idx += 1

    return idx, schema.idMap[key], key


def parse_value(toks, start_idx, tables_with_alias, schema, default_tables=None, use_capital=False):
    idx = start_idx
    len_ = len(toks)

    SELECT = "SELECT"
    AND = "AND"
    CURRENT_CLAUSE_KEYWORDS = CLAUSE_KEYWORDS_CAPITAL
    CURRENT_JOIN_KEYWORDS = JOIN_KEYWORDS_CAPITAL
    if not use_capital:
        SELECT = SELECT.lower()
        AND = AND.lower()
        CURRENT_CLAUSE_KEYWORDS = CLAUSE_KEYWORDS
        CURRENT_JOIN_KEYWORDS = JOIN_KEYWORDS

    isBlock = False
    if toks[idx] == "(":
        isBlock = True
        idx += 1

    if toks[idx] == SELECT:
        idx, val = parse_sql(toks, idx, tables_with_alias, schema, use_capital=use_capital)
    elif '"' in toks[idx] or "'" in toks[idx]:  # token is a string value
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
                and toks[end_idx] != AND
                and toks[end_idx] not in CURRENT_CLAUSE_KEYWORDS
                and toks[end_idx] not in CURRENT_JOIN_KEYWORDS
            ):
                end_idx += 1

            idx, val = parse_col_unit(
                toks[:end_idx], start_idx, tables_with_alias, schema, default_tables, use_capital=use_capital
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

    NOT = "NOT"
    CURRENT_WHERE_OPS = WHERE_OPS_CAPITAL
    BETWEEN = "BETWEEN"
    AND = "AND"
    CURRENT_CLAUSE_KEYWORDS = CLAUSE_KEYWORDS_CAPITAL
    CURRENT_JOIN_KEYWORDS = JOIN_KEYWORDS_CAPITAL
    CURRENT_COND_OPS = COND_OPS_CAPITAL
    CURRENT_AGG_OPS = AGG_OPS_CAPITAL
    if not use_capital:
        NOT = NOT.lower()
        CURRENT_WHERE_OPS = WHERE_OPS
        BETWEEN = BETWEEN.lower()
        AND = AND.lower()
        CURRENT_CLAUSE_KEYWORDS = CLAUSE_KEYWORDS
        CURRENT_JOIN_KEYWORDS = JOIN_KEYWORDS
        CURRENT_COND_OPS = COND_OPS
        CURRENT_AGG_OPS = AGG_OPS

    while idx < len_:
        agg_id = 0
        if idx < len_ and toks[idx] in CURRENT_AGG_OPS:
            agg_id = CURRENT_AGG_OPS.index(toks[idx])
            idx += 1

        idx, val_unit = parse_val_unit(
            toks, idx, tables_with_alias, schema, default_tables, use_capital=use_capital
        )
        not_op = False
        if toks[idx] == NOT:
            not_op = True
            idx += 1

        assert (
            idx < len_ and toks[idx] in CURRENT_WHERE_OPS
        ), "Error condition: idx: {}, tok: {}".format(idx, toks[idx])
        op_id = CURRENT_WHERE_OPS.index(toks[idx])
        idx += 1
        val1 = val2 = None
        if op_id == CURRENT_WHERE_OPS.index(
            BETWEEN
        ):  # between..and... special case: dual values
            idx, val1 = parse_value(
                toks, idx, tables_with_alias, schema, default_tables, use_capital=use_capital
            )
            assert toks[idx] == AND
            idx += 1
            idx, val2 = parse_value(
                toks, idx, tables_with_alias, schema, default_tables, use_capital=use_capital
            )
        else:  # normal case: single value
            idx, val1 = parse_value(
                toks, idx, tables_with_alias, schema, default_tables, use_capital=use_capital
            )
            val2 = None

        conds.append((not_op, op_id, val_unit, val1, val2))

        if idx < len_ and (
            toks[idx] in CURRENT_CLAUSE_KEYWORDS
            or toks[idx] in (")", ";")
            or toks[idx] in CURRENT_JOIN_KEYWORDS
        ):
            break

        if idx < len_ and toks[idx] in CURRENT_COND_OPS:
            conds.append(toks[idx])
            idx += 1  # skip and/or

    return idx, conds


def parse_select(toks, start_idx, tables_with_alias, schema, default_tables=None, use_capital=False):
    SELECT = "SELECT"
    DISTINCT = "DISTINCT"
    NONE = "NONE"
    CURRENT_AGG_OPS = AGG_OPS_CAPITAL
    CURRENT_CLAUSE_KEYWORDS = CLAUSE_KEYWORDS_CAPITAL
    if not use_capital:
        SELECT = "select"
        DISTINCT = DISTINCT.lower()
        NONE = NONE.lower()
        CURRENT_AGG_OPS = AGG_OPS
        CURRENT_CLAUSE_KEYWORDS = CLAUSE_KEYWORDS
    idx = start_idx
    len_ = len(toks)

    assert toks[idx] == SELECT, "'select' not found"
    idx += 1
    isDistinct = False
    if idx < len_ and toks[idx] == DISTINCT:
        idx += 1
        isDistinct = True
    val_units = []

    while idx < len_ and toks[idx] not in CURRENT_CLAUSE_KEYWORDS:
        agg_id = CURRENT_AGG_OPS.index(NONE)
        if toks[idx] in CURRENT_AGG_OPS:
            agg_id = CURRENT_AGG_OPS.index(toks[idx])
            idx += 1
        idx, val_unit = parse_val_unit(
            toks, idx, tables_with_alias, schema, default_tables, use_capital=use_capital
        )
        val_units.append((agg_id, val_unit))
        if idx < len_ and toks[idx] == ",":
            idx += 1  # skip ','

    return idx, (isDistinct, val_units)


def parse_from(toks, start_idx, tables_with_alias, schema, use_capital=False):
    SELECT = 'SELECT'
    FROM = 'FROM'
    JOIN = 'JOIN'
    ON = 'ON'
    AND = 'AND'
    CURRENT_CLAUSE_KEYWORDS = CLAUSE_KEYWORDS_CAPITAL
    if not use_capital:
        SELECT = 'select'
        FROM = 'from'
        JOIN = 'join'
        ON = 'on'
        AND = 'and'
        CURRENT_CLAUSE_KEYWORDS = CLAUSE_KEYWORDS

    """
    Assume in the from clause, all table units are combined with join
    """
    assert FROM in toks[start_idx:], "'from' not found"

    len_ = len(toks)
    idx = toks.index(FROM, start_idx) + 1
    default_tables = []
    table_units = []
    conds = []
    last_table = None

    while idx < len_:
        isBlock = False
        if toks[idx] == "(":
            isBlock = True
            idx += 1

        if toks[idx] == SELECT:
            idx, sql = parse_sql(toks, idx, tables_with_alias, schema, use_capital=use_capital)
            table_units.append((TABLE_TYPE["sql"], sql))
            last_table = sql['from']['table_units'][0][1].strip('_')
        else:
            if idx < len_ and toks[idx] == JOIN:
                idx += 1  # skip join
            idx, table_unit, table_name = parse_table_unit(
                toks, idx, tables_with_alias, schema, use_capital=use_capital
            )
            table_units.append((TABLE_TYPE["table_unit"], table_unit))
            default_tables.append(table_name)
        if idx < len_ and toks[idx] == ON:
            idx += 1  # skip on
            idx, this_conds = parse_condition(
                toks, idx, tables_with_alias, schema, default_tables, use_capital=use_capital
            )
            if len(conds) > 0:
                conds.append(AND)
            conds.extend(this_conds)

        if isBlock:
            assert toks[idx] == ")"
            idx += 1
        if idx < len_ and toks[idx] in ['a', 'A']:
            # 对于不适用as 的a、b别名，进行特殊处理
            assert last_table is not None, 'last_table should be a table name strin, not None'
            tables_with_alias[toks[idx]] = last_table
            idx += 2
        elif idx < len_ and toks[idx] in ['b', 'B']:
            assert last_table is not None, 'last_table should be a table name strin, not None'
            tables_with_alias[toks[idx]] = last_table
            idx += 1
        if idx < len_ and (toks[idx] in CURRENT_CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
            break

    return idx, table_units, conds, default_tables


def parse_where(toks, start_idx, tables_with_alias, schema, default_tables, use_capital=False):
    WHERE = "WHERE"
    if not use_capital:
        WHERE = "where"
    idx = start_idx
    len_ = len(toks)

    if idx >= len_ or toks[idx] != WHERE:
        return idx, []

    idx += 1
    idx, conds = parse_condition(toks, idx, tables_with_alias, schema, default_tables, use_capital=use_capital)
    return idx, conds


def parse_group_by(toks, start_idx, tables_with_alias, schema, default_tables, use_capital=False):
    GROUP = "GROUP"
    BY = "BY"
    CURRENT_CLAUSE_KEYWORDS = CLAUSE_KEYWORDS_CAPITAL
    if not use_capital:
        GROUP = "group"
        BY = "by"
        CURRENT_CLAUSE_KEYWORDS = CLAUSE_KEYWORDS
    idx = start_idx
    len_ = len(toks)
    col_units = []

    if idx >= len_ or toks[idx] != GROUP:
        return idx, col_units

    idx += 1
    assert toks[idx] == BY
    idx += 1

    while idx < len_ and not (toks[idx] in CURRENT_CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
        idx, col_unit = parse_col_unit(
            toks, idx, tables_with_alias, schema, default_tables, use_capital=use_capital
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
    ORDER = "ORDER"
    BY = "BY"
    CURRENT_CLAUSE_KEYWORDS = CLAUSE_KEYWORDS_CAPITAL
    CURRENT_ORDER_OPS = ORDER_OPS_CAPITAL
    CURRENT_AGG_OPS = AGG_OPS_CAPITAL
    NONE = 'NONE'
    if not use_capital:
        order_type = "asc"  # default type is 'asc'
        ORDER = "order"
        BY = "by"
        CURRENT_CLAUSE_KEYWORDS = CLAUSE_KEYWORDS
        CURRENT_ORDER_OPS = ORDER_OPS
        CURRENT_AGG_OPS = AGG_OPS
        NONE = 'none'

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


def load_data(fpath):
    with open(fpath) as f:
        data = json.load(f)
    return data


def get_sql(schema, query):
    toks = tokenize(query)
    tables_with_alias = get_tables_with_alias(schema.schema, toks)
    _, sql = parse_sql(toks, 0, tables_with_alias, schema)

    return sql


def skip_semicolon(toks, start_idx):
    idx = start_idx
    while idx < len(toks) and toks[idx] == ";":
        idx += 1
    return idx
