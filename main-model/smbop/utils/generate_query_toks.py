import sqlparse
from tqdm import tqdm
import json
import re

# 数据集里的日期全部都不带引号, 进行修正
# TODO: 解码时再去掉
def fix_date(query: str):
    regex = "(\d{4}-\d{2}-\d{2})"
    regex_short = "(\d{4}-\d{2})"
    matched =  re.findall(regex, query)
    matched_short = re.findall(regex_short, query)
    for date in matched:
        # 先进行检查
        start_index = query.find(date)
        end_index = start_index + 10
        start_with_quote = start_index > 0 and query[start_index - 1] in ["'", '"']
        end_with_quote = end_index < len(query) and query[end_index] in ["'", '"']
        if not (start_with_quote and end_with_quote):
            query = query.replace(date, "'" + date + "'")

    for date in matched_short:
        start_index = query.find(date)
        end_index = start_index + 7
        start_with_quote = start_index > 0 and query[start_index - 1] in ["'", '"']
        end_with_quote = end_index < len(query) and query[end_index] in ["'", '"']
        if not (start_with_quote and end_with_quote):
            # 两端都没有引号才会进行添加，保证不重复添加
            query = query.replace(date, "'" + date + "'")
    return query

def fix_time(query_toks: list):
    regex = "(\d{2}:\d{2}:\d{2})"
    for index in range(len(query_toks)):
        if len(re.findall(regex, query_toks[index])) > 0 and len(re.findall('["\'`]', query_toks[index])) == 0:
            query_toks[index] = '"' + query_toks[index] + '"'
    return query_toks

def contains_chinese(y):
    # 把被引号包裹的值排除
    if (y.startswith("'") and y.endswith("'")) or (y.startswith('"') and y.endswith('"')) or (y.startswith('`') and y.endswith('`')):
        return False
    for _char in y:
        if '\u4e00' <= _char <= '\u9fa5':
            return True
    return False

# AI_SEARCH数据集里面没有query_toks字段，能否给加上?
# nltk的word_tokenize对条件值比较复杂并且包含中文的SQL语句效果不好
# 使用sqlparse完成tokenize
# TODO: 舍弃数据集自带的query_toks字段，将SQL语句重新进行parse（大小写问题）
def tokenize_query(query: str, use_back_quote=False):
    # def capitalize(y):
    #     if 'Name' in str(y.ttype) or 'Literal' in str(y.ttype) or 'Error' in str(y.ttype):
    #         return y
    #     else:
    #         y.value = y.value.upper()
    #         return y

    # 对query和question进行分词
    # TODO: 形如ORDER BY 这种类型的TOKEN需要拆分成两个，否则后续的解析会报错
    # TODO: 将关键词全部标准化为大写字母
    # TODO: DuSQL数据集的中文关键词会被Parser拆分（Token.Error），需要设法拼接回来
    # query = fix_date(query)
    parsed = sqlparse.parse(query)[0].flatten()
    # map(capitalize, parsed)
    current_query_toks = []
    current_concated = ''
    meet_error = False
    for x in parsed:
        if 'Error' in str(x.ttype) or 'Literal' in str(x.ttype):
            current_concated = current_concated + x.value
            if 'Error' in str(x.ttype):
                meet_error = True
            continue
        if 'Name' in str(x.ttype):
            current_concated = current_concated + x.value.lower()
            continue
        if 'Punctuation' in str(x.ttype) and str(x.normalized) == '.':
            current_concated = current_concated + x.value.lower()
            continue
        if current_concated != '':
            if use_back_quote and (meet_error or contains_chinese(current_concated)):
                # 中文关键字用反引号包裹，让msp能够正常解析SQL，use_back_quote只应该在被送入msp之前设置为True
                # token序列的用途只是用于语法树解析，所以.左右两边的表名和列名不分开包裹也可以
                current_concated = '`' + current_concated + '`'
            if meet_error:
                meet_error = False
            current_query_toks.append(current_concated)
            current_concated = ''
        if x.normalized == ' ':
            continue
        # 其它关键字要拆开
        current_query_toks.extend(x.value.lower().split(' '))
        meet_error = False
    if current_concated != '':
        if use_back_quote and (meet_error or contains_chinese(current_concated)):
            current_concated = '`' + current_concated + '`'
        if meet_error:
            meet_error = False
        current_query_toks.append(current_concated)
    
    # current_query_toks = [x.value for x in filter(lambda y: y.value != ' ', parsed)]
    # current_question_toks = self._tokenizer.tokenize(unit['question'])
    # current_question_toks_str_list = [item.text for item in current_question_toks]
    return current_query_toks
    # unit['question_toks'] = current_question_toks_str_list

'''
    摘自DuSQL Baseline
'''

CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except')
JOIN_KEYWORDS = ('join', 'on', 'as')

COND_OPS = ('not_in', 'between', '==', '>', '<', '>=', '<=', '!=', 'in', 'like')
UNIT_OPS = ('none', '-', '+', "*", '/')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
TABLE_TYPE = {
    'sql': "sql",
    'table_unit': "table_unit",
}

LOGIC_AND_OR = ('and', 'or')
SQL_OPS = ('intersect', 'union', 'except')
ORDER_OPS = ('desc', 'asc')

CONST_COLUMN = set(['time_now'])

EXPECT_BRACKET_PRE_TOKENS = set(AGG_OPS + SQL_OPS + COND_OPS + CLAUSE_KEYWORDS + ('from', ',', 'distinct'))

g_empty_sql = {"select": [], "from": {"conds": [], "table_units": []},
               "where": [], "groupBy": [], "having": [], "orderBy": [], "limit": None,
               "except": None, "intersect": None, "union": None}

VALUE = '1'
def tokenize_dusql(string, single_equal=True, math=True, use_back_quote=False, replace_double_quote=True):
    """
    Args:

    Returns:
    """
    # string = fix_time(string)
    # 去掉中文引号
    if replace_double_quote:
        string = string.replace("\"", "\'").lower()
        assert string.count("'") % 2 == 0, "Unexpected quote"

    def _extract_value(string):
        """extract values in sql"""
        fields = string.split("'")
        for idx, tok in enumerate(fields):
            if idx % 2 == 1:
                fields[idx] = "'%s'" % (tok)
        return fields

    def _resplit(tmp_tokens, fn_split, fn_omit):
        """resplit"""
        new_tokens = []
        for token in tmp_tokens:
            token = token.strip()
            if fn_omit(token):
                new_tokens.append(token)
            elif re.match(r'\d\d\d\d-\d\d(-\d\d)?', token):
                new_tokens.append("'%s'" % (token))
            else:
                new_tokens.extend(fn_split(token))
        return new_tokens

    tokens_tmp = _extract_value(string)

    two_bytes_op = ['==', '!=', '>=', '<=', '<>', '<in>']
    if single_equal:
        sep1 = re.compile(r'([ \+\-\*/\(\)=,><;])')  # 单字节运算符
    else:
        sep1 = re.compile(r'([ \+\-\*/\(\),><;])')  # 单字节运算符          
    sep2 = re.compile('(' + '|'.join(two_bytes_op) + ')')   # 多字节运算符
    tokens_tmp = _resplit(tokens_tmp, lambda x: x.split(' '), lambda x: x.startswith("'"))
    tokens_tmp = _resplit(tokens_tmp, lambda x: re.split(sep2, x), lambda x: x.startswith("'"))
    tokens_tmp = _resplit(tokens_tmp, lambda x: re.split(sep1, x),
                          lambda x: x in two_bytes_op or x.startswith("'"))
    tokens = list(filter(lambda x: x.strip() not in  ('', 'distinct', 'DISTINCT'), tokens_tmp))
    def _post_merge(tokens):
        """merge:
              * col name with "(", ")"
              * values with +/-
        """
        idx = 1
        while idx < len(tokens):
            if tokens[idx] == '(' and tokens[idx - 1] not in EXPECT_BRACKET_PRE_TOKENS and tokens[idx - 1] != '=':
                # 兼容单引号，这里可能有问题
                while idx < len(tokens):
                    tmp_tok = tokens.pop(idx)
                    tokens[idx - 1] += tmp_tok
                    if tmp_tok == ')':
                        break
            elif tokens[idx] in ('+', '-') and tokens[idx - 1] in COND_OPS and idx + 1 < len(tokens):
                tokens[idx] += tokens[idx + 1]
                tokens.pop(idx + 1)
                idx += 1
            else:
                idx += 1
        return tokens
    tokens = _post_merge(tokens)
    if single_equal:
        tokens = [i if i != '==' else '=' for i in tokens ] 

    # 加入反引号
    if use_back_quote:
        for index in range(len(tokens)):
            if contains_chinese(tokens[index]):
                tokens[index] = '`' + tokens[index] + '`'
    
    # tokens = [tok.upper() for tok in tokens]
    tokens = fix_time(tokens)
    return tokens


def generate_query_toks(data_path):
    if 'dusql' in data_path or 'debug' in data_path or 'nl2sql' in data_path or 'cspider' in data_path:
        changed_flag = False
        with open(data_path, 'r') as f:
            new_data = json.load(f)
            for unit in tqdm(new_data):
                # if 'query_toks' in unit:
                #     return
                changed_flag = True
                if 'dusql' in data_path or 'debug' in data_path or 'cspider' in data_path:
                    query_toks = tokenize_dusql(unit['query'])
                elif 'nl2sql' in data_path:
                    query_toks = tokenize_dusql(unit['query'], replace_double_quote=False)
                else:
                    query_toks = tokenize_query(unit['query'])
                unit['query_toks'] = query_toks
                unit['query_toks_no_value'] = query_toks
        # 替换原有内容
        if changed_flag:
            with open(data_path, 'w') as f:
                json.dump(new_data, f, ensure_ascii=False)

        # changed_flag = False
        # if data_path is None:
        #     real_dataset_path = dataset_path + 'dev.json'
        # with open(dataset_path + 'dev.json', 'r') as f:
        #     new_dev = json.load(f)
        #     for unit in tqdm(new_dev):
        #         # if 'query_toks' in unit:
        #         #     continue
        #         changed_flag = True
        #         tokenize_query(unit)
        # # 替换原有内容
        # if changed_flag:
        #     with open(dataset_path + 'dev.json', 'w') as f:
        #         json.dump(new_dev, f, ensure_ascii=False)