from allennlp.common.checks import ConfigurationError
from allennlp.data import DatasetReader, TokenIndexer, Field, Instance
from allennlp.data.fields import TextField, ListField, IndexField, MetadataField
from allennlp.data.fields import (
    TextField,
    ListField,
    IndexField,
    MetadataField,
    ArrayField,
)

import anytree
from anytree.search import *
from collections import defaultdict
from overrides import overrides
from time import time
from typing import Dict
from smbop.utils import moz_sql_parser as msp

import smbop.utils.node_util as node_util
import smbop.utils.hashing as hashing
import smbop.utils.ra_preproc as ra_preproc
import smbop.utils.ra_postproc as ra_postproc
from smbop.eval_final.evaluation import evaluate_single, evaluate_single_using_json
from anytree import Node, LevelOrderGroupIter
import dill
import itertools
from collections import defaultdict, OrderedDict
import json
import logging
import numpy as np
import os
from smbop.utils.replacer import Replacer
import time
from smbop.dataset_readers.enc_preproc import *
import smbop.dataset_readers.disamb_sql as disamb_sql
from smbop.utils.cache import TensorCache
from smbop.utils.generate_query_toks import tokenize_dusql, fix_date
import sqlparse
from tqdm import tqdm
import logging
import cn2an
from decimal import *
from nltk.util import ngrams
from typing import List

from rouge import Rouge

logger = logging.getLogger(__name__)

import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

#These DB take an extra hour to proccess.
LONG_DB =  ['wta_1',
 'car_1',
 'chinook_1',
 'wine_1',
 'soccer_1',
 'sakila_1',
 'baseball_1',
 'college_2',
 'flight_4',
 'store_1',
 'flight_2',
 'world_1',
 'formula_1',
 'bike_1',
 'csu_1',
 'inn_1']

special_token_dict = {
    'table': '[unused1]',
    'text': '[unused2]',
    'number': '[unused3]',
    'time': '[unused4]',
    'binary': '[unused5]',
    'boolean': '[unused6]',
    'others': '[unused7]',
    'real': '[unused8]'
}

mbart_special_token_dict = {
    'table': '#',
    'text': '##',
    'number': '!!',
    'time': '!!!',
    'binary': '!!!!',
    'boolean': '$$$',
    'others': '***',
    'real': '^^'
}

missing_vocab = [
    '《', '》', "“", "”", "邛", "崃", "缬"
]

rouge = Rouge()

@DatasetReader.register("smbop")
class SmbopSpiderDatasetReader(DatasetReader):
    def __init__(
        self,
        lazy: bool = True,
        question_token_indexers: Dict[str, TokenIndexer] = None,
        keep_if_unparsable: bool = True,
        tables_file: str = None,
        # 数据路径
        dataset_path: str = "dataset/database",
        cache_directory: str = "cache/train",
        test_cache_directory: str = None,
        include_table_name_in_column=True,
        fix_issue_16_primary_keys=False,
        qq_max_dist=2,
        cc_max_dist=2,
        tt_max_dist=2,
        max_instances=10000000,
        decoder_timesteps=9,
        limit_instances=-1,
        value_pred=True,
        use_longdb=True,
        use_extra_value=True,
        use_common_sep=False,
        use_schema_linking=True,
        use_schema_graph=True,
    ):
        super().__init__(
            # lazy=lazy,
            # cache_directory=cache_directory,
            # max_instances=max_instances,
            #  manual_distributed_sharding=True,
            # manual_multi_process_sharding=True,
        )
        global special_token_dict
        self.use_extra_value = use_extra_value
        self.cache_directory = cache_directory
        self.cache = TensorCache(cache_directory)
        self.test_cache = None
        if test_cache_directory is not None:
            self.test_cache_directory = test_cache_directory
            self.test_cache = TensorCache(test_cache_directory)
        self.value_pred = value_pred
        self._decoder_timesteps = decoder_timesteps
        self._max_instances = max_instances
        self.limit_instances = limit_instances
        self.load_less = limit_instances!=-1
        self._use_schema_linking = use_schema_linking
        self._use_schema_graph = use_schema_graph
        self._utterance_token_indexers = question_token_indexers
        self._tokenizer = self._utterance_token_indexers["tokens"]._allennlp_tokenizer
        if 'bert' in self._tokenizer.tokenizer.name_or_path:
            self._tokenizer.tokenizer.add_special_tokens({'additional_special_tokens': list(special_token_dict.values())})
            self._tokenizer.tokenizer.add_tokens(missing_vocab)

        if 'bart' in self._tokenizer.tokenizer.name_or_path:
            self.cls_token = self._tokenizer.tokenize(self._tokenizer.tokenizer.cls_token)[0]
            self.eos_token = self._tokenizer.tokenize(self._tokenizer.tokenizer.eos_token)[0]
        else:
            self.cls_token = self._tokenizer.tokenize("a")[0]
            self.eos_token = self._tokenizer.tokenize("a")[-1]
        self._keep_if_unparsable = keep_if_unparsable

        self._tables_file = tables_file
        self._dataset_path = dataset_path

        self.use_common_sep = use_common_sep
        self.replaced_token = []
        self.__SEP_TOKEN = ''
        if self.use_common_sep:
            # special_token_dict = {
            #     'table': '[SEP]',
            #     'text': '[SEP]',
            #     'number': '[SEP]',
            #     'time': '[SEP]',
            #     'binary': '[SEP]',
            #     'boolean': '[SEP]',
            #     'others': '[SEP]',
            #     'real': '[SEP]'
            # }
            self.__SEP_TOKEN = self._tokenizer.single_sequence_end_tokens[0]
            for key in special_token_dict:
                type_token = special_token_dict[key]
                self.replaced_token.append(self._tokenizer.tokenize(type_token)[1])
        # ratsql
        # 编码 + 预处理
        if not use_longdb:
            self.filter_longdb = lambda x: x not in LONG_DB
        else:
            self.filter_longdb = lambda x: True
            
        self.enc_preproc = EncPreproc(
            tables_file,
            dataset_path,
            include_table_name_in_column,
            fix_issue_16_primary_keys,
            qq_max_dist,
            cc_max_dist,
            tt_max_dist,
            use_longdb,
        )
        self._create_action_dicts()
        self.replacer = Replacer(tables_file)

    def _create_action_dicts(self):
        aliases = [
            'a',
            'b'
        ]
        if 'nl2sql' in self._dataset_path:
            unary_ops = [
                "keep",
                "min",
                "count",
                "max",
                "avg",
                "sum",
                "Table",
                "literal",
            ]
            binary_ops = [
                "eq",
                "lte",
                "lt",
                "neq",
                "gte",
                "gt",
                "And",
                "Or",
                "Val_list",
                "Project",
                "Selection",
            ]
        else:
            unary_ops = [
                "keep",
                "min",
                "count",
                "max",
                "avg",
                "sum",
                "Table",
                "Subquery",
                "distinct",
                "literal",
            ]

            # 根据aliases增加运算符
            # for alias in aliases:
            #     unary_ops.append('as_' + alias)

            # 二元运算符没有包括四则运算..
            binary_ops = [
                "eq",
                "like",
                "nlike",
                "add",
                "sub",
                "mul",
                "div",
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
                "Orderby_desc",
                "Orderby_asc",
                "Project",
                "Selection",
                "Limit",
                "Groupby",
            ]
        self.binary_op_count = len(binary_ops)
        self.unary_op_count = len(unary_ops)
        self._op_names = [
            k for k in itertools.chain(binary_ops, unary_ops, ["nan", "Value"])
        ]
        self._type_dict = OrderedDict({k: i for i, k in enumerate(self._op_names)})
        self.keep_id = self._type_dict["keep"]
        self._ACTIONS = {k: 1 for k in unary_ops}
        self._ACTIONS.update({k: 2 for k in binary_ops})
        self._ACTIONS = OrderedDict(self._ACTIONS)
        self.hasher = hashing.Hasher("cpu")

    def _init_fields(self, tree_obj):
        tree_obj = node_util.add_max_depth_att(tree_obj) # 递归为每个节点求最大深度
        tree_obj = node_util.tree2maxdepth(tree_obj) # 将每个叶子节点都pad到最大深度
        tree_obj = self.hasher.add_hash_att(tree_obj, self._type_dict)
        hash_gold_tree = tree_obj.hash
        hash_gold_levelorder = []
        for tree_list in LevelOrderGroupIter(tree_obj):
            hash_gold_levelorder.append([tree.hash for tree in tree_list])

        pad_el = hash_gold_levelorder[0]
        for i in range(self._decoder_timesteps - len(hash_gold_levelorder) + 2):
            hash_gold_levelorder.insert(0, pad_el)
        hash_gold_levelorder = hash_gold_levelorder[::-1]
        max_size = max(len(level) for level in hash_gold_levelorder)
        for level in hash_gold_levelorder:
            level.extend([-1] * (max_size - len(level)))
        hash_gold_levelorder = np.array(hash_gold_levelorder)
        return (
            hash_gold_levelorder,
            hash_gold_tree,
        )

    def process_instance(self, instance: Instance, index: int):
        return instance

    @overrides
    def _read(self, file_path: str):
        print('_read is called!!')
        if file_path.endswith(".json"):
            yield from self._read_examples_file(file_path)
        else:
            raise ConfigurationError(f"Don't know how to read filetype of {file_path}")

    def _read_examples_file(self, file_path: str):
        # cache_dir = os.path.join("cache", file_path.split("/")[-1])

        cur_cache = self.cache
        if 'test' in file_path:
            assert self.test_cache is not None
            cur_cache = self.test_cache

        cnt = 0
        cache_buffer = []
        cont_flag = True
        sent_set = set()
        for total_cnt,ins in cur_cache:
            if cnt >= self._max_instances:
                break
            if ins is not None:
                # 此处替换掉所有类型Token
                if self.use_common_sep:
                    ins.fields['enc'].tokens = [x if x not in self.replaced_token else self.__SEP_TOKEN for x in ins.fields['enc'].tokens]
                    ins.fields['enc_original'].metadata = np.array([x if x not in self.replaced_token else self.__SEP_TOKEN for x in ins.fields['enc_original'].metadata])
                if not self._use_schema_linking:
                    for _i in range(len(ins.fields['relation'].array)):
                        for _j in range(len(ins.fields['relation'].array)):
                            if ins.fields['relation'].array[_i][_j] in [15, 16, 17, 27, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]:
                                ins.fields['relation'].array[_i][_j] = 15
                if not self._use_schema_graph:
                    for _i in range(len(ins.fields['relation'].array)):
                        for _j in range(len(ins.fields['relation'].array)):
                            if ins.fields['relation'].array[_i][_j] in range(5, 37) and ins.fields['relation'].array[_i][_j] not in [15, 16, 17, 27]:
                                ins.fields['relation'].array[_i][_j] = 5
                yield ins
                cnt += 1
            sent_set.add(total_cnt)
            if self.load_less and len(sent_set) > self.limit_instances:
                cont_flag = False
                break

        if cont_flag:
            with open(file_path, "r") as data_file:
                json_obj = json.load(data_file)
                for total_cnt, ex in enumerate(json_obj):
                    if cnt >= self._max_instances:
                        break
                    if len(cache_buffer)>50:
                        cur_cache.write(cache_buffer)
                        cache_buffer = []
                    if total_cnt in sent_set:
                        continue
                    else:    
                        ins = self.create_instance(ex, file_path)
                        cache_buffer.append([total_cnt, ins])
                    if ins is not None:
                        yield ins
                        cnt +=1
            cur_cache.write(cache_buffer)


    def process_instance(self, instance: Instance, index: int):
        return instance

    def create_instance(self, ex, file_path):
        sql = None
        sql_with_values = None
        true_gold_sql = None
        if "query" in ex:
            true_gold_sql = ex['query']
        

        if "query_toks" in ex:
            try:
                # 在这一段之前不能添加反引号
                # ex = disamb_sql.fix_number_value(ex)
                if 'nl2sql' not in file_path:
                    # nl2sql数据集不要进行这一步
                    disamb_sql.sanitize_toks(ex['query_toks'])
                    disamb_sql.sanitize_toks(ex['query_toks_no_value'])
                    sql = disamb_sql.disambiguate_items(
                        ex["db_id"],
                        ex["query_toks"],
                        self._tables_file,
                        allow_aliases=False,
                    )  # FIXME: 需要在此处完成表名的规范化：将T1, T2, ...等别名替换为原本的表名
                # 不区分sql_with_values和sql
                else:
                    sql = ex['query_for_parse']
                sql = disamb_sql.sanitize(sql)
                
                # 去掉多余空格(CSpider)
                sql = sql.strip()
            except Exception as e:
                # there are two examples in the train set that are wrongly formatted, skip them
                logging.exception(e)
                print(f"error with {ex['query']}")
                print(" ".join(e.args))
                with open('error_sql.log', 'a') as f:
                    f.write(" ".join(e.args) + f" error with {ex['query']}" + "\n")
                sql = ex["query"]
                sql_with_values = ex["query"]
                sql = disamb_sql.sanitize(sql)
                # 跳过出错的实例
                return None

        # 对日期进行规范化处理
        def date_regularize(question: str) -> str:
            pattern1 = re.compile(r'\d{2,4}年\d{1,2}月\d{1,2}日')
            pattern2 = re.compile(r'\d{2,4}年\d{1,2}月\d{1,2}号')
            pattern3 = re.compile(r'\d{2,4}年\d{1,2}月\d{1,2}')
            pattern4 = re.compile(r'\d{2,4}\.\d{1,2}\.\d{1,2}')
            pattern5 = re.compile(r'\d{2,4}年\d{1,2}月')
            pattern6 = re.compile(r'\d{4}-\d{2}')
            pattern7 = re.compile(r'\d{4}\.\d{2}')
            digits_pattern = re.compile(r'\d{1,4}')
            patterns = [
                pattern1, pattern2, pattern3, pattern4, pattern5, pattern6, pattern7
            ]
            for pattern in patterns:
                dates = pattern.findall(question)
                for date in dates:
                    digits = digits_pattern.findall(date)
                    for idx in range(len(digits)):
                        if len(digits[idx]) <= 1:
                            digits[idx] = '0' + digits[idx]
                        elif idx == 0 and len(digits[idx]) == 2:
                            # 补齐4位年份
                            try:
                                if int(digits[idx]) > 50:
                                    digits[idx] = '19' + digits[idx]
                                else:
                                    digits[idx] = '20' + digits[idx]
                            except Exception as e:
                                # 吞掉错误
                                logging.debug(e)
                                continue
                    date_str = '-'.join(digits)
                    question = question.replace(date, date_str)
            return question

        # 对百分数进行规范化处理
        def percentage_regularize(question: str, dataset_path='dusql') -> str:
            pattern1 = re.compile(r'[0-9.]+%')
            # TODO: handle pattern2
            # pattern2 = re.compile(r'百分之[0-9点]+')
            percentage_list = pattern1.findall(question)
            for percentage in percentage_list:
                digits = percentage[0:-1]
                try:
                    number = Decimal(digits)
                    if 'nl2sql' not in dataset_path:
                        number = number / 100
                    if isinstance(number, float) and number.is_integer():
                        number = str(number).split('.')[0]
                    else:
                        number = str(number)
                except Exception as e:
                    logging.exception(f'error occured while handling percentage of question: {question}')
                    continue
                question = question.replace(percentage, number)

            pattern2 = re.compile(r'百分之[0-9零一二两三四五六七八九十百千万亿１２３４５６７８９０〇\.点]+')
            chinese_percentage_list = pattern2.findall(question)
            for percentage in chinese_percentage_list:
                digits = percentage[3:]
                if digits == '百':
                    number = 100
                else:
                    try:
                        number = cn2an.cn2an(digits, 'smart')
                        if isinstance(number, float) and number.is_integer():
                            number = str(number).split('.')[0]
                        else:
                            number = str(number)
                        number = Decimal(number) # 精度问题
                        if 'nl2sql' not in dataset_path:
                            number = number / 100
                    except Exception as e:
                        logging.exception(f'error occured while handling percentage of question: {question}')
                        continue
                question = question.replace(percentage, str(number))
            return question
        
        if self.use_extra_value:
            utterance = date_regularize(ex["question"])
            utterance = percentage_regularize(utterance, self._dataset_path)
        else:
            utterance = ex['question']
            
        ins = self.text_to_instance(
            utterance=utterance,
            db_id=ex["db_id"],
            sql=sql,
            sql_with_values=sql,
            true_gold_sql=true_gold_sql,
            ex=ex
        )
        return ins


    def text_to_instance(
        self, utterance: str, db_id: str, sql=None, sql_with_values=None, true_gold_sql=None, ex=None
    ):
        fields: Dict[str, Field] = {
            "db_id": MetadataField(db_id),
        }

        tokenized_utterance = self._tokenizer.tokenize(utterance) # 对问题进行分词
        has_gold = true_gold_sql != ""
        # 对SQL进行一定的处理
        # 需要在此处添加上反引号
        if has_gold:
            try:
                if 'dusql' in self._dataset_path or 'debug' in self._dataset_path:
                    # sql = fix_date(sql)
                    sql_tokens_with_backquote = tokenize_dusql(sql, use_back_quote=True)
                    sql = ' '.join(sql_tokens_with_backquote)
                elif 'cspider' in self._dataset_path:
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
                            return None
                        except Exception as e:
                            logging.exception(e)
                            with open('error_sql.log', 'a') as f:
                                f.write(f"could'nt create AST for:  {sql}\n")
                            return None
                if not set_op_flag:
                    logging.exception(e) # 有4个数据出错了，LIMIT和数字没分开，不管
                    print(f"could'nt create AST for:  {sql}") # 带中文的SQL语句无法创建AST...-> 在中文列名周围用反引号进行包裹可破
                    with open('error_sql.log', 'a') as f:
                        f.write(f"could'nt create AST for:  {sql}\n")
                    return None
            except Exception as e:
                logging.exception(e) # 有部分数据存在其它错
                print(f"could'nt create AST for:  {sql}")
                with open('error_sql.log', 'a') as f:
                    f.write(f"could'nt create AST for:  {sql}\n")
                return None
            tree_obj = ra_preproc.ast_to_ra(tree_dict["query"])
            tree_obj_values = ra_preproc.ast_to_ra(tree_dict_values["query"])

            # 中文数据集中要进行替换
            for leaf in tree_obj.leaves:
                leaf.val = self.replacer.pre(leaf.val, db_id)
                if not self.value_pred and (node_util.is_number(leaf.val) or leaf.parent.name == 'literal'):
                    # if node_util.is_number(leaf.val):
                    #     leaf.val = "'value'"
                    # else:
                    leaf.val = "value"
            # for debugging
            try:
                gold_sql = ra_postproc.ra_to_sql(tree_obj)
                # 结构一致即可
                # if 'nl2sql' in self._dataset_path:
                #     assert gold_sql.upper() == true_gold_sql.replace('==', '=').upper()
                # else:
                #     acc = evaluate_single_using_json(true_gold_sql, gold_sql, db_id, db_dir=f'{self._dataset_path}db_content.json', table_file=f'{self._dataset_path}db_schema.json', value_match=self.value_pred)
                #     if acc != 1:
                #         with open('bad_generation_rule.log', 'a') as f:
                #             err_qid = ex['question_id']
                #             f.write(f'{err_qid}: {true_gold_sql}\n')
            except Exception as e:
                logging.exception(e)

            # arit_list = anytree.search.findall(
            #     tree_obj, filter_=lambda x: x.name in ["sub", "add"]
            # )  # TODO: fixme
            # haslist_list = anytree.search.findall(
            #     tree_obj,
            #     filter_=lambda x: hasattr(x, "val") and isinstance(x.val, list),
            # )
            # if arit_list or haslist_list:
            #     print(f"could'nt create RA for:  {sql}")
            #     with open('error_sql.log', 'a') as f:
            #         f.write(f"could'nt create RA for:  {sql}\n")
            #     return None
            
            # value_pred注定为true的话，sql参数可以不传了
            # if self.value_pred:
            #     for a, b in zip(tree_obj_values.leaves, tree_obj.leaves):
            #         if b.name == "Table" or ("." in str(b.val)):
            #             continue
            #         b.val = a.val
            #         if (
            #             isinstance(a.val, int) or isinstance(a.val, float)
            #         ) and b.parent.name == "literal":
            #             # 只会复制数字值？
            #             parent_node = b.parent
            #             parent_node.children = []
            #             parent_node.name = "Value"
            #             parent_node.val = b.val

            
            try:
                # Cspider有两个case有问题: order by count(*) >= 5 ...
                leafs = list(set(node_util.get_leafs(tree_obj))) # LEAFS是从SQL语句parse出来的叶子节点
            except Exception as e:
                logging.exception(e)
                # 跳过这两个case
                return None

            hash_gold_levelorder, hash_gold_tree = self._init_fields(tree_obj)

            fields.update(
                {
                    "hash_gold_levelorder": ArrayField(
                        hash_gold_levelorder, padding_value=-1, dtype=np.int64
                    ),
                    "hash_gold_tree": ArrayField(
                        np.array(hash_gold_tree), padding_value=-1, dtype=np.int64
                    ),
                    "gold_sql": MetadataField(true_gold_sql),
                    "tree_obj": MetadataField(tree_obj),
                }
            )
        
        # desc: 一个schema处理后的结果
        desc = self.enc_preproc.get_desc(tokenized_utterance, db_id)
        if not self._use_schema_linking:
            desc['sc_link'] = {"q_col_match": {}, "q_tab_match": {}}
            desc['cv_link'] = {"num_date_match": {}, "cell_match": {}}
        entities, added_values, relation, numeral_list, added_span = self.extract_relation(desc) # 在这一步内进行表过滤，主要看relation怎么被使用
        #   TODO: relation的计算依赖SQL语句的token序列，然而有些数据集没有这个字段，直接分词分出来的token序列是不准确的。需要视情况对这一部分进行重写。
        #   FIXME: 参与构成叶子节点的DB_CONSTANT需要是中文表名，从SQL语句中分离出来的叶子节点应该转换成中文
        if 'bart' in self._tokenizer.tokenizer.name_or_path:
            question_concated = [[x] for x in tokenized_utterance[:-2]]
        else:
            question_concated = [[x] for x in tokenized_utterance[1:-1]]
        
        if self.value_pred and 'cspider' not in self._dataset_path and self.use_extra_value:
            extracted_db_value = self.extracting_db_values(db_id, tokenized_utterance[1:-1])
            extracted_db_value.extend(added_span)
        elif self.value_pred and 'cspider' in self._dataset_path:
            extracted_db_value = added_span
        else:
            extracted_db_value = []

        added_span_list = []
        for temp in extracted_db_value:
            added_span_list.extend(temp)

        # if not self.value_pred:
        #     added_values.extend(['value'])

        if not self.use_extra_value:
            added_span_list = added_span
            numeral_list = ['1', '2', '3', '4', '5']

        # 将表名替换掉
        if 'nl2sql' in self._dataset_path:
            table_name = desc['tables'][0][0]
            if table_name.lower() in entities:
                table_name_index = entities.index(table_name.lower())
                entities[table_name_index] = '_table_'

        (
            schema_tokens_pre, schema_tokens_pre_mask,
            added_span_token_list, added_span_mask_list,
            numeral_token_list, numeral_mask_list,
            added_span_start, added_span_end
        ) = self.table_text_encoding_amended(
            entities[len(added_values) + 1 :],
            added_span_list,
            numeral_list
        ) # +1 是为了去掉任意列

        schema_size = len(entities)
        # schema_tokens_pre = added_values + ["*"] + schema_tokens_pre

        # schema_tokens = [
        #     [y for y in x if y.text not in ["_"]]
        #     for x in [self._tokenizer.tokenize(x)[1:-1] for x in schema_tokens_pre]
        # ]
        if 'bart' in self._tokenizer.tokenizer.name_or_path:
            schema_tokens = [
                [y for y in x]
                for x in [self._tokenizer.tokenize(x)[:-2] for x in added_values + ['*']]
            ]
        else:
            schema_tokens = [
                [y for y in x]
                for x in [self._tokenizer.tokenize(x)[1:-1] for x in added_values + ['*']]
            ]
        schema_tokens.extend(schema_tokens_pre)

        entities_as_leafs = [x.split("::")[0] for x in entities[len(added_values) + 1 :]]
        entities_as_leafs = added_values + ["*"] + entities_as_leafs
        orig_entities = [self.replacer.post(x, db_id) for x in entities_as_leafs]
        entities_as_leafs = entities_as_leafs
        if 'nl2sql' in self._dataset_path:
            # nl2sql单表数据集特殊处理
            for index, entity in enumerate(entities_as_leafs):
                if table_name in entity and '.' in entity:
                    entities_as_leafs[index] = entity.split('.')[1]
                elif table_name.lower() == entity:
                    # 替换成统一的列名
                    entities_as_leafs[index] = '_table_'
            orig_entities = entities_as_leafs
        if 'nl2sql' in self._dataset_path:
            entities_as_leafs_hash, entities_as_leafs_types = self.hash_schema_nl2sql(
                entities_as_leafs, added_values
            )
        else:
            entities_as_leafs_hash, entities_as_leafs_types = self.hash_schema(
                entities_as_leafs, added_values
            )

        fields.update(
            {
                "relation": ArrayField(relation, padding_value=-1, dtype=np.int32),
                "entities": MetadataField(entities_as_leafs),
                 "orig_entities": MetadataField(orig_entities),
                 "leaf_hash": ArrayField(
                    entities_as_leafs_hash, padding_value=-1, dtype=np.int64
                ),
                "leaf_types": ArrayField(
                    entities_as_leafs_types,
                    padding_value=self._type_dict["nan"],
                    dtype=np.int32,
                )
            })
        # FIXME (DONE): leaf_indices 没有被计算出来。目测是Entities as leafs 需要转换成中文。
        if has_gold:
            leaf_indices, is_gold_leaf, depth = self.is_gold_leafs(
                tree_obj, leafs, schema_size, entities_as_leafs
            )  # TODO: 如果要插入锚文本需要在这个位置
            fields.update(
                {
                    "is_gold_leaf": ArrayField(
                        is_gold_leaf, padding_value=0, dtype=np.int32
                    ),
                    "leaf_indices": ArrayField(
                        leaf_indices, padding_value=-1, dtype=np.int32
                    ),
                    "depth": ArrayField(depth, padding_value=0, dtype=np.int32),
                }
            )

        enc_field_list = [] # 混合序列 [CLS] + 自然语言问题 + [SEP] + 补充值 + * + 表名 + ( + 表名.列名:类型 + , + ) + ...
        offsets = []
        mask_list = (
            [False]
            + ([True] * len(question_concated))
            + [False]
            + ([True] * len(added_values))
            + [True]
            + schema_tokens_pre_mask
            + [False]
        )
        token_list = [[self.cls_token]] + question_concated + [[self.eos_token]] + schema_tokens + [[self.eos_token]]

        added_span_offset = 0
        if len(added_span_token_list) > 0:
            mask_list = mask_list + [False] * len(added_span_mask_list) + [False]
            token_list = token_list + added_span_token_list + [[self.eos_token]]

        if len(numeral_token_list) > 0:
            mask_list = mask_list + [False] * len(numeral_mask_list) + [False]
            token_list = token_list + numeral_token_list + [[self.eos_token]]

        for mask, x in zip(mask_list, token_list):
            start_offset = len(enc_field_list)
            enc_field_list.extend(x)
            if mask:
                added_span_offset = len(enc_field_list)
                offsets.append([start_offset, len(enc_field_list) - 1]) # TODO: 理论上此处应该不需要改，但是要检查。
        
        # 取出特殊span单独处理

        utt_len = len(tokenized_utterance[1:-1])
        if self.value_pred:
            span_hash_array = self.hash_spans(tokenized_utterance)
            fields["span_hash"] = ArrayField(
                span_hash_array, padding_value=-1, dtype=np.int64
            ) # TODO: 测试空值
            added_span_hash_array = self.hash_extracted_spans(enc_field_list, added_span_start, added_span_end, added_span_offset)
            # 出于效率考虑，考虑利用mask
            fields["added_span_start"] = ArrayField(
                np.array(added_span_start), padding_value=0, dtype=np.int64
            )
            fields["added_span_end"] = ArrayField(
                np.array(added_span_end), padding_value=0, dtype=np.int64
            )
            fields["added_span_index_mask"] = ArrayField(
                np.array([True] * len(added_span_start)), padding_value=False, dtype=np.bool8
            )
            fields["added_span_hash"] = ArrayField(
                added_span_hash_array, padding_value=-1, dtype=np.int64
            )
            fields["added_span_offset"] = ArrayField(
                np.array(
                    [
                        [0, added_span_offset],
                        [added_span_offset + 1, len(enc_field_list)]
                    ]
                ),
                dtype=np.int32
            )

        if has_gold and self.value_pred:
            literal_list = node_util.get_literals(tree_obj)
            value_list = np.array(
                [self.hash_text(x) for x in node_util.get_literals(tree_obj)],
                dtype=np.int64,
            )
            is_gold_span = np.isin(span_hash_array.reshape([-1]), value_list).reshape(
                [utt_len, utt_len]
            )
            fields["gold_value_span"] = ArrayField(
                value_list, padding_value=-1, dtype=np.int64
            )
            is_gold_added_span = np.isin(added_span_hash_array, value_list)
            # 计算叶子满足率: 一个查询语句的叶子是否都能出现在备选叶子中
            # 表名和列名一般都能够满足，因此只关注值
            if len(literal_list) > 0:
                if_satisfied = [x in span_hash_array or x in added_span_hash_array for x in value_list]
                serialization_list = []
                if not all(if_satisfied):
                    serialization_list.append(ex['question_id'])
                    for _i, satisfied in enumerate(if_satisfied):
                        if not satisfied:
                            serialization_list.append(literal_list[_i])
                    with open('leaf_err_log_cspider.log', 'a', encoding='utf8') as f:
                        f.write(' '.join(serialization_list) + '\n')

            fields["is_gold_span"] = ArrayField(
                is_gold_span, padding_value=False, dtype=np.bool
            )
            fields["is_gold_added_span"] = ArrayField(
                is_gold_added_span, padding_value=False, dtype=np.bool
            )

        # lengths不会涉及到后续的值
        fields["lengths"] = ArrayField(
            np.array(
                [
                    [0, len(question_concated) - 1],
                    [len(question_concated), len(question_concated) + schema_size - 1],
                ]
            ),
            dtype=np.int32,
        )
        fields["offsets"] = ArrayField(
            np.array(offsets), padding_value=0, dtype=np.int32
        )
        fields["enc"] = TextField(enc_field_list)
        # 统计一下序列长度分布
        # with open('enc_length_stat', 'a') as ff:
        #     ff.write(str(len(enc_field_list)) + '\n')
        # 方便debug
        fields["enc_original"] = MetadataField(np.array(enc_field_list))

        ins = Instance(fields)
        return ins

    def extract_relation(self, desc):
        def parse_col(col_list):
            col_type = col_list[0]
            col_name, table = "_".join(col_list[1:]).split("_<table-sep>_")
            return f'{table}.{col_name}::{col_type.replace("<type: ","")[:-1]}'

        question_concated = [x.replace('##', '').replace('▁', '') for x in desc["question"]]
        col_concated = [parse_col(x) for x in desc["columns"]] # 在此之前要把表和列完成筛选。RAT-SQL处理过程中，columns字段需要是中文
        table_concated = ["_".join(x).lower() for x in desc["tables"]] # RAT-SQL处理过程中，tables字段需要是中文
        enc = question_concated + col_concated + table_concated # ENC: 问句 + 表名.列名:类型 + 表名 混合序列　TODO: 注意这里跟送进预训练模型的序列化细节是不一样的?
        relation = self.enc_preproc.compute_relations(
            desc,
            len(enc),
            len(question_concated),
            len(col_concated),
            range(len(col_concated) + 1),
            range(len(table_concated) + 1),
        ) # Blackbox from RAT-SQL
        unsorted_entities = col_concated + table_concated
        rel_dict = defaultdict(dict)
        # can do this with one loop
        for i, x in enumerate(list(range(len(question_concated))) + unsorted_entities):
            for j, y in enumerate(
                list(range(len(question_concated))) + unsorted_entities
            ):
                rel_dict[x][y] = relation[i, j] # rel_dict: 问句用下标表示，DB constant用名称表示，从relation矩阵中抽取出"顺序无关"的信息
        entities_sorted = sorted(list(enumerate(unsorted_entities)), key=lambda x: x[1]) 
        entities = [x[1] for x in entities_sorted] # entities在这里被定义

        def extract_date(question: str) -> list:
            # 暂时只进行年份的额外抽取
            year_pattern = re.compile(r'[零一二三四五六七八九]{0,2}[零一二三四五六七八九0-9]{2}年')
            num_dict = {
                '零': '0',
                '一': '1',
                '二': '2',
                '三': '3',
                '四': '4',
                '五': '5',
                '六': '6',
                '七': '7',
                '八': '8',
                '九': '9',
                '0': '0',
                '1': '1',
                '2': '2',
                '3': '3',
                '4': '4',
                '5': '5',
                '6': '6',
                '7': '7',
                '8': '8',
                '9': '9'
            }
            year_list = year_pattern.findall(question)
            result_list = []
            for year in year_list:
                year = year[:-1]
                new_year_list = []
                for _i, _char in enumerate(year):
                    if _char in num_dict:
                        new_year_list.append(num_dict[_char])
                year = ''.join(new_year_list)
                if len(year) == 2:
                    year = '20' + year
                if year not in result_list:
                    result_list.append(year)
                if 'nl2sql' in self._dataset_path and (year + '年') not in result_list:
                    result_list.append(year + '年')
            return result_list

        # 提取数字的规范化表示
        def extract_regularized_numeral(question: str) -> list:
            # 兼容汉字和数字混写的部分
            numeral_pattern = re.compile(r'[负0-9零一二两三四五六七八九十百千万亿１２３４５６７８９０〇\.点]+')
            chinese_numeral_pattern = re.compile(r'[负零一二两三四五六七八九十百千万亿１２３４５６７８９０〇点]')
            mixed_numeral_pattern = re.compile(r'[负零一二两三四五六七八九１２３４５６７８９０〇点0-9]')
            chinese_numeral_pattern_unit = {
                '万': 10000, '十万': 100000, '百万': 1000000, '千万': 10000000, '亿': 100000000, 
                '十亿': 1000000000, '百亿': 10000000000, '千亿': 100000000000, '万亿': 1000000000000}
            minus_numeral_patterns = ['为负', '负值', '负数', '负增长']
            unit_dict = {
                '公里': 1000,
                '平方公里': 1000000
            }
            numeral_list = numeral_pattern.findall(question)
            result_list = []
            for numeral in numeral_list:
                num = 0
                temp_list = []
                try:
                    num = cn2an.cn2an(numeral, 'smart')
                    # 判断 公里 / 平方公里 等单位
                    numeral_index = question.find(numeral)
                    for unit, value in unit_dict.items():
                        try:
                            if unit == question[numeral_index + len(numeral) : numeral_index + len(numeral) + len(unit)]:
                                new_num = num * value
                                if isinstance(new_num, float) and new_num.is_integer():
                                    new_num = str(new_num).split('.')[0]
                                else:
                                    new_num = str(new_num)
                                result_list.append(new_num)
                        except:
                            continue

                    # 判断是否为整数
                    if isinstance(num, float) and num.is_integer():
                        num = str(num).split('.')[0]
                    else:
                        if 'nl2sql' in self._dataset_path:
                            for pattern_unit, number in chinese_numeral_pattern_unit.items():
                                if pattern_unit in numeral:
                                    temp = num / number
                                    if isinstance(temp, float) and temp.is_integer():
                                        temp = str(temp).split('.')[0]
                                    else:
                                        temp = str(temp)
                                    result_list.append(temp)
                        num = str(num)
                
                    if numeral in chinese_numeral_pattern_unit and len(mixed_numeral_pattern.findall(numeral)) <= 0:
                        if str(chinese_numeral_pattern_unit[numeral]) not in result_list:
                            result_list.append(str(chinese_numeral_pattern_unit[numeral]))
                except Exception as e:
                    logging.exception(e)
                    continue

                if len(chinese_numeral_pattern.findall(numeral)) > 0:
                    result_list.append(num)
                
                for minus_pattern in minus_numeral_patterns:
                    if minus_pattern in question:
                        result_list.append('0')
            return result_list

        numeral_list = extract_regularized_numeral(''.join(question_concated))
        year_list = extract_date(''.join(question_concated))
        numeral_list.extend(year_list)
        numeral_list.extend(["1", "2", "3", "4", "5"])
        added_values = ["time_now"]
        added_spans = []
        if self.value_pred:
            added_spans = [
                "是",
                "否",
                "y",
                "t",
                "女",
                "男",
                "n",
                "null",
            ]
        else:
            added_spans = ["value"]
            added_values.append('value')
            # numeral_list = ['1']
        entities = added_values + entities
        new_enc = list(range(len(question_concated))) + entities # 为什么前面要加一串下标？-> 计算new relation
        new_relation = np.zeros([len(new_enc), len(new_enc)])
        for i, x in enumerate(new_enc):
            for j, y in enumerate(new_enc):
                if y in added_values or x in added_values:
                    continue
                new_relation[i][j] = rel_dict[x][y]
        return entities, added_values, new_relation, numeral_list, added_spans
    # FIXME (DONE): leafs中有些涉及列的部分是T1.列名，T2.列名的形式，没办法和表名.列名的形式进行匹配。需要将叶子节点的名称规范化。
    def is_gold_leafs(self, tree_obj, leafs, schema_size, entities_as_leafs):
        enitities_leaf_dict = {ent: i for i, ent in enumerate(entities_as_leafs)}
        indices = []
        for leaf in leafs:
            leaf = str(leaf)
            if leaf in enitities_leaf_dict:
                indices.append(enitities_leaf_dict[leaf])
        is_gold_leaf = np.array(
            [1 if (i in indices) else 0 for i in range(schema_size)]
        )
        indices = np.array(indices)
        depth = np.array([1] * max([leaf.depth for leaf in tree_obj.leaves]))
        return indices, is_gold_leaf, depth

    def hash_schema(self, leaf_text, added_values=None):
        beam_hash = []
        beam_types = []

        for leaf in leaf_text:
            leaf = leaf.strip()
            # TODO: fix this
            if (len(leaf.split(".")) == 2) or ("*" == leaf) or leaf in added_values:
                leaf_node = Node("Value", val=leaf)
                type_ = self._type_dict["Value"]
            else:
                leaf_node = Node("Table", val=leaf)
                type_ = self._type_dict["Table"]
            leaf_node = self.hasher.add_hash_att(leaf_node, self._type_dict)
            beam_hash.append(leaf_node.hash)
            beam_types.append(type_)
        beam_hash = np.array(beam_hash, dtype=np.int64)
        beam_types = np.array(beam_types, dtype=np.int32)
        return beam_hash, beam_types

    def hash_schema_nl2sql(self, leaf_text, added_values=None):
        beam_hash = []
        beam_types = []

        for leaf in leaf_text:
            leaf = leaf.strip()
            if leaf == '_table_':
                leaf_node = Node("Table", val=leaf)
                type_ = self._type_dict["Table"]
            else:
                leaf_node = Node("Value", val=leaf)
                type_ = self._type_dict["Value"]
            leaf_node = self.hasher.add_hash_att(leaf_node, self._type_dict)
            beam_hash.append(leaf_node.hash)
            beam_types.append(type_)
        beam_hash = np.array(beam_hash, dtype=np.int64)
        beam_types = np.array(beam_types, dtype=np.int32)
        return beam_hash, beam_types


    def hash_text(self, text):
        return self.hasher.set_hash([self._type_dict["Value"], hashing.dethash(text)])

    def hash_spans(self, tokenized_utterance):
        utt = [x.text for x in tokenized_utterance[1:-1]]
        utt_len = len(utt)
        span_hash_array = -np.ones([utt_len, utt_len], dtype=int)
        for i_ in range(utt_len):
            for j_ in range(utt_len):
                if i_ <= j_:
                    # bert系列的tokenizer对中文会自动加空格, 计算span hash基本都会错
                    token_list = []
                    for token in utt[i_ : j_ + 1]:
                        token_list.append(token.strip())
                    span_text = ''.join(token_list)
                    if 'bert' in self._tokenizer.tokenizer.name_or_path:
                        # 将tokenizer的##去掉
                        span_text = span_text.replace('##', '').replace('▁', '')
                    span_hash_array[i_, j_] = self.hash_text(span_text)
        return span_hash_array

    def hash_extracted_spans(self, enc, span_start, span_end, span_base):
        assert len(span_start) == len(span_end)
        enc_str = enc[1:]
        enc_str = [x.text for x in enc_str]
        span_hash_array = -np.ones([len(span_start)])
        for i_ in range(len(span_start)):
            token_list = []
            start_index = span_start[i_] + span_base
            end_index = span_end[i_] + span_base
            for token in enc_str[start_index: end_index + 1]:
                token_list.append(token.strip())
            span_text = ''.join(token_list)
            if 'bert' in self._tokenizer.tokenizer.name_or_path:
                span_text = span_text.replace('##', '').replace('▁', '')
            span_hash_array[i_] = self.hash_text(span_text)
        return span_hash_array
        

    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["enc"].token_indexers = self._utterance_token_indexers

    def table_text_encoding_amended(self, entity_text_list, added_span_list, numeral_list):
        # BERT Tokenizer存在保留列
        token_list = []
        mask_list = []
        # 分段添加，拼接的时候再加入[SEP]
        added_span_token_list = []
        added_span_mask_list = []
        numeral_token_list = []
        numeral_mask_list = []

        for i, curr in enumerate(entity_text_list):
            if "::" in curr:  # col
                column_name, column_type = curr.split('::')
                # 不要重复表名了
                column_name_without_table = column_name.split('.')[1]
                type_token = special_token_dict[column_type]
                if 'bart' in self._tokenizer.tokenizer.name_or_path:
                    type_token = mbart_special_token_dict[column_type]
                    curr_list = self._tokenizer.tokenize(type_token)[:-2]
                    curr_list.extend(self._tokenizer.tokenize(column_name_without_table)[:-2])
                else:
                    curr_list = self._tokenizer.tokenize(type_token)[1:-1]
                    curr_list.extend(self._tokenizer.tokenize(column_name_without_table)[1:-1])
                token_list.append(curr_list)
                mask_list.extend([True])
            else:
                type_token = special_token_dict['table']
                if 'bart' in self._tokenizer.tokenizer.name_or_path:
                    type_token = mbart_special_token_dict['table']
                    curr_list = self._tokenizer.tokenize(type_token)[:-2]
                    curr_list.extend(self._tokenizer.tokenize(curr)[:-2])
                else:
                    curr_list = self._tokenizer.tokenize(type_token)[1:-1]
                    curr_list.extend(self._tokenizer.tokenize(curr)[1:-1])
                token_list.append(curr_list)
                mask_list.extend([True])

        # 添加一个[SEP], 此后的mask全部False. 
        # token_list.append(self._tokenizer.tokenize('[SEP]')[1:-1])
        # mask_list.extend([False])

        # 保存span的开始位置和结束位置
        # 从原序列的[SEP]之后开始算下标
        added_span_start = []
        added_span_end = []
        span_base = 0

        for curr in added_span_list:
            type_token = special_token_dict['text']
            added_span_token_list.append(self._tokenizer.tokenize(type_token)[1:-1])
            added_span_mask_list.extend([True])  # TODO: 要确定类型标识符是不是需要留下, 留下更方便计算一些
            span_base += 1
            tokenized_curr = self._tokenizer.tokenize(curr)[1:-1]
            added_span_start.append(span_base)
            added_span_end.append(span_base + len(tokenized_curr) - 1)
            span_base += len(tokenized_curr)
            added_span_token_list.extend([[x] for x in tokenized_curr])
            added_span_mask_list.extend([True] * len(tokenized_curr))
        
        # if len(numeral_list) > 0:
        #     token_list.append(self._tokenizer.tokenize('[SEP]')[1:-1])
        #     mask_list.extend([False])
        if len(added_span_list) > 0:
            span_base += 1 # 跳过[SEP]

        for curr in numeral_list:
            type_token = special_token_dict['number']
            numeral_token_list.append(self._tokenizer.tokenize(type_token)[1:-1])
            span_base += 1
            numeral_mask_list.extend([True])
            tokenized_curr = self._tokenizer.tokenize(curr)[1:-1]
            added_span_start.append(span_base)
            added_span_end.append(span_base + len(tokenized_curr) - 1)
            span_base += len(tokenized_curr)
            numeral_token_list.extend([[x] for x in tokenized_curr])
            numeral_mask_list.extend([True] * len(tokenized_curr))

        return token_list, mask_list, added_span_token_list, added_span_mask_list, numeral_token_list, numeral_mask_list, added_span_start, added_span_end

    def extracting_db_values(self, db_id, tokenized_utterance, top_k=2):
        """查找DB中可能是叶子节点值的数据
            1. 按列查找
            2. 每列TOP 2
            3. 只查找text类型数据
            4. 传入的tokenized_utterance要把[CLS]和[SEP]去掉
        """

        # def _ngram_process(s: str, n: List[int]) -> List[str]:
        #     ngram = []
        #     for x in n:
        #         ngram = ngram + list(ngrams(s, x))
        #     ngram = ["".join(x) for x in ngram]
        #     # print(ngram)
        #     ngram = list(set(ngram))
        #     # print(ngram)
        #     return ngram
        # bert系列的token
        tokenized_utterance_str = [x.text.replace('##', '').replace('▁', '') for x in tokenized_utterance]

        db_content = None
        for db_data in self.enc_preproc.db_content:
            if db_data['db_id'] == db_id:
                db_content = db_data
                break
        assert db_content is not None
        extracted_value = []
        TYPE = 'type'
        if 'cspider' in self._dataset_path or 'nl2sql' in self._dataset_path:
            TYPE = 'types'
        for table in db_content['tables'].values():
            for column_id, column_type in enumerate(table[TYPE]):
                current_column = []
                if column_type != 'text':
                    continue
                for row in table['cell']:
                    try:
                        # 对某行数据取出某列
                        data_score = 0
                        column_data = row[column_id]
                        
                        # base = len(column_data) - 1
                        tokenized_column_data = [x.text.replace('##', '').replace('▁', '') for x in self._tokenizer.tokenize(column_data)[1:-1]]
                        # bigram_list = []
                        # for pos in range(len(tokenized_column_data) - 1):
                        #     bigram = tokenized_column_data[pos: pos + 2]
                        #     bigram_list.append(bigram)

                        # appear_set = set()
                        # for pos in range(len(tokenized_utterance_str)):
                        #     bigram = tokenized_utterance_str[pos: pos + 2]
                        #     if bigram not in bigram_list or ''.join(bigram) in appear_set:
                        #         continue
                        #     appear_set.add(''.join(bigram))
                        #     data_score += 1.0 / base

                        # 使用Rouge-L F1值
                        whitespace_splitted_column_data = ' '.join(tokenized_column_data)
                        whitespace_splitted_utterance = ' '.join(tokenized_utterance_str)
                        data_score = rouge.get_scores(whitespace_splitted_utterance, whitespace_splitted_column_data)[0]['rouge-l']['f']

                        # 相同的值不要重复插入
                        if data_score > 0 and (column_data, data_score) not in current_column:
                            current_column.append((column_data, data_score))
                    except Exception as e:
                        logging.exception(e)
                        continue
                if len(current_column) > 0:
                    current_column = sorted(current_column, key=lambda x: x[1], reverse=True)[:top_k]
                    extracted_value.append([x[0] for x in current_column])
        return extracted_value

                        
# 生成 表名 (表名.列名:类型) 的序列，并用token list来说明这个序列中哪些是需要编码的DB CONSTANT，哪些是用于分隔的符号 TODO: 修改这个方式, 括号和逗号有点多余
def table_text_encoding(entity_text_list):
    token_list = []
    mask_list = []
    for i, curr in enumerate(entity_text_list):
        if "::" in curr:  # col
            token_list.append(curr)
            if (i + 1) < len(entity_text_list) and "::" in entity_text_list[i + 1]:
                token_list.append(",")
            else:
                token_list.append(")\n")
            mask_list.extend([True, False])
        else:
            token_list.append(curr)
            token_list.append("(")
            mask_list.extend([True, False])

    return token_list, mask_list
