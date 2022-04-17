"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Preprocessing Spider examples released by Yu et al. 2017.
"""
import numpy as np
import src.utils.utils as utils
import scipy.sparse as ssp

from moz_sp import denormalize, extract_values
import moz_sp.sql_tokenizer as sql_tokenizer
from src.data_processor.processor_utils import get_table_aware_transformer_encoder_inputs
from src.data_processor.processor_utils import get_transformer_output_value_mask
from src.data_processor.processor_utils import get_ast
from src.data_processor.processor_utils import Text2SQLExample
from src.data_processor.processor_utils import START_TOKEN, EOS_TOKEN, NUM_TOKEN, STR_TOKEN
from src.data_processor.vocab_utils import functional_tokens
import src.data_processor.tokenizers as tok
import src.data_processor.vectorizers as vec
from src.utils.utils import SEQ2SEQ_PG, BRIDGE

import cn2an
from decimal import *
import re

import logging
RESERVED_TOKEN_TYPE = sql_tokenizer.RESERVED_TOKEN


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


def extract_date(question: str, dataset_path='dusql') -> list:
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
        if 'nl2sql' in dataset_path and (year + '年') not in result_list:
            result_list.append(year + '年')
    return result_list
# 提取数字的规范化表示
def extract_regularized_numeral(question: str, dataset_path='dusql') -> list:
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
                if 'nl2sql' in dataset_path:
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


def preprocess_example(split, example, args, parsed_programs, text_tokenize, program_tokenize,
                       post_process, trans_utils, schema_graph, vocabs, verbose=False):
    example.text = date_regularize(example.text)
    example.text = percentage_regularize(example.text, dataset_path=args.data_dir)
    number_list = extract_regularized_numeral(example.text, dataset_path=args.data_dir)
    year_list = extract_date(example.text, dataset_path=args.data_dir)
    value_list = number_list + year_list
    # 实验: 抽取数字值
    for val in value_list:
        example.text = example.text + ', ' + val

    tu = trans_utils
    text_vocab = vocabs['text']
    program_vocab = vocabs['program']

    def get_memory_values(features, raw_text, args):
        if (args.pretrained_transformer.startswith('bert-') and args.pretrained_transformer.endswith('-uncased')) or (args.pretrained_transformer.startswith('./bert-') and args.pretrained_transformer.endswith('-uncased')):
            return utils.restore_feature_case(features, raw_text, tu)
        elif 'chinese-roberta' in args.pretrained_transformer:
            return utils.restore_feature_case(features, raw_text, tu)
        elif 'bert-base-multilingual-cased' in args.pretrained_transformer:
            return utils.restore_feature_case(features, raw_text, tu)
        else:
            return features

    def get_text_schema_adjacency_matrix(text_features, s_M):
        schema_size = s_M.shape[0]
        text_size = len(text_features)
        full_size = schema_size + text_size
        M = ssp.lil_matrix((full_size, full_size), dtype=np.int)
        M[-schema_size:, -schema_size:] = s_M
        return M

    # sanity check
    ############################
    query_oov = False
    denormalized = False
    schema_truncated = False
    token_restored = True
    ############################

    # Text feature extraction and set program ground truth list
    if isinstance(example, Text2SQLExample):
        if args.pretrained_transformer:
            text_features = text_tokenize(example.text)
            text_tokens, token_starts, token_ends = get_memory_values(text_features, example.text, args)
            if not token_starts:
                token_restored = False
        else:
            text_tokens = text_tokenize(example.text, functional_tokens)
            text_features = [t.lower() for t in text_tokens]
        example.text_tokens = text_features
        example.text_ptr_values = text_tokens # 去掉## 符号之后的token -> 指针的值
        example.text_token_starts = token_starts
        example.text_token_ends = token_ends
        example.text_ids = vec.vectorize(text_features, text_vocab) # 得到ID序列
        example.text_ptr_input_ids = vec.vectorize(text_features, text_vocab)
        program_list = example.program_list
    else:
        text_tokens = example.example.text_ptr_values
        text_features = example.example.text_tokens
        program_list = example.example.program_list

    # Schema feature extraction
    if args.model_id in [BRIDGE]:
        question_encoding = example.text if args.use_picklist else None
        tables = sorted([schema_graph.get_table_id(t_name) for t_name in example.gt_table_names]) \
            if args.use_oracle_tables else None
        table_po, field_po = schema_graph.get_schema_perceived_order(tables)
        schema_features, matched_values = schema_graph.get_serialization(
            tu, flatten_features=True, table_po=table_po, field_po=field_po,
            use_typed_field_markers=args.use_typed_field_markers, use_graph_encoding=args.use_graph_encoding,
            question_encoding=question_encoding, top_k_matches=args.top_k_picklist_matches,
            match_threshold=args.anchor_text_match_threshold, num_values_per_field=args.num_values_per_field,
            no_anchor_text=args.no_anchor_text) # 此处完成问题的序列化
        example.matched_values = matched_values
        example.input_tokens, example.input_ptr_values, num_excluded_tables, num_excluded_fields = \
            get_table_aware_transformer_encoder_inputs(text_tokens, text_features, schema_features, trans_utils) # 增加控制符
        schema_truncated = (num_excluded_fields > 0)
        num_included_nodes = schema_graph.get_num_perceived_nodes(table_po) + 1 - num_excluded_tables - num_excluded_fields
        example.ptr_input_ids = vec.vectorize(example.input_tokens, text_vocab) # 序列化后的输入转换为ID
        if args.read_picklist: # Read Picklist可能需要设置为True
            example.transformer_output_value_mask, value_features, value_tokens = \
                get_transformer_output_value_mask(example.input_tokens, matched_values, tu)
        example.primary_key_ids = schema_graph.get_primary_key_ids(num_included_nodes, table_po=table_po, field_po=field_po)
        example.foreign_key_ids = schema_graph.get_foreign_key_ids(num_included_nodes, table_po=table_po, field_po=field_po)
        example.field_type_ids = schema_graph.get_field_type_ids(num_included_nodes, table_po=table_po, field_po=field_po)
        example.table_masks = schema_graph.get_table_masks(num_included_nodes, table_po=table_po, field_po=field_po)
        example.field_table_pos = schema_graph.get_field_table_pos(num_included_nodes, table_po=table_po, field_po=field_po)
        example.schema_M = schema_graph.adj_matrix
        example.M = get_text_schema_adjacency_matrix(text_features, example.schema_M)
    else:
        num_included_nodes = schema_graph.num_nodes

    # Value copy feature extraction TODO: constant_memory添加picklist及正则匹配得出的部分
    if args.read_picklist:
        constant_memory_features = text_features + value_features
        constant_memory = text_tokens + value_tokens
        example.text_ptr_values = constant_memory
    else:
        constant_memory_features = text_features
    constant_ptr_value_ids, constant_unique_input_ids = vec.vectorize_ptr_in(constant_memory_features, program_vocab)
    if isinstance(example, Text2SQLExample):
        example.text_ptr_value_ids = constant_ptr_value_ids
    example.ptr_value_ids = constant_ptr_value_ids + [program_vocab.size + len(constant_memory_features) + x
                                                      for x in range(num_included_nodes)]

    if not args.leaderboard_submission:
        error_list = []
        for j, program in enumerate(program_list):
            if isinstance(example, Text2SQLExample):
                ast, denormalized = get_ast(program, parsed_programs, args.denormalize_sql, schema_graph) # 此处将SQL去除别名，所有列名增加表名
                if ast:
                    example.program_ast_list.append(ast)
                    program_tokens = program_tokenize(ast, schema=schema_graph,  # TODO: 英文也需要escape(加引号)吗 -> 要
                                                      omit_from_clause=args.omit_from_clause,
                                                      no_join_condition=args.no_join_condition,
                                                      in_execution_order=args.process_sql_in_execution_order) # 给出按执行顺序的SQL token
                    assert(len(program_tokens) > 0)
                else:
                    program_tokens = ['from']
                program_tokens = [START_TOKEN] + program_tokens + [EOS_TOKEN]
                program_input_ids = vec.vectorize(program_tokens, program_vocab) # TODO: UNK, 确认原本的数据集此处是否出现UNK -> 有
                example.program_input_ids_list.append(program_input_ids)
                if ast:
                    example.values = extract_values(ast, schema_graph)
                else:
                    example.values = []
                # Model I. Vanilla pointer-generator output
                if args.model_id in [SEQ2SEQ_PG]:
                    program_text_ptr_value_ids = vec.vectorize_ptr_out(program_tokens, program_vocab,
                                                                       constant_unique_input_ids)
                    example.program_text_ptr_value_ids_list.append(program_text_ptr_value_ids)
                    # sanity check
                    #   NL pointer output contains tokens that does not belong to any of the following categories
                    #     - reserved tokens
                    #     - tokens in the NL input
                    #     - tokens from environment variables (e.g. table schema)
                    ############################
                    if program_vocab.unk_id in program_text_ptr_value_ids:
                        # unk_indices = [i for i, x in enumerate(program_text_ptr_value_ids) if x == program_vocab.unk_id]
                        # print('OOV I: {}'.format(' '.join([program_tokens[i] for i in unk_indices])))
                        # example.pretty_print(schema=schema_graph,
                        #                      de_vectorize_ptr=vec.de_vectorize_ptr,
                        #                      de_vectorize_field_ptr=vec.de_vectorize_field_ptr,
                        #                      rev_vocab=program_vocab,
                        #                      post_process=post_process)
                        query_oov = True
                    ############################
                # Model II. Bridge output
                if ast:
                    denormalized_ast, _ = denormalize(ast, schema_graph, return_parse_tree=True)
                    example.program_denormalized_ast_list.append(denormalized_ast)
                    tokenizer_output = program_tokenize(denormalized_ast,
                                                        return_token_types=True,
                                                        schema=schema_graph,
                                                        keep_singleton_fields=True,
                                                        omit_from_clause=args.omit_from_clause,
                                                        no_join_condition=args.no_join_condition,
                                                        atomic_value=False,
                                                        num_token=NUM_TOKEN, str_token=STR_TOKEN,
                                                        in_execution_order=args.process_sql_in_execution_order)
                    program_singleton_field_tokens, program_singleton_field_token_types = tokenizer_output[:2]
                else:
                    program_singleton_field_tokens = ['from']
                    program_singleton_field_token_types = [RESERVED_TOKEN_TYPE]
                program_singleton_field_tokens = [START_TOKEN] + program_singleton_field_tokens + [EOS_TOKEN]
                program_singleton_field_token_types = \
                    [RESERVED_TOKEN_TYPE] + program_singleton_field_token_types + [RESERVED_TOKEN_TYPE]
                example.program_singleton_field_tokens_list.append(program_singleton_field_tokens)
                example.program_singleton_field_token_types_list.append(program_singleton_field_token_types)
                program_singleton_field_input_ids = vec.vectorize_singleton(
                    program_singleton_field_tokens, program_singleton_field_token_types, program_vocab)
                example.program_singleton_field_input_ids_list.append(program_singleton_field_input_ids)
            else:
                # Model II. Bridge output
                example.program_singleton_field_input_ids_list.append(
                    example.example.program_singleton_field_input_ids_list[j])
                program_singleton_field_tokens = example.example.program_singleton_field_tokens_list[j]
                program_singleton_field_token_types = example.example.program_singleton_field_token_types_list[j]

            try:
                program_field_ptr_value_ids = vec.vectorize_field_ptr_out(program_singleton_field_tokens,
                                                                          program_singleton_field_token_types,
                                                                          program_vocab,
                                                                          constant_unique_input_ids,
                                                                          max_memory_size=len(constant_memory_features),
                                                                          schema=schema_graph,
                                                                          num_included_nodes=num_included_nodes)
            except Exception as e:
                logging.exception(e)
                program_field_ptr_value_ids = []
            example.program_text_and_field_ptr_value_ids_list.append(program_field_ptr_value_ids)
            if example.gt_table_names_list:
                table_ids = [schema_graph.get_table_id(table_name) for table_name in example.gt_table_names_list[j]]
                example.table_ids_list.append(table_ids)
                assert ([schema_graph.get_table(x).name for x in table_ids] == example.gt_table_names)
            # sanity check
            ############################
            #   NL+Schema pointer output contains tokens that does not belong to any of the following categories
            if verbose:
                if program_vocab.unk_id in program_field_ptr_value_ids:
                    unk_indices = [i for i, x in enumerate(program_field_ptr_value_ids) if x == program_vocab.unk_id]
                    print('OOV II: {}'.format(' '.join([program_singleton_field_tokens[i] for i in unk_indices])))
                    example.pretty_print(schema=schema_graph,
                                         de_vectorize_ptr=vec.de_vectorize_ptr,
                                         de_vectorize_field_ptr=vec.de_vectorize_field_ptr,
                                         rev_vocab=program_vocab,
                                         post_process=post_process,
                                         use_table_aware_te=(args.model_id in [BRIDGE]))
                    query_oov = True
            if program_vocab.unk_field_id in program_field_ptr_value_ids:
                example.pretty_print(schema=schema_graph,
                                     de_vectorize_ptr=vec.de_vectorize_ptr,
                                     de_vectorize_field_ptr=vec.de_vectorize_field_ptr,
                                     rev_vocab=program_vocab,
                                     post_process=post_process,
                                     use_table_aware_te=(args.model_id in [BRIDGE]))
            if program_vocab.unk_table_id in program_field_ptr_value_ids:
                example.pretty_print(schema=schema_graph,
                                     de_vectorize_ptr=vec.de_vectorize_ptr,
                                     de_vectorize_field_ptr=vec.de_vectorize_field_ptr,
                                     rev_vocab=program_vocab,
                                     post_process=post_process,
                                     use_table_aware_te=(args.model_id in [BRIDGE]))
            ############################
            # Store the ground truth queries after preprocessing to run a relaxed evaluation or
            # to evaluate with partial queries
            if split == 'dev':
                input_tokens = text_tokens
                if args.model_id in [BRIDGE]:
                    _p = vec.de_vectorize_field_ptr(program_field_ptr_value_ids, program_vocab, input_tokens,
                                                    schema=schema_graph, post_process=post_process)
                elif args.model_id in [SEQ2SEQ_PG]:
                    _p = vec.de_vectorize_ptr(program_text_ptr_value_ids, program_vocab, input_tokens,
                                              post_process=post_process)
                else:
                    _p = program
                example.gt_program_list.append(_p)
            # sanity check
            ############################
            # try:
            #     assert(equal_ignoring_trivial_diffs(_p, program.lower(), verbose=True))
            # except Exception:
            #     print('_p:\t\t{}'.format(_p))
            #     print('program:\t{}'.format(program))
            #     print()
            #     import pdb
            #     pdb.set_trace()
            ############################
                
        with open('error_log.log', 'a') as f:
            for ex in error_list:
                f.write(f'{ex}\n')
        example.run_unit_tests()

    return query_oov, denormalized, schema_truncated, token_restored