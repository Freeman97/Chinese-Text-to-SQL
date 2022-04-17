from allennlp.data import TokenIndexer
import json
from tqdm import tqdm
import re
import cn2an
import logging
from rouge import Rouge
from functools import reduce

import sys
sys.setrecursionlimit(2000)

rouge = Rouge()
top_k = 2

def rougeL_f_match_score(utt_list: list, val_list: list):
    utt_splitted = ' '.join(utt_list)
    val_splitted = ' '.join(val_list)
    return rouge.get_scores(utt_splitted, val_splitted)[0]['rouge-l']['f']

def rouge2_f_match_score(utt_list: list, val_list: list):
    utt_splitted = ' '.join(utt_list)
    val_splitted = ' '.join(val_list)
    return rouge.get_scores(utt_splitted, val_splitted)[0]['rouge-2']['f']

def rouge1_f_match_score(utt_list: list, val_list: list):
    utt_splitted = ' '.join(utt_list)
    val_splitted = ' '.join(val_list)
    return rouge.get_scores(utt_splitted, val_splitted)[0]['rouge-1']['f']

def bleu2_match_score(utt_list: list, val_list: list):
    data_score = 0
    base = len(val_list) - 1
    bigram_list = []
    for pos in range(len(val_list) - 1):
        bigram = val_list[pos: pos + 2]
        bigram_list.append(bigram)
    appear_set = set()
    for pos in range(len(utt_list)):
        bigram = utt_list[pos: pos + 2]
        if bigram not in bigram_list or ''.join(bigram) in appear_set:
            continue
        appear_set.add(''.join(bigram))
        data_score += 1.0 / base
    
    return data_score

def extract_text_value_test(data_file: str, content_file: str, dataset_name: str, method: str):
    tokenizer = TokenIndexer.by_name("pretrained_transformer")(model_name="hfl/chinese-roberta-wwm-ext-large")._allennlp_tokenizer.tokenizer
    data = []
    content = []
    with open(data_file, 'r', encoding='utf8') as f:
        data = json.load(f)
    
    with open(content_file, 'r', encoding='utf8') as f:
        content = json.load(f)
    
    extraction_result = []

    fufilled_case = 0
    total_cond_val = 0
    extracted_cond_val = 0

    for ex in tqdm(data):
        utt = ex['question']
        utt_list = tokenizer.tokenize(utt)
        db_id = ex['db_id']
        # 条件值搜索
        db_content = None
        for db_data in content:
            if db_data['db_id'] == db_id:
                db_content = db_data
                break
        assert db_content is not None
        extracted_value = []
        TYPE = 'type'
        if 'cspider' == dataset_name or 'nl2sql' == dataset_name:
            TYPE = 'types'

        # 提取所有引号中的条件值，完整出现在问题中的除外
        val_list = re.findall("'([^']*)'", ex['query'])
        val_list_doublequote = re.findall('"([^"]*)"', ex['query'])
        val_list.extend(val_list_doublequote)
        cond_val = []
        for val in val_list:
            if val not in cond_val and val not in utt:
                try:
                    temp = float(val)
                    continue
                except:
                    cond_val.append(val)

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
                        tokenized_column_data = tokenizer.tokenize(column_data)
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
                        data_score = 0
                        if method == 'rougel':
                            data_score = rougeL_f_match_score(utt_list, tokenized_column_data)
                        elif method == 'rouge2':
                            data_score = rouge2_f_match_score(utt_list, tokenized_column_data)
                        elif method == 'rouge1':
                            data_score = rouge1_f_match_score(utt_list, tokenized_column_data)
                        elif method == 'bleu2':
                            data_score = bleu2_match_score(utt_list, tokenized_column_data)
                        
                        # 相同的值不要重复插入
                        if data_score > 0 and (column_data, data_score) not in current_column:
                            current_column.append((column_data, data_score))
                    except Exception as e:
                        logging.exception(e)
                        continue
                if len(current_column) > 0:
                    current_column = sorted(current_column, key=lambda x: x[1], reverse=True)[:top_k]
                    extracted_value.append([x[0] for x in current_column])
        if len(extracted_value) > 0:
            extracted_value = reduce(lambda x, y: x + y, extracted_value)
        if len(cond_val) > 0:
            extraction_result.append({
                'question_id': ex['question_id'],
                'condition_value': cond_val,
                'extracted_value': extracted_value
            })
        fufilled = True
        total_cond_val += len(cond_val)
        for x in cond_val:
            if x in extracted_value:
                extracted_cond_val += 1
            else:
                fufilled = False
        
        fufilled_case += int(fufilled)

    cond_val_recall = extracted_cond_val / total_cond_val
    question_wise_cond_val_recall = fufilled_case / len(data)
    print('condition value recall: ' + str(cond_val_recall))
    print('question wise condition value recall: ' + str(question_wise_cond_val_recall))
    with open(f'{dataset_name}_{method}_result.json', 'w', encoding='utf8') as f:
        json.dump(extraction_result, f)

if __name__ == '__main__':
    dataset_name = 'dusql'
    data_file = f'{dataset_name}/dev.json'
    content_file = f'{dataset_name}/db_content.json'
    method = 'rouge1'
    extract_text_value_test(data_file, content_file, dataset_name, method)
