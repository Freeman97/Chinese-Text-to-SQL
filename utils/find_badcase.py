import json
import re

# special case: 公里、平方公里
chinese_numeral_large = ['百', '千', '万', '十万', '百万', '千万', '亿', '十亿', '百亿', '千亿', '万亿', '公里', '平方公里', '平方千米']
chinese_numeral_single = ['一', '二', '两', '三', '四', '五', '六', '七', '八', '九', '十']
numeral_single = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
chinese_numeral = []

def _find_chinese_numeral_expression(question: str) -> bool:
    for cn in chinese_numeral:
        if cn in question:
            return True
    return False

def _find_date_expression(question: str) -> bool:
    # Y(2~4)年M(1~2)月D(1~2)日
    pattern1 = re.compile(r'\d{2,4}年\d{1,2}月\d{1,2}日')
    pattern2 = re.compile(r'\d{2,4}年\d{1,2}月\d{1,2}号')
    pattern3 = re.compile(r'\d{2,4}年\d{1,2}月\d{1,2}')
    pattern4 = re.compile(r'\d{2,4}年\d{1,2}月')
    pattern5 = re.compile(r'\d{2,4}\.\d{1,2}\.\d{1,2}')

    if any([pattern1.search(question), pattern2.search(question), pattern3.search(question), pattern4.search(question), pattern5.search(question)]):
        return True
    else:
        return False

def _find_ratio(question: str) -> bool:
    pattern = re.compile(r'\d{1,2}%')
    if pattern.search(question):
        return True
    else:
        return False

def _find_year_expression(question: str) -> bool:
    pattern5 = re.compile(r'\d{2}年\d{1,2}月')
    if pattern5.search(question):
        return True
    else:
        return False

def _write_file(file_name: str, data: list):
    with open(file_name, 'w', encoding='utf8') as f:
        for line in data:
            f.write(line)

def find_badcase(generate_debug_dataset=False):

    # initialize numeral
    for cnl in chinese_numeral_large:
        for cns in chinese_numeral_single:
                chinese_numeral.append(cns + cnl)
                # special case: 一 "个" 亿
                chinese_numeral.append(cns + '个' + cnl)
    
    for cnl in chinese_numeral_large:
        for ns in numeral_single:
            chinese_numeral.append(ns + cnl)
            chinese_numeral.append(ns + '个' + cnl)

    qid_dict = {}
    for line in open('pred_err_wo_gold_cspider.sql', 'r', encoding='utf8'):
        qid, wrong_query = line.split('\t')
        qid_dict[qid] = wrong_query

    with open('cspider/test.json', 'r', encoding='utf8') as f:
        train_data = json.load(f)
    
    date_wrong_case = []
    numeral_wrong_case = []
    year_wrong_case = []
    ratio_wrong_case = []
    other_oov_case = []
    others = []
    debug_dataset = []
    for data in train_data:
        if data['question_id'] in qid_dict:
            question = data['question']
            if _find_date_expression(question):
                date_wrong_case.append(data['question_id'] + '\t' + qid_dict[data['question_id']] + '\t\t\t' + data['query'] + '\n\t\t\t' + question + '\n')
                if generate_debug_dataset:
                    debug_dataset.append(data)
            elif _find_chinese_numeral_expression(question):
                numeral_wrong_case.append(data['question_id'] + '\t' + qid_dict[data['question_id']] + '\t\t\t' + data['query'] + '\n\t\t\t' + question + '\n')
                if generate_debug_dataset:
                    debug_dataset.append(data)
            elif _find_year_expression(question):
                year_wrong_case.append(data['question_id'] + '\t' + qid_dict[data['question_id']] + '\t\t\t' + data['query'] + '\n\t\t\t' + question + '\n')
                if generate_debug_dataset:
                    debug_dataset.append(data)
            elif _find_ratio(question):
                ratio_wrong_case.append(data['question_id'] + '\t' + qid_dict[data['question_id']] + '\t\t\t' + data['query'] + '\n\t\t\t' + question + '\n')
                if generate_debug_dataset:
                    debug_dataset.append(data)
            elif 'VALUE' in qid_dict[data['question_id']]:
                other_oov_case.append(data['question_id'] + '\t' + qid_dict[data['question_id']] + '\t\t\t' + data['query'] + '\n\t\t\t' + question + '\n')
                if generate_debug_dataset:
                    debug_dataset.append(data)
            else:
                others.append(data['question_id'] + '\t' + qid_dict[data['question_id']] + '\t\t\t' + data['query'] + '\n\t\t\t' + question + '\n')
                if generate_debug_dataset:
                    debug_dataset.append(data)

    # 写文件
    _write_file('analysis_cspider/date_wrong.sql', date_wrong_case)
    _write_file('analysis_cspider/numeral_wrong.sql', numeral_wrong_case)
    _write_file('analysis_cspider/year_wrong.sql', year_wrong_case)
    _write_file('analysis_cspider/ratio_wrong.sql', ratio_wrong_case)
    _write_file('analysis_cspider/other_oov.sql', other_oov_case)
    _write_file('analysis_cspider/others.sql', others)

    if generate_debug_dataset:
        with open('analysis_cspider/debug.json', 'w', encoding='utf8') as f:
            json.dump(debug_dataset, f, ensure_ascii=False)

if __name__ == '__main__':
    find_badcase(generate_debug_dataset=True)

