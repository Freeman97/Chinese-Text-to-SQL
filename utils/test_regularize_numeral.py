import re
import logging
import cn2an
from decimal import *

def extract_regularized_numeral(question: str) -> list:
    # 兼容汉字和数字混写的部分
    numeral_pattern = re.compile(r'[负0-9零一二两三四五六七八九十百千万亿１２３４５６７８９０〇\.点]+')
    chinese_numeral_pattern = re.compile(r'[负零一二两三四五六七八九十百千万亿１２３４５６７８９０〇点]')
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
                num = str(num)
        except Exception as e:
            logging.exception(e)
            continue

        if len(chinese_numeral_pattern.findall(numeral)) > 0:
            result_list.append(num)
    return result_list

def percentage_regularize(question: str) -> str:
    pattern1 = re.compile(r'[0-9.]+%')
    # TODO: handle pattern2
    # pattern2 = re.compile(r'百分之[0-9点]+')
    percentage_list = pattern1.findall(question)
    for percentage in percentage_list:
        digits = percentage[0:-1]
        try:
            number = Decimal(digits)
            number = number / 100
        except Exception as e:
            logging.exception(f'error occured while handling percentage of question: {question}')
            continue
        question = question.replace(percentage, str(number))
    
    pattern2 = re.compile(r'百分之[0-9零一二两三四五六七八九十百千万亿１２３４５６７８９０〇\.点]+')
    chinese_percentage_list = pattern2.findall(question)
    for percentage in chinese_percentage_list:
        digits = percentage[3:]
        try:
            number = cn2an.cn2an(digits, 'smart')
            number = Decimal(str(number)) # 精度问题
            number = number / 100
        except Exception as e:
            logging.exception(f'error occured while handling percentage of question: {question}')
            continue
        question = question.replace(percentage, str(number))
    return question


def date_regularize(question: str) -> str:
    pattern1 = re.compile(r'\d{2,4}年\d{1,2}月\d{1,2}日')
    pattern2 = re.compile(r'\d{2,4}年\d{1,2}月\d{1,2}号')
    pattern3 = re.compile(r'\d{2,4}年\d{1,2}月\d{1,2}')
    pattern4 = re.compile(r'\d{2,4}\.\d{1,2}\.\d{1,2}')
    pattern5 = re.compile(r'\d{2,4}年\d{1,2}月')
    pattern6 = re.compile(r'\d{2,4}-\d{1,2}')
    pattern7 = re.compile(r'\d{2,4}\.\d{1,2}')
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
            date_str = '-'.join(digits)
            question = question.replace(date, date_str)
    return question

if __name__ == '__main__':
    # print(percentage_regularize("背光灯寿命不少于5万小时，且市场份额不低于23%的电视机品牌，属于哪家公司，产品定位是什么"))
    # print(extract_regularized_numeral("背光灯寿命不少于一万三千五百零九亿小时，且市场份额不低于百分之4点25的电视机品牌，属于哪家公司，产品定位是什么"))
    # print(extract_regularized_numeral("这辆车行驶了15万公里, 耗油一万零五十升. 广东省的面积是337万平方公里"))
    # print(percentage_regularize("背光灯寿命不少于5万小时，且市场份额不低于4.88%的电视机品牌，属于哪家公司，产品定位是什么"))
    print(date_regularize("2021.12包含2021年12月6吗?"))