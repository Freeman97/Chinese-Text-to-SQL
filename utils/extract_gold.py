import json

def extract_gold():
    with open('test_new.json', 'r', encoding='utf8') as f:
        data = json.load(f)
    with open('nl2sql_test_gold.sql', 'w', encoding='utf8') as f:
        for data_unit in data:
            f.write(f"{data_unit['question_id']}\t{data_unit['query']}\t{data_unit['db_id']}\n")

if __name__ == '__main__':
    extract_gold()