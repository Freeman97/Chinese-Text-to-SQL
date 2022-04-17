import json

def extract_case(log_file, data_file, section, dataset_name):
    with open(log_file, 'r', encoding='utf8') as f:
        failed_leaves = {}
        for line in f.readlines():
            temp = line.split(' ')
            qid = temp[0]
            temp.remove(qid)
            temp[-1] = temp[-1].replace('\n', '')
            failed_leaves[qid] = temp
    
    bad_cases = []
    with open(data_file, 'r', encoding='utf8') as f:
        data_list = json.load(f)
    
    for data in data_list:
        if data['question_id'] in failed_leaves:
            data['failed_leaves'] = failed_leaves[data['question_id']]
            if '%' in data['question'] or '百分之' in data['question']:
                bad_cases.append(data)
    
    with open(f"{dataset_name}/{section}_bad_leaves_percentage.json", 'w', encoding='utf8') as f:
        json.dump(bad_cases, f, ensure_ascii=False)
    
if __name__ == '__main__':
    extract_case('dev_leaf_err_log.log', 'nl2sql/dev.json', 'dev', 'nl2sql')