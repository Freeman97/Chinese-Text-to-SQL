import json

def output_postprocess(output_file, test_file, output_name='output.sql'):
    sql_list = []
    test_list = []
    with open(output_file, 'r') as f:
        sql_list = f.readlines()

    with open(test_file, 'r') as f:
        test_list = json.load(f)

    assert(len(sql_list) == len(test_list))

    output_list = []
    for sql, test_data in zip(sql_list, test_list):
        output_list.append(test_data['question_id'] + '\t' + sql.replace('\n', '') + '\t' + test_data['db_id'] + '\n')

    with open(output_name, 'w') as f:
        f.writelines(output_list)

if __name__ == '__main__':
    output_postprocess('bridge_test_nl2sql.sql', 'data/nl2sql/test.json', 'output_test_nl2sql.sql')