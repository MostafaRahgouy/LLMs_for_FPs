from loader import DataLoader
from utils import create_directory

internal_sep = ' =; '
external_sep = ' ==;; '


def get_zero_shot_data_task1(data):
    zero_shot_data = []

    for sample in data:
        facts = 'FACTS: ' + f'{internal_sep}'.join(sample['context'].split('=')[1:])
        question = 'QUESTION: ' + sample['question']
        inp = facts + external_sep + question
        pair = {'input': inp, 'output': sample['answer']}
        zero_shot_data.append(pair)
    return zero_shot_data


def get_zero_shot_data_task2(data):
    zero_shot_dta = []

    for sample in data:
        facts = 'FACTS: ' + f'{internal_sep}'.join(sample['context'].split('=')[1:])
        question = 'QUESTION: ' + sample['question']
        inp = facts + external_sep + question
        pair = {'input': inp, 'output': sample['answer']}
        zero_shot_dta.append(pair)
    return zero_shot_dta


def get_zero_shot_data_task3(data):
    zero_shot_data = []

    for sample in data:
        pair = {'input': 'QUESTION: ' + sample['question'], 'output': sample['answer']}
        zero_shot_data.append(pair)
    return zero_shot_data


if __name__ == '__main__':
    OUT_PATH = '../data/zero_shot'
    create_directory(OUT_PATH)

    TASK1_OBJ = DataLoader(path=f'../data/original', dataset_name='real')
    TASK2_OBJ = DataLoader(path=f'../data/original', dataset_name='real/distractor')

    TASK1_TEST_DATA = TASK1_OBJ.get_test()
    TASK2_TEST_DATA = TASK2_OBJ.get_test()
    TASK3_TEST_DATA = TASK1_OBJ.get_test()  # for task_3 we can use the original data without using any fact

    TASK1_ZEROSHOT_DATA = get_zero_shot_data_task1(TASK1_TEST_DATA)
    TASK1_OBJ.write(path=f'{OUT_PATH}/task_1.json', data=TASK1_ZEROSHOT_DATA)

    TASK2_ZEROSHOT_DATA = get_zero_shot_data_task2(TASK2_TEST_DATA)
    TASK2_OBJ.write(path=f'{OUT_PATH}/task_2.json', data=TASK2_ZEROSHOT_DATA)

    TASK3_ZEROSHOT_DATA = get_zero_shot_data_task3(TASK3_TEST_DATA)
    TASK1_OBJ.write(path=f'{OUT_PATH}/task_3.json', data=TASK3_ZEROSHOT_DATA)
