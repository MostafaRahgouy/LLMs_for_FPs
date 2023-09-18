from loader import DataLoader
from utils import create_directory

internal_sep = ' =; '
external_sep = ' ==;; '


def get_task_1_data(samples):
    pairs = []
    for sam in samples:
        facts = 'FACTS: ' + f'{internal_sep}'.join(sam['context'].split('=')[1:])
        question = 'QUESTION: ' + sam['question']
        inp = facts + external_sep + question

        program = 'PROGRAM: ' + f'{internal_sep}'.join(sam['program'].split('=')[1:])
        out = program

        pairs.append({'input': inp, 'output': out})
    return pairs


def get_task_2_data(samples):
    pairs = []
    for sam in samples:
        facts = 'FACTS: ' + f'{internal_sep}'.join(sam['context'].split('=')[1:])
        question = 'QUESTION: ' + sam['question']
        inp = facts + external_sep + question

        program = 'PROGRAM: ' + f'{internal_sep}'.join(sam['program'].split('=')[1:])
        out = program

        pairs.append({'input': inp, 'output': out})
    return pairs


def get_task_3_data(samples):
    # In task_3 facts are expected as part of output (a component of program)
    pairs = []
    for sam in samples:
        question = 'QUESTION: ' + sam['question']
        inp = question

        facts = f'{internal_sep}'.join(sam['context'].split('=')[1:])
        program = 'PROGRAM: ' + facts + internal_sep + f'{internal_sep}'.join(sam['program'].split('=')[1:])
        out = program

        pairs.append({'input': inp, 'output': out})
    return pairs


if __name__ == '__main__':

    TASK2ITEM = {'task_1': {'func': lambda x: get_task_1_data(x), 'datasets': ['real', 'synth']},
                  'task_2': {'func': lambda x: get_task_2_data(x), 'datasets': ['real/distractor', 'synth/distractor']},
                  'task_3': {'func': lambda x: get_task_3_data(x), 'datasets': ['real', 'synth']}}

    COMPONENTS = {'train': lambda x: x.get_train(),
                  'val': lambda x: x.get_val(),
                  'test': lambda x: x.get_test()}

    for TASK_NAME, TASK_ITEM in TASK2ITEM.items():

        for DATASET in TASK_ITEM['datasets']:
            DATA_OBJ = DataLoader(path='../data/original', dataset_name=DATASET)

            for COMP_NAME, DATA_COMP in COMPONENTS.items():
                DATA = DATA_COMP(DATA_OBJ)
                PAIRS = TASK_ITEM['func'](DATA)
                assert len(DATA) == len(PAIRS)
                PATH = f'../data/program_based/{TASK_NAME}/{DATASET.split("/")[0]}'
                create_directory(PATH)
                DATA_OBJ.write(path=f'{PATH}/{COMP_NAME}.json', data=PAIRS)
