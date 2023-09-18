import os
import pint
from loader import DataLoader
from evaluation import get_program_performance

PINT = pint.UnitRegistry(system='mks', autoconvert_offset_to_baseunit=True)
PINT.load_definitions('../assets/units.txt')

if __name__ == '__main__':

    for ROOT, DIRECTORIES, FILES in os.walk(f'../assets/models/fine_tune/'):
        for FILENAME in FILES:
            if FILENAME == 'pred.json':
                TASK = ROOT.split('/')[-2]
                DATA_OBJ = DataLoader(path=f'../data/program_based/{TASK}', dataset_name='real')
                TEST_PAIRS = DATA_OBJ.get_test()

                INPUTS = [item['input'] for item in TEST_PAIRS]
                LABEL_OUTPUTS = [item['output'] for item in TEST_PAIRS]

                FILE_PATH = os.path.join(ROOT, FILENAME)
                PRED_OUTPUTS = DATA_OBJ.read(FILE_PATH)

                if TASK == 'task_3':
                    IS_TASK_3 = True
                else:
                    IS_TASK_3 = False

                PROGRAM_SCORE, VALID_SCORE, FACTS_SCORE = get_program_performance(INPUTS, LABEL_OUTPUTS,
                                                                              PRED_OUTPUTS, PINT, is_task_3=IS_TASK_3)
                RESULT = {'PROGRAM_SCORE': PROGRAM_SCORE,
                          'VALID_SCORE': VALID_SCORE, 'FACTS_SCORE': FACTS_SCORE}

                DATA_OBJ.write(f'{ROOT}/result.json', RESULT)
