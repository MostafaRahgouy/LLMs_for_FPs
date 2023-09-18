import pint
import openai
from models import zero_few_shot_experimentation
from utils import create_directory, set_seed
from loader import DataLoader
from evaluation import get_program_score
from loader import read_data, write

PINT = pint.UnitRegistry(system='mks', autoconvert_offset_to_baseunit=True)
PINT.load_definitions('../assets/units.txt')

# Set up your OpenAI API credentials
openai.api_key = ''  # Todo: add you openai key here

if __name__ == '__main__':
    set_seed(42)

    MODEL2Name = {
        "GPT3.5": 'gpt-3.5-turbo-16k-0613',
        # "GPT4": 'gpt-4'
    }

    TASKS = [
        'task_1',
        'task_2',
        'task_3'
    ]

    SHOT2NAME = {
        '1_shot': 1,
        '3_shot': 3,
        '5_shot': 5
    }

    for MODEL_NAME, MODEL in MODEL2Name.items():
        for TASK in TASKS:
            for SHOT_NAME, SHOT in SHOT2NAME.items():

                IS_TASK_3 = True if TASK == 'task_3' else False

                DATA_OBJ = DataLoader(path=f'../data/program_based/{TASK}', dataset_name='real')
                TEST_DATA = DATA_OBJ.get_test()
                TRAIN_DATA = DATA_OBJ.get_train()

                MODEL_OUTPUTS = zero_few_shot_experimentation(test=TEST_DATA,
                                                              train=TRAIN_DATA,
                                                              model_type="fewshot-openai",
                                                              few_shot_no=SHOT,
                                                              model_name=MODEL,
                                                              max_tokens=1000)

                PROGRAM_SCORE, VALID_SCORE, FACTS_SCORE = 0, 0, 0
                for IDX, ITEM in enumerate(MODEL_OUTPUTS):
                    try:
                        PRED_PROGRAM = ITEM['openai_out_text']
                        # PRED_PROGRAM = eval(ITEM['openai_out_text'])['program']
                        LABEL_PROGRAM = ITEM['output']

                        P_SCORE, V_SCORE, F_SCORE = get_program_score(ITEM['input'], LABEL_PROGRAM, PRED_PROGRAM, PINT,
                                                                      is_task_3=IS_TASK_3)

                        PROGRAM_SCORE += P_SCORE
                        VALID_SCORE += V_SCORE
                        FACTS_SCORE += F_SCORE

                        MODEL_OUTPUTS[IDX]['pred_program'] = PRED_PROGRAM
                        MODEL_OUTPUTS[IDX].update({'score': P_SCORE})

                        del MODEL_OUTPUTS[IDX]['openai_out_text']
                        del MODEL_OUTPUTS[IDX]['check']

                    except:
                        MODEL_OUTPUTS[IDX].update({'score': 0})
                        continue

                PROGRAM_SCORE /= len(TEST_DATA)
                FACTS_SCORE /= VALID_SCORE if VALID_SCORE != 0 else 1
                VALID_SCORE /= len(TEST_DATA)

                create_directory(f'../assets/models/few_shot/{MODEL_NAME}/{SHOT_NAME}/{TASK}/')
                write(f'../assets/models/few_shot/{MODEL_NAME}/{SHOT_NAME}/{TASK}/pred.json', MODEL_OUTPUTS)
                write(f'../assets/models/few_shot/{MODEL_NAME}/{SHOT_NAME}/{TASK}/result.json',
                      {'program_score': round(PROGRAM_SCORE, 4),
                       'valid': round(VALID_SCORE, 4),
                       'facts_score': round(FACTS_SCORE, 4)})
