import pint
import openai
from models import zero_few_shot_experimentation
from src.evaluation import get_direct_score
from utils import create_directory, set_seed
from loader import read_data, write

PINT = pint.UnitRegistry(system='mks', autoconvert_offset_to_baseunit=True)
PINT.load_definitions('../assets/units.txt')

openai.api_key = ''  # Todo: add your openai key here

if __name__ == '__main__':
    set_seed(42)

    TASK2DATA = {
        'task_1': read_data('../data/zero_shot/task_1.json'),
        'task_2': read_data('../data/zero_shot/task_2.json'),
        'task_3': read_data('../data/zero_shot/task_3.json')
    }

    MODEL2Name = {
        "GPT3.5": 'gpt-3.5-turbo-16k-0613',
        "GPT4": 'gpt-4'
    }

    for MODEL_NAME, MODEL in MODEL2Name.items():

        for TASK_NAME, TASK_DATA in TASK2DATA.items():

            MODEL_OUTPUTS = zero_few_shot_experimentation(test=TASK_DATA,
                                                          train=None,
                                                          model_name=MODEL,
                                                          max_tokens=800)

            TASK_RESULT = 0
            for IDX, ITEM in enumerate(MODEL_OUTPUTS):
                try:
                    PRED_ITEM = eval(ITEM['openai_out_text'])
                    PRED = PRED_ITEM['answer']
                    LABEL = ITEM['output']
                    SCORE = get_direct_score(LABEL, PRED, PINT)
                    TASK_RESULT += SCORE
                    MODEL_OUTPUTS[IDX].update({'pred_answer': PRED})
                    MODEL_OUTPUTS[IDX].update({'pred_desc': PRED_ITEM['description']})
                    MODEL_OUTPUTS[IDX].update({'score': SCORE})
                    del MODEL_OUTPUTS[IDX]['openai_out_text']
                    del MODEL_OUTPUTS[IDX]['check']
                except:
                    MODEL_OUTPUTS[IDX].update({'score': 0})
                    continue
            TASK_RESULT /= len(TASK_DATA)

            create_directory(f'../assets/models/zero_shot/{MODEL_NAME}/{TASK_NAME}/')
            write(f'../assets/models/zero_shot/{MODEL_NAME}/{TASK_NAME}/pred.json', MODEL_OUTPUTS)
            write(f'../assets/models/zero_shot/{MODEL_NAME}/{TASK_NAME}/result.json',
                  {'final_result': round(TASK_RESULT, 2)})
