import pint
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
from src.evaluation import get_direct_score
from utils import create_directory, set_seed
from loader import read_data, write

PINT = pint.UnitRegistry(system='mks', autoconvert_offset_to_baseunit=True)
PINT.load_definitions('../assets/units.txt')


def tokenize_sample(sample, tokenizer):
    prompt = 'Answer the following question as a numeric value. \n'

    inp = [prompt + sample['input'] + "\n let's think step-by-step"]

    source = tokenizer.batch_encode_plus(inp, max_length=512, padding='max_length',
                                         return_tensors='pt', truncation=True)

    input_ids, attention_mask = source['input_ids'], source['attention_mask']

    return input_ids.squeeze(1), attention_mask.squeeze(1)


def get_flant5_pred(model_name, data, device):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

    result = 0
    for idx, sample in enumerate(tqdm(data)):
        input_ids, attention_mask = tokenize_sample(sample, tokenizer)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        generated_ids = model.generate(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       max_length=512, num_beams=5, early_stopping=True).to(device)
        gen_texts = tokenizer.decode(generated_ids.squeeze(), skip_special_tokens=True,
                                     clean_up_tokenization_spaces=True)

        try:
            pred = gen_texts.split("answer is")[-1].replace('.', '').replace(':', '')
            pred = gen_texts.split("answer")[-1].replace('.', '').replace(":", '') if len(pred.split()) > 5 else pred
            score = get_direct_score(sample['output'], pred, PINT)
            data[idx].update({'pred': pred})
            data[idx].update({'desc': gen_texts})
            data[idx].update({'score': score})
            result += score
        except:
            data[idx].update({'pred': gen_texts})
            data[idx].update({'score': 0})
            continue

    result /= len(data)
    return data, result


if __name__ == '__main__':
    set_seed(42)
    TASK2DATA = {'task_1': read_data('../data/zero_shot/task_1.json'),
                 'task_2': read_data('../data/zero_shot/task_2.json'),
                 'task_3': read_data('../data/zero_shot/task_3.json')}

    MODEL_NAME = 'google/flan-t5-xl'
    DEVICE = 'cuda'

    for TASK_NAME, TASK_DATA in TASK2DATA.items():
        PREDICTIONS, TASK_RESULT = get_flant5_pred(model_name=MODEL_NAME, data=TASK_DATA, device=DEVICE)

        create_directory(f'../assets/models/zero_shot/{MODEL_NAME.split("/")[-1]}/{TASK_NAME}/')
        write(f'../assets/models/zero_shot/{MODEL_NAME.split("/")[-1]}/{TASK_NAME}/pred.json', PREDICTIONS)
        write(f'../assets/models/zero_shot/{MODEL_NAME.split("/")[-1]}/{TASK_NAME}/result.json',
              {'final_result': TASK_RESULT})
