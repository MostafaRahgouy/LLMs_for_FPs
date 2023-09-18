import os
import pytorch_lightning as pl
from utils import set_seed
from loader import DataLoader
from models import FLANT5, FLANDataModule

if __name__ == '__main__':
    set_seed(42)
    pl.seed_everything(42)

    TASK2PROMPT = {
        'task_1': 'Answer the following question by reasoning step-by-step. \n',
        'task_2': 'Answer the following question by reasoning step-by-step. \n',
        'task_3': 'Answer the following question by reasoning step-by-step. \n'
    }

    MODELS = ['google/flan-t5-base']  # add/remove other versions of t5 here

    for M in MODELS:
        for TASK_NAME, TASK_PROMPT in TASK2PROMPT.items():

            REAL_OBJ = DataLoader(path=f'../data/program_based/{TASK_NAME}/', dataset_name='real')
            TEST_PAIRS = REAL_OBJ.get_test()

            for ROOT, DIRECTORIES, FILES in os.walk(f'../assets/models/fine_tune/{M.split("/")[-1]}/{TASK_NAME}/'):
                for FILENAME in FILES:
                    if FILENAME.split('.')[-1] == 'ckpt':
                        FILE_PATH = os.path.join(ROOT, FILENAME)
                        MODEL = FLANT5.load_from_checkpoint(FILE_PATH)
                        MODEL.set_pred_path(f'{ROOT}/pred.json')
                        TOKENIZER = MODEL.get_tokenizer()
                        DATA_MODULE = FLANDataModule(train_pairs=[],
                                                     val_pairs=[],
                                                     test_pairs=TEST_PAIRS,
                                                     tokenizer=TOKENIZER,
                                                     train_bs=-1,
                                                     val_bs=-1,
                                                     test_bs=8,
                                                     prompt=TASK_PROMPT)

                        TRAINER = pl.Trainer(accelerator="gpu", devices=1, logger=False)
                        TRAINER.test(MODEL, DATA_MODULE)
