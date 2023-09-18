import os
import pytorch_lightning as pl
from utils import set_seed
from loader import DataLoader
from models import T5Model, DataModule

if __name__ == '__main__':
    set_seed(42)
    pl.seed_everything(42)

    TASKS = [
        'task_1',
        'task_2',
        'task_3'
    ]

    MODELS = ['t5-base']

    for M in MODELS:
        for TASK_NAME in TASKS:

            REAL_OBJ = DataLoader(path=f'../data/program_based/{TASK_NAME}/', dataset_name='real')
            TEST_PAIRS = REAL_OBJ.get_test()

            for ROOT, DIRECTORIES, FILES in os.walk(f'../assets/models/fine_tune/{M}/{TASK_NAME}'):
                for FILENAME in FILES:
                    if FILENAME.split('.')[-1] == 'ckpt':
                        FILE_PATH = os.path.join(ROOT, FILENAME)
                        MODEL = T5Model.load_from_checkpoint(FILE_PATH)
                        MODEL.set_pred_path(f'{ROOT}/pred.json')
                        TOKENIZER = MODEL.get_tokenizer()
                        DATA_MODULE = DataModule(train_pairs=[],
                                                 val_pairs=[],
                                                 test_pairs=TEST_PAIRS,
                                                 tokenizer=TOKENIZER,
                                                 train_bs=-1,
                                                 val_bs=-1,
                                                 test_bs=8
                                                 )

                        TRAINER = pl.Trainer(accelerator="gpu", devices=1, logger=False)
                        TRAINER.test(MODEL, DATA_MODULE)
