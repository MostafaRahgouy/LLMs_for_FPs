import pytorch_lightning as pl
from utils import set_seed, create_directory
from loader import DataLoader
from models import FLANT5, FLANDataModule
from pytorch_lightning.loggers import CSVLogger

if __name__ == '__main__':

    # SEED
    set_seed(42)
    pl.seed_everything(42)

    TASK2PROMPT = {
        'task_1': 'Answer the following question by reasoning step-by-step. \n',
        'task_2': 'Answer the following question by reasoning step-by-step. \n',
        'task_3': 'Answer the following question by reasoning step-by-step. \n'
    }

    DATASETS2EP = {
        'real': 50,
        'synth': 5,
        'both': 5
    }

    MODELS = ['google/flan-t5-base']  # add/remove other versions of t5 here

    for M in MODELS:

        for TASK_NAME, TASK_PROMPT in TASK2PROMPT.items():

            for D, EP in DATASETS2EP.items():

                MODEL_DIR = f'../assets/models/fine_tune/{M.split("/")[-1]}/{TASK_NAME}/{D}'
                create_directory(MODEL_DIR)

                # READ DATA
                REAL_OBJ = DataLoader(path=f'../data/program_based/{TASK_NAME}', dataset_name='real')

                if D == 'both':
                    SYNTH_OBJ = DataLoader(path=f'../data/program_based/{TASK_NAME}', dataset_name='synth')
                    TRAIN_PAIRS = REAL_OBJ.get_train() + SYNTH_OBJ.get_train()
                else:
                    DATA_OBJ = DataLoader(path=f'../data/program_based/{TASK_NAME}', dataset_name=D)
                    TRAIN_PAIRS = DATA_OBJ.get_train()

                VAL_PAIRS = REAL_OBJ.get_val()
                TEST_PAIRS = REAL_OBJ.get_test()  # test and val always come from real

                # CALL PYTORCH LIGHTNING MODEL
                MODEL = FLANT5(model_name=M)
                TOKENIZER = MODEL.get_tokenizer()
                DATA_MODULE = FLANDataModule(TRAIN_PAIRS, VAL_PAIRS, TEST_PAIRS, TOKENIZER,
                                             train_bs=8, val_bs=8, test_bs=-1, prompt=TASK_PROMPT)

                CHECKPOINT_CALLBACK = pl.callbacks.ModelCheckpoint(
                    dirpath=MODEL_DIR,
                    filename=D,
                    monitor='val_loss',
                    mode='min',
                    save_top_k=1,
                )

                logger = CSVLogger(MODEL_DIR)
                # TRAIN THE MODEL
                TRAINER = pl.Trainer(accelerator='cuda', max_epochs=EP, callbacks=[CHECKPOINT_CALLBACK],
                                     log_every_n_steps=10,
                                     logger=logger)  # you can add logger here for log the metrics values
                TRAINER.fit(MODEL, DATA_MODULE)
