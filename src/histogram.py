import os
import pint
from loader import DataLoader, read_data
from evaluation import get_program_score
import matplotlib.pyplot as plt
import numpy as np

PINT = pint.UnitRegistry(system='mks', autoconvert_offset_to_baseunit=True)
PINT.load_definitions('../assets/units.txt')


def plot_histograms(task2hop):
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.5))

    for idx, (task, value) in enumerate(task2hop.items()):
        label = value['label']
        fine = value['fine']
        few = value['few']

        all_categories = set(label.keys()) | set(fine.keys()) | set(few.keys())

        frequencies1 = [label.get(category, 0) for category in all_categories]
        frequencies2 = [fine.get(category, 0) for category in all_categories]
        frequencies3 = [few.get(category, 0) for category in all_categories]

        x = np.arange(len(all_categories))
        width = 0.2  # Width of the bars
        colors = ['#229954', '#884EA0', '#212F3C']

        axs[idx].bar(x - width, frequencies1, width, label='GROUND-TRUTH', color=colors[0])
        axs[idx].bar(x, frequencies2, width, label='FINE-TUNE', color=colors[1])
        axs[idx].bar(x + width, frequencies3, width, label='FEW-SHOT', color=colors[2])

        axs[idx].set_xlabel('Number of Hops')
        axs[idx].set_ylabel('Frequency')
        axs[idx].set_title(f'{task}')
        axs[idx].set_xticks(x, all_categories)  # Set x-axis labels to match all categories
        axs[idx].legend()

        axs[idx].set_yscale('log')
        # axs[idx].gca().yaxis.set_major_formatter(plt.ScalarFormatter())

    # Save the plot as an image file (e.g., PNG)
    plt.savefig('./histgram.pdf')
    plt.clf()


def extract_num_of_hops(program, sep=' =; '):
    program = program.replace('PROGRAM: ', '').strip()
    program = program.split(sep)
    count = 0
    for item in program:
        if item[0] == 'Q' and '->' not in item.split():
            count += 1
    return 1 if count == 0 else count


if __name__ == '__main__':
    ORG_DATA_OBJ = DataLoader(path=f'../data/original', dataset_name='real')

    ORG_TEST_DATA = ORG_DATA_OBJ.get_test()

    HOP2NUM = {}
    for SAMPLE in ORG_TEST_DATA:
        TEST_HOP = extract_num_of_hops(SAMPLE['program'], sep='=')
        if TEST_HOP not in HOP2NUM.keys():
            HOP2NUM[TEST_HOP] = 1
        else:
            HOP2NUM[TEST_HOP] += 1

    TASKS = ['task_1', 'task_2', 'task_3']
    TASKS2HOP = {}
    for TASK in TASKS:
        DATA_OBJ = DataLoader(path=f'../data/program_based/{TASK}', dataset_name='real')
        TEST_DATA = DATA_OBJ.get_test()
        FINE_TUNE_DATA = read_data(f'../assets/models/fine_tune/FLANT5-base/{TASK}/real/pred.json')
        FEW_SHOT_DATA = read_data(f'../assets/models/few_shot/GPT3.5/5_shot/{TASK}/pred.json')\

        IS_TASK_3 = True if TASK == 'task_3' else False

        FINE_HOP2NUM, FEW_HOP2NUM = {}, {}
        for LABEL, FINE, FEW in zip(TEST_DATA, FINE_TUNE_DATA, FEW_SHOT_DATA):

            LABEL_PROGRAM = LABEL['output']

            FINE_PROGRAM = FINE

            FEW_PROGRAM = FEW['pred_program']

            FINE_SCORE, FINE_VALID, _ = get_program_score(LABEL['input'], LABEL_PROGRAM, FINE_PROGRAM, PINT, is_task_3=IS_TASK_3)

            FEW_SCORE, FEW_VALID, _ = get_program_score(LABEL['input'], LABEL_PROGRAM, FEW_PROGRAM, PINT, is_task_3=IS_TASK_3)

            if FINE_VALID:
                FINE_HOP_COUNT = extract_num_of_hops(FINE_PROGRAM)
                if FINE_HOP_COUNT not in FINE_HOP2NUM.keys():
                    FINE_HOP2NUM[FINE_HOP_COUNT] = 1
                else:
                    FINE_HOP2NUM[FINE_HOP_COUNT] += 1

            if FEW_VALID:
                FEW_HOP_COUNT = extract_num_of_hops(FEW_PROGRAM)
                FEW_HOP_COUNT = 1 if FEW_HOP_COUNT == 0 else FEW_HOP_COUNT
                if FEW_HOP_COUNT not in FEW_HOP2NUM.keys():
                    FEW_HOP2NUM[FEW_HOP_COUNT] = 1
                else:
                    FEW_HOP2NUM[FEW_HOP_COUNT] += 1

        TASKS2HOP[TASK] = {'label': HOP2NUM, 'fine': FINE_HOP2NUM, 'few': FEW_HOP2NUM}
    plot_histograms(TASKS2HOP)
