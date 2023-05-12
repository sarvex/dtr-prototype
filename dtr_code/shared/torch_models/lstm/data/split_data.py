'''
This source is obtained from
https://github.com/jiangqy/LSTM-Classification-pytorch/blob/master/data/split_data.py
'''

import os
TRAIN_FILE = 'r8-train-all-terms.txt'
TEST_FILE = 'r8-test-all-terms.txt'
TRAID_DIR = 'train_txt'
TEST_DIR = 'test_txt'

if __name__=='__main__':
    train_file = []
    with open(os.path.join(TRAIN_FILE), 'r') as fp:
        labels = {}
        train_label = []
        train_file = []
        for count, lines in enumerate(fp, start=1):
            label = lines.split()[0].strip()
            txt = lines.replace(label, '')
            if label not in labels:
                labels[label] = len(labels)
                    # writing '#count.txt' file
            filename = f'{count}.txt'
            with open(os.path.join(TRAID_DIR, filename), 'w') as fp_train:
                train_file.append(filename)
                fp_train.write(txt)
            # record #count label
            train_label.append(labels[label])
        with open('train_txt.txt', 'w') as fp_file:
            for file in train_file:
                fp_file.write(file + '\n')
        with open('train_label.txt', 'w') as fp_label:
            for t in train_label:
                fp_label.write(str(t) + '\n')
    print(labels)
    with open(os.path.join(TEST_FILE), 'r') as fp:
        test_label = []
        test_file = []
        for count, lines in enumerate(fp, start=1):
            label = lines.split()[0].strip()
            txt = lines.replace(label, '')
                    # writing '#count.txt' file
            filename = f'{count}.txt'
            with open(os.path.join(TEST_DIR, filename), 'w') as fp_test:
                test_file.append(filename)
                fp_test.write(txt)
            # record #count label
            test_label.append(labels[label])
        with open('test_txt.txt', 'w') as fp_file:
            for file in test_file:
                fp_file.write(file + '\n')
        with open('test_label.txt', 'w') as fp_label:
            for t in test_label:
                fp_label.write(str(t) + '\n')