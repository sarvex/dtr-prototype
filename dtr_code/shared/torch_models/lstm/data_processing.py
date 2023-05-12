'''
This source is obtained from
https://github.com/jiangqy/LSTM-Classification-pytorch/blob/master/utils/DataProcessing.py
'''
import os
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import random


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class Corpus(object):
    def __init__(self, DATA_DIR, filenames):
        self.dictionary = Dictionary()
        self.data = self.tokenize(DATA_DIR, filenames)

    def tokenize(self, DATA_DIR, filenames):
        for filename in filenames:
            path = os.path.join(DATA_DIR, filename)
            with open(path, 'r') as f:
                tokens = 0
                for line in f:
                    words = line.split() + ['<eos>']
                    tokens += len(words)
                    for word in words:
                        self.dictionary.add_word(word)

            # Tokenize file content
            with open(path, 'r') as f:
                ids = torch.LongTensor(tokens)
                token = 0
                for line in f:
                    words = line.split() + ['<eos>']
                    for word in words:
                        ids[token] = self.dictionary.word2idx[word]
                        token += 1

        return ids

class TxtDatasetProcessing(Dataset):
    def __init__(self, data_path, txt_path, txt_filename, label_filename, sen_len, corpus):
        self.txt_path = os.path.join(data_path, txt_path)
        # reading txt file from file
        txt_filepath = os.path.join(data_path, txt_filename)
        with open(txt_filepath, 'r') as fp:
            self.txt_filename = [x.strip() for x in fp]
        # reading labels from file
        label_filepath = os.path.join(data_path, label_filename)
        with open(label_filepath, 'r') as fp_label:
            labels = [int(x.strip()) for x in fp_label]
        self.label = labels
        self.corpus = corpus
        self.sen_len = sen_len


    def __getitem__(self, index):
        filename = os.path.join(self.txt_path, self.txt_filename[index])
        fp = open(filename, 'r')
        txt = torch.LongTensor(np.zeros(self.sen_len, dtype=np.int64))
        count = 0
        clip = False
        for words in fp:
            for word in words.split():
                if word.strip() in self.corpus.dictionary.word2idx:
                    if count > self.sen_len - 1:
                        clip = True
                        break
                    txt[count] = self.corpus.dictionary.word2idx[word.strip()]
                    count += 1
            if clip: break
        label = torch.LongTensor([self.label[index]])
        return txt, label
    def __len__(self):
        return len(self.txt_filename)

def random_batch(sentence_len, batch_size, testflag=True):
    tinput = [torch.LongTensor(torch.randint(sentence_len - 1, (1, batch_size))) for _ in range(sentence_len)]
    label = torch.LongTensor([[random.randint(0, 7)] for _ in (range(batch_size) if not testflag else range(sentence_len))])
    return tinput, label