from __future__ import division
from __future__ import print_function

import os
import math

import torch

import random
import math

from .vocab import Vocab
from .tree import Tree


# loading GLOVE word vectors
# if .pth file is found, will load that
# else will load from .txt file & save
def load_word_vectors(path):
    if os.path.isfile(f'{path}.pth') and os.path.isfile(f'{path}.vocab'):
        print('==> File found, loading to memory')
        vectors = torch.load(f'{path}.pth')
        vocab = Vocab(filename=f'{path}.vocab')
        return vocab, vectors
    # saved file not found, read from txt file
    # and create tensors for word vectors
    print('==> File not found, preparing, be patient')
    count = sum(
        1 for _ in open(f'{path}.txt', 'r', encoding='utf8', errors='ignore')
    )
    with open(f'{path}.txt', 'r') as f:
        contents = f.readline().rstrip('\n').split(' ')
        dim = len(contents[1:])
    words = [None] * (count)
    vectors = torch.zeros(count, dim, dtype=torch.float, device='cpu')
    with open(f'{path}.txt', 'r', encoding='utf8', errors='ignore') as f:
        for idx, line in enumerate(f):
            contents = line.rstrip('\n').split(' ')
            words[idx] = contents[0]
            values = list(map(float, contents[1:]))
            vectors[idx] = torch.tensor(values, dtype=torch.float, device='cpu')
    with open(f'{path}.vocab', 'w', encoding='utf8', errors='ignore') as f:
        for word in words:
            f.write(word + '\n')
    vocab = Vocab(filename=f'{path}.vocab')
    torch.save(vectors, f'{path}.pth')
    return vocab, vectors


# write unique words from a set of files to a new file
def build_vocab(filenames, vocabfile):
    vocab = set()
    for filename in filenames:
        with open(filename, 'r') as f:
            for line in f:
                tokens = line.rstrip('\n').split(' ')
                vocab |= set(tokens)
    with open(vocabfile, 'w') as f:
        for token in sorted(vocab):
            f.write(token + '\n')


# mapping from scalar to vector
def map_label_to_target(label, num_classes):
    target = torch.zeros(1, num_classes, dtype=torch.float, device='cpu')
    ceil = int(math.ceil(label))
    floor = int(math.floor(label))
    if ceil == floor:
        target[0, floor-1] = 1
    else:
        target[0, floor-1] = ceil - label
        target[0, ceil-1] = label - floor
    return target

def make_const(*dim, use_dtr=False):
    return torch.zeros(dim).cuda()

def t(in_dim, batch_size, use_dtr):
    return make_const(batch_size, in_dim, use_dtr=use_dtr)

def gen_tree(in_dim, batch_size, depth=6, use_dtr=False):
        return Tree(t(in_dim, batch_size, use_dtr), [] if depth == 0 
                        else [gen_tree(in_dim, batch_size, depth - 1, use_dtr), gen_tree(in_dim, batch_size, depth - 1, use_dtr)])