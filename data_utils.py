import os
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tokenizers import (ByteLevelBPETokenizer,
                            CharBPETokenizer,
                            SentencePieceBPETokenizer)

vocab = "./datasets/vocab.json"
merges = "./datasets/merges.txt"

def build_tokenizer(fnames, max_seq_len, dat_fname):
    if os.path.exists(dat_fname):
        print('loading tokenizer:', dat_fname)
        tokenizer = pickle.load(open(dat_fname, 'rb'))
    elif os.path.exists(fnames[1]+'.txt'):
        with open(fnames[1]+'.txt', encoding='utf-8') as fp:
            text = fp.read()
        tokenizer = Tokenizer(max_seq_len)
        tokenizer.fit_on_text(text)
        pickle.dump(tokenizer, open(dat_fname, 'wb'))
    else:
        text = ''
        for fname in fnames:
            print(fname)
            reader = pd.read_excel(fname)
            for i in range(reader.shape[0]):
                column_name1 = reader.iloc[i][0].lower().strip()
                text_raw1 = reader.iloc[i][2].lower().strip()
                column_2 = reader.iloc[i][1].lower().strip()
                text_raw2 = reader.iloc[i][3].lower().strip()
                text_raw = column_name1 + " " + text_raw1 + " " + column_2+ " " + text_raw2
                text += text_raw + " "

        print("Finished write txt file")
        tokenizer = Tokenizer(max_seq_len)
        tokenizer.fit_on_text(text)
        pickle.dump(tokenizer, open(dat_fname, 'wb'))
    return tokenizer


def _load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        word, vec = line.split(' ', 1)
        if word2idx is None or word in word2idx.keys():
            word_vec[word] = np.array(list(map(float, vec.split())))
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, dat_fname):
    if os.path.exists(dat_fname):
        print('loading embedding_matrix:', dat_fname)
        embedding_matrix = pickle.load(open(dat_fname, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
        fname = './glove/glove.twitter.27B.' + str(embed_dim) + 'd.txt' \
            if embed_dim != 300 else './glove/glove.840B.300d.txt'
        word_vec = _load_word_vec(fname, word2idx=word2idx)
        print('building embedding_matrix:', dat_fname)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(dat_fname, 'wb'))
    return embedding_matrix


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class Tokenizer(object):
    def __init__(self, max_seq_len, lower=True):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        words = text.split()
        tokenizer1 = SentencePieceBPETokenizer(vocab, merges)
        for word in words:
            for sub_word in tokenizer1.encode(word).tokens:
                if sub_word not in self.word2idx:
                    self.word2idx[sub_word] = self.idx
                    self.idx2word[self.idx] = sub_word
                    self.idx += 1

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        if self.lower:
            text = [x.lower() for x in text]
        words = text
        unknownidx = len(self.word2idx)+1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        if self.lower:
            text = [x.lower() for x in text]
        words = text
        unknownidx = len(self.word2idx)+1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class Dataset(Dataset):
    def __init__(self, fname, tokenizer,dat_fname):
        if os.path.exists(dat_fname):
            print('loading dataset:', dat_fname)
            self.data = pickle.load(open(dat_fname, 'rb'))

        else:
            tokenizer1 = SentencePieceBPETokenizer(vocab, merges)
            reader = pd.read_excel(fname)
            all_data = []
            for i in range(reader.shape[0]):

                text_raw1=[]
                text_raw2=[]
                column_name1 = tokenizer1.encode(reader.iloc[i][0].lower().strip()).tokens
                [text_raw1.extend(tokenizer1.encode(x).tokens) for x in reader.iloc[i][2].lower().strip().split(' ')]
                column_name2 = tokenizer1.encode(reader.iloc[i][1].lower().strip()).tokens
                [text_raw2.extend(tokenizer1.encode(x).tokens) for x in reader.iloc[i][3].lower().strip().split(' ')]
                class_n = reader.iloc[i][4]

                text_raw_indices1 = tokenizer.text_to_sequence(text_raw1)
                aspect_indices1 = tokenizer.text_to_sequence(column_name1)
                text_raw_indices2 = tokenizer.text_to_sequence(text_raw2)
                aspect_indices2 = tokenizer.text_to_sequence(column_name2)
                data = {
                    'text_raw_indices1': text_raw_indices1,
                    'aspect_indices1': aspect_indices1,
                    'text_raw_indices2': text_raw_indices2,
                    'aspect_indices2': aspect_indices2,
                    'class_n': int(class_n),
                }
                all_data.append(data)
            self.data = all_data

            pickle.dump(self.data, open(dat_fname, 'wb'))

            print("Finished write data file")




        

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
