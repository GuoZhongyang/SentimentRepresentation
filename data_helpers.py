# coding=utf-8
import itertools
from collections import Counter

import numpy as np
import math
import os
import re
from gensim.models import word2vec


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9']", " ", string)
    # string = re.sub(r"\'s", " \'s", string)
    # string = re.sub(r"\'ve", " \'ve", string)
    # string = re.sub(r"n\'t", " n\'t", string)
    # string = re.sub(r"\'re", " \'re", string)
    # string = re.sub(r"\'d", " \'d", string)
    # string = re.sub(r"\'ll", " \'ll", string)
    # string = re.sub(r",", " , ", string)
    # string = re.sub(r"!", " ! ", string)
    # string = re.sub(r"\(", " \( ", string)
    # string = re.sub(r"\)", " \) ", string)
    # string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    # return string.strip().lower()
    return string.strip()


def load_data_and_labels():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open("/Users/guo/TrainData/rt-polaritydata/rt-polarity.pos").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open("/Users/guo/TrainData/rt-polaritydata/rt-polarity.neg").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]
    # Generate labels
    positive_labels = [1 for _ in positive_examples]
    negative_labels = [0 for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def pad_sentences(sentences, padding_word="</s>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    # sequence_length = max(len(x) for x in sentences)
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def build_input_data_with_word2vec(sentences, labels, word2vec):
    """Map sentences and labels to vectors based on a pretrained word2vec"""
    x_vec = []
    MRvocab = {}  # For vocab
    for sent in sentences:
        vec = []
        for word in sent:
            if word in word2vec:
                if word not in MRvocab:  # For vocab
                    MRvocab[word] = word2vec[word]  # For vocab
                vec.append(word2vec[word])
            else:
                vec.append(word2vec['</s>'])
        x_vec.append(vec)
    # f = open("VecForMR", "w")  # For vocab
    # f.write(str(len(MRvocab)) + " 300\n")  # For vocab
    # for word in MRvocab:  # For vocab
    #     f.write(word + " " + ' '.join([str(x) for x in list(MRvocab[word])]) + "\n")  # For vocab
    x_vec = np.array(x_vec)
    y_vec = np.array(labels)
    return [x_vec, y_vec]


def load_data_with_word2vec(word2vec):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels()
    sentences_padded = pad_sentences(sentences)
    # vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    return build_input_data_with_word2vec(sentences_padded, labels, word2vec)


def buildGram(sentences, min1=5, min2=5, sw1=False, sw2=False):
    temp_dic = {}
    dic = {}
    for line in sentences:
        for token in line:
            if token not in temp_dic:
                temp_dic[token] = 1
            else:
                temp_dic[token] += 1
    temp_set = set()
    for word in temp_dic:
        if temp_dic[word] > min1:
            temp_set.add(word)
    # if sw1:
    #     self.loadStopWords()
    #     temp_set -= self.stopwords
    count = 0
    for word in temp_set:
        dic[word] = count
        count += 1
    print('unigram', len(dic))

    gram2 = {}
    # 临时变量，存储bigram的次数，用于min-count过滤
    for line in sentences:
        doc = line
        for i in range(len(doc) - 1):
            t = tuple(doc[i:i + 2])
            if t not in gram2:
                gram2[t] = 1
            else:
                gram2[t] += 1
    print ('original bigram', len(gram2))
    remove_set = set()
    for g in gram2:
        if gram2[g] <= min2:
            remove_set.add(g)
            # if sw2:
            #     if g[0] in self.stopwords and g[1] in self.stopwords:
            #         remove_set.add(g)
    for g in remove_set:
        del gram2[g]
    print ('bigram min-count -%d %d ' % (min2, len(gram2)))
    uni_count = len(dic)
    # 当前字典维度，表示有效unigram的个数
    count = uni_count
    for g in gram2:
        dic[g] = count
        count += 1
    print ('bigram', len(dic) - uni_count)
    return dic


def buildDocsTFIDF(dic, sentences):
    idf = {}
    docs = []
    docs_length1 = []
    # unigram对应的各文档有效长度
    docs_length2 = []
    # bigram对应的各文档有效长度

    for line in sentences:
        docs.append({})
        doc = line
        count1 = 0
        count2 = 0
        temp_set = set()
        for word in doc:
            if word in dic:
                idx = dic[word]
                count1 += 1
                temp_set.add(idx)

                if idx not in docs[-1]:
                    docs[-1][idx] = 1
                else:
                    docs[-1][idx] += 1

        for i in range(len(doc) - 1):
            t = tuple(doc[i:i + 2])
            if t in dic:
                count2 += 1
                idx = dic[t]
                temp_set.add(idx)

                if idx not in docs[-1]:
                    docs[-1][idx] = 1
                else:
                    docs[-1][idx] += 1

        for idx in temp_set:
            if idx not in idf:
                idf[idx] = 1
            else:
                idf[idx] += 1

        docs_length1.append(count1)
        docs_length2.append(count2)

    N = len(docs) + 0.0
    for idx in idf:
        idf[idx] = math.log(N / idf[idx])

    for i in range(len(docs)):
        doc = docs[i]
        for idx in doc:
            doc[idx] = doc[idx] * idf[idx]

    return docs, idf


def load_data():
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels()
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]


def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def load_pretrained_word2vec(infile):
    if isinstance(infile, str):
        infile = open(infile)

    word2vec = {}
    for idx, line in enumerate(infile):
        if idx == 0:
            vocab_size, dim = line.strip().split()
        else:
            tks = line.strip().split()
            word2vec[tks[0]] = map(float, tks[1:])

    return word2vec


def load_google_word2vec(path):
    model = word2vec.Word2Vec.load_word2vec_format(path, binary=True)
    return model
