#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import logging
import math
import sys
import time
from collections import namedtuple

import mxnet as mx
import numpy as np

import data_helpers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)  # get a logger to accuracies are printed
logs = sys.stderr
CNNModel = namedtuple("CNNModel", ['cnn_exec', 'symbol', 'data', 'sent', 'label', 'param_blocks'])


class mix_cnn():
    def __init__(self):
        self.dic = {}
        self.idf = {}
        self.sent_vec = []
        self.theta = []
        self.embedded_lenth = 2

    def initTheta(self):
        '''
        随机初始化 Theta
        '''
        for i in range(len(self.dic)):
            self.theta.append(np.random.rand(self.embedded_lenth) - np.random.rand(self.embedded_lenth))

    def make_text_cnn(self, sentence_size, num_embed, batch_size, vocab_size,
                      num_label=2, filter_list={3, 4, 5}, num_filter=10,
                      dropout=0., sent_drop=0., sent_embed=10, with_embedding=True):
        sent_embed = self.embedded_lenth
        input_x = mx.sym.Variable('data')  # placeholder for input
        input_s = mx.sym.Variable('sent')  # placeholder for input
        input_y = mx.sym.Variable('softmax_label')  # placeholder for output

        # embedding layer
        if not with_embedding:
            embed_layer = mx.sym.Embedding(data=input_x, input_dim=vocab_size, output_dim=num_embed, name='vocab_embed')
            conv_input = mx.sym.Reshape(data=embed_layer, target_shape=(batch_size, 1, sentence_size, num_embed))
        else:
            conv_input = input_x

        # create convolution + (max) pooling layer for each filter operation
        pooled_outputs = []
        for i, filter_size in enumerate(filter_list):
            convi = mx.sym.Convolution(data=conv_input, kernel=(filter_size, num_embed), num_filter=num_filter)
            relui = mx.sym.Activation(data=convi, act_type='relu')
            pooli = mx.sym.Pooling(data=relui, pool_type='max', kernel=(sentence_size - filter_size + 1, 1),
                                   stride=(1, 1))
            pooled_outputs.append(pooli)

        # combine all pooled outputs
        total_filters = num_filter * len(filter_list)
        concat = mx.sym.Concat(*pooled_outputs, dim=1)
        h_pool = mx.sym.Reshape(data=concat, target_shape=(batch_size, total_filters))

        # dropout layer
        if dropout > 0.0:
            h_drop = mx.sym.Dropout(data=h_pool, p=dropout)
        else:
            h_drop = h_pool

        if sent_drop > 0.0:
            s_drop = mx.sym.Dropout(data=input_s, p=sent_drop)
        else:
            s_drop = input_s

        # fully connected
        cls_weight = mx.sym.Variable('cls_weight')
        cls_bias = mx.sym.Variable('cls_bias')
        combine = mx.sym.Concat(h_drop, s_drop, dim=1)
        fc = mx.sym.FullyConnected(data=combine, weight=cls_weight, bias=cls_bias, num_hidden=num_label)

        # softmax output
        sm = mx.sym.SoftmaxOutput(data=fc, label=input_y, name='softmax')

        return sm

    def setup_cnn_model(self, ctx, batch_size, sentence_size, num_embed, vocab_size,
                        dropout=0.5, sent_drop=0.5, initializer=mx.initializer.Uniform(0.1), with_embedding=True):
        cnn = self.make_text_cnn(sentence_size, num_embed, batch_size=batch_size,
                                 vocab_size=vocab_size, dropout=dropout, sent_drop=sent_drop, with_embedding=with_embedding)
        arg_names = cnn.list_arguments()
        sent_embed = self.embedded_lenth
        input_shapes = {}
        if with_embedding:
            input_shapes['data'] = (batch_size, 1, sentence_size, num_embed)
        else:
            input_shapes['data'] = (batch_size, sentence_size)
        input_shapes['sent'] = (batch_size, sent_embed)
        arg_shape, out_shape, aux_shape = cnn.infer_shape(**input_shapes)
        arg_arrays = [mx.nd.zeros(s, ctx) for s in arg_shape]
        args_grad = {}
        for shape, name in zip(arg_shape, arg_names):
            if name in ['softmax_label', 'data']:  # input, output
                continue
            args_grad[name] = mx.nd.zeros(shape, ctx)

        cnn_exec = cnn.bind(ctx=ctx, args=arg_arrays, args_grad=args_grad, grad_req='add')

        param_blocks = []
        arg_dict = dict(zip(arg_names, cnn_exec.arg_arrays))
        for i, name in enumerate(arg_names):
            if name in ['softmax_label', 'data', 'sent']:  # input, output
                continue
            initializer(name, arg_dict[name])
            param_blocks.append((i, arg_dict[name], args_grad[name], name))

        param_blocks.append((i + 1, arg_dict['sent'], args_grad['sent'], 'sent'))

        # out_dict = dict(zip(cnn.list_outputs(), cnn_exec.outputs))

        data = cnn_exec.arg_dict['data']
        sent = cnn_exec.arg_dict['sent']
        label = cnn_exec.arg_dict['softmax_label']

        return CNNModel(cnn_exec=cnn_exec, symbol=cnn, data=data, sent=sent, label=label, param_blocks=param_blocks)

    def train_cnn(self, model, X_train_batch, y_train_batch, X_dev_batch, y_dev_batch,  x_sent_train, x_sent_dev,
                  batch_size, optimizer='rmsprop', max_grad_norm=5.0, learning_rate=0.0005, epoch=200):
        m = model
        # create optimizer
        opt = mx.optimizer.create(optimizer)

        opt.lr = learning_rate
        updater = mx.optimizer.get_updater(opt)

        opt2 = mx.optimizer.create(optimizer)
        opt2.lr = learning_rate/5
        updater2 = mx.optimizer.get_updater(opt2)
        zishu = np.zeros(len(y_train_batch))
        for idx in range(len(y_train_batch)):
            for word in x_sent_train[idx]:
                zishu[idx] += x_sent_train[idx][word]

        for iteration in range(epoch):
            tic = time.time()
            num_correct = 0
            num_total = 0
            for begin in range(0, X_train_batch.shape[0], batch_size):
                batchX = X_train_batch[begin:begin + batch_size]
                batchY = y_train_batch[begin:begin + batch_size]
                if batchX.shape[0] != batch_size:
                    continue
                doc_S = x_sent_train[begin:begin + batch_size]
                batchS = np.zeros([batch_size, self.embedded_lenth])
                for index in range(batch_size):
                    for word in doc_S[index]:
                        batchS[index] += doc_S[index][word] * self.theta[word]

                m.data[:] = batchX
                m.label[:] = batchY
                m.sent[:] = batchS
                # forward
                m.cnn_exec.forward(is_train=True)

                # backward
                m.cnn_exec.backward()

                # eval on training data
                num_correct += sum(batchY == np.argmax(m.cnn_exec.outputs[0].asnumpy(), axis=1))
                num_total += len(batchY)

                # update weights
                norm = 0
                for idx, weight, grad, name in m.param_blocks:

                    grad /= batch_size
                    l2_norm = mx.nd.norm(grad).asscalar()
                    norm += l2_norm * l2_norm

                norm = math.sqrt(norm)
                for idx, weight, grad, name in m.param_blocks:
                    if name != 'sent':
                        if norm > max_grad_norm:
                            grad *= (max_grad_norm / norm)

                        updater(idx, grad, weight)
                    else:
                        w1 = weight.asnumpy()
                        updater(idx, grad, weight)
                        dw = weight.asnumpy() - w1
                        for doc_idx in range(batch_size):
                            for word in doc_S[doc_idx]:
                                self.theta[word] += doc_S[doc_idx][word] * dw[doc_idx] / zishu[begin + doc_idx] * 50
                    # reset gradient to zero
                    grad[:] = 0.0

            # decay learning rate
            if iteration % 50 == 0 and iteration > 0:
                opt.lr *= 0.5
                print('reset learning rate to %g' % opt.lr, file=logs)

            # end of training loop
            toc = time.time()
            train_time = toc - tic
            train_acc = num_correct * 100 / float(num_total)

            # saving checkpoint

            # if (iteration + 1) % 10 == 0:
            #     prefix = 'cnn'
            #     m.symbol.save('%s-symbol.json' % prefix)
            #     save_dict = {('arg:%s' % k): v for k, v in m.cnn_exec.arg_dict.items()}
            #     save_dict.update({('aux:%s' % k): v for k, v in m.cnn_exec.aux_dict.items()})
            #     param_name = '%s-%04d.params' % (prefix, iteration)
            #     mx.nd.save(param_name, save_dict)
            #     print('Saved checkpoint to %s' % param_name, file=logs)

            # evaluate on dev set
            num_correct = 0
            num_total = 0
            for begin in range(0, X_dev_batch.shape[0], batch_size):
                batchX = X_dev_batch[begin:begin + batch_size]
                batchY = y_dev_batch[begin:begin + batch_size]
                if batchX.shape[0] != batch_size:
                    continue
                doc_S = x_sent_dev[begin:begin + batch_size]
                batchS = np.zeros([batch_size, self.embedded_lenth])
                for index in range(batch_size):
                    for word in doc_S[index]:
                        batchS[index] += doc_S[index][word] * self.theta[word]

                m.data[:] = batchX
                m.sent[:] = batchS
                m.cnn_exec.forward(is_train=False)

                num_correct += sum(batchY == np.argmax(m.cnn_exec.outputs[0].asnumpy(), axis=1))
                num_total += len(batchY)

            dev_acc = num_correct * 100 / float(num_total)
            print('Iter [%d] Train: Time: %.3fs, Training Accuracy: %.3f \
                    --- Dev Accuracy thus far: %.3f' % (iteration, train_time, train_acc, dev_acc), file=logs)


if __name__ == '__main__':
    mix_model = mix_cnn()
    print('Loading data...')
    # word2vec = data_helpers.load_google_word2vec('/Users/guo/TrainData/google300/GoogleNews-vectors-negative300.bin')

    word2vec = data_helpers.load_pretrained_word2vec('VecForMR_.txt')
    sentences, labels = data_helpers.load_data_and_labels()
    sentences_padded = data_helpers.pad_sentences(sentences)
    x, y = data_helpers.build_input_data_with_word2vec(sentences_padded, labels, word2vec)
    mix_model.dic = data_helpers.buildGram(sentences, min1=6, min2=7)
    mix_model.initTheta()
    x_sent, mix_model.idf = data_helpers.buildDocsTFIDF(mix_model.dic, sentences)
    x_sent = np.array(x_sent)
    # randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    cv = 1
    cv_length = len(y)/10
    sample_test = shuffle_indices[cv_length * cv:cv_length * (cv + 1)]
    sample_train = np.concatenate((shuffle_indices[:cv_length * cv], shuffle_indices[cv_length * (cv + 1):]))

    # x_shuffled = x[shuffle_indices]
    # y_shuffled = y[shuffle_indices]

    # split train/dev set
    x_train, x_dev = x[sample_train], x[sample_test]
    y_train, y_dev = y[sample_train], y[sample_test]
    x_sent_train, x_sent_dev = x_sent[sample_train], x_sent[sample_test]

    print('Train/Dev split: %d/%d' % (len(y_train), len(y_dev)))
    print('train shape:', x_train.shape)
    print('dev shape:', x_dev.shape)

    # reshape for convolution input
    x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1], x_train.shape[2]))
    x_dev = np.reshape(x_dev, (x_dev.shape[0], 1, x_dev.shape[1], x_dev.shape[2]))

    num_embed = x_train.shape[-1]
    sentence_size = x_train.shape[2]
    print('sentence max words', sentence_size)
    print('embedding size', num_embed)
    batch_size = 200

    cnn_model = mix_model.setup_cnn_model(mx.cpu(1), batch_size, sentence_size,
                                          num_embed, None, dropout=0.5, sent_drop=0.)
    mix_model.train_cnn(cnn_model, x_train, y_train, x_dev, y_dev, x_sent_train, x_sent_dev, batch_size)
