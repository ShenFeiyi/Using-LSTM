#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import sys
import jieba
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Embedding


def getSource(textpath, **kwarg):
    print('Get text from source...')
    items = os.listdir(textpath)
    textnames = [os.path.join(textpath,name) for name in items if name.endswith('txt')]

    article = ''
    ignore = kwarg['ignore'] if 'ignore' in kwarg else ['\u3000', '\x1a']

    for name in textnames:
        with open(name,'r',encoding='utf-8') as txt:
            content = txt.read()
            for word in content:
                if not word in ignore:
                    article += word

    token_text = list(jieba.cut(article))
    tokens = list(set(token_text))
    tokens_indices = {token: tokens.index(token) for token in tokens}
    print('Number of tokens:', len(tokens))
    
    return article, tokens, tokens_indices

def build_dataset(txt, tokens, tokens_indices, **kwarg):
    maxlen = kwarg['maxlen'] if 'maxlen' in kwarg else 75
    step = kwarg['step'] if 'step' in kwarg else 5
    
    print('Building dataset...')
    sentences = []
    next_tokens = []

    token_text = list(jieba.cut(article))
    for i in range(0, len(token_text)-maxlen, step):
        sentences.append(list(map(lambda t: tokens_indices[t], token_text[i: i+maxlen])))
        next_tokens.append(tokens_indices[token_text[i+maxlen]])
    print('Number of sequences:', len(sentences))

    next_tokens_one_hot = []
    for i in next_tokens:
        y = np.zeros((len(tokens),), dtype=np.bool)
        y[i] = 1
        next_tokens_one_hot.append(y)

    dataset = tf.data.Dataset.from_tensor_slices((sentences, next_tokens_one_hot))
    dataset = dataset.shuffle(buffer_size=4096)
    dataset = dataset.batch(128)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset

def build_model(length_of_tokens, **kwarg):
    learning_rate = kwarg['learning_rate'] if 'learning_rate' in kwarg else 1e-2
    
    print('Building model...')
    model = Sequential()
    model.add(Embedding(length_of_tokens,256))
    model.add(LSTM(256))
    model.add(Dense(length_of_tokens, activation='softmax'))

    optimizer = optimizers.RMSprop(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model

def sample(preds):
    preds = np.asarray(preds).astype('float64')
    #preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def prepare_folder(modelpath):
    if os.path.exists(modelpath):
        items = os.listdir(modelpath)
        items = [os.path.join(modelpath,item) for item in items]
        for item in items:
            os.remove(item)
    else:
        os.mkdir(modelpath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='train')
    parser.add_argument('--raw', action='store_true', help='train from scratch')
    parser.add_argument('--max', type=int, default=75, help='max length for sentences')
    parser.add_argument('--len', type=int, default=100, help='length of the generated article')
    parser.add_argument('--epochs', type=int, default=10, help='epochs, 12min/epoch, 5epochs/hour')
    args = parser.parse_args()
    params = vars(args)

    maxlen = int(params['max']) if 'max' in params else 75
    text_len = int(params['len']) if 'len' in params else 100

    textpath = os.path.join('.','Mao')
    article, tokens, tokens_indices = getSource(textpath)
    print(f'Article: {article[:5]}...')
    print(f'Type of article: {type(article)}')
    print(f'Length of article: {len(article)}')

    if params['train']:
        dataset = build_dataset(article, tokens, tokens_indices, maxlen=75)
        if os.path.exists(os.path.join('.','model','LSTM.h5')) and not params['raw']:
            model = load_model(os.path.join('.','model','LSTM.h5'))
        else:
            model = build_model(len(tokens))

        prepare_folder(os.path.join('.','model'))

        callbacks_list = [
            # 在每轮完成后保存权重
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join('.','model','LSTM.h5'),
                monitor='loss',
                save_best_only=True,
            ),
            # 不再改善时降低学习率
            keras.callbacks.ReduceLROnPlateau(
                monitor='loss',
                factor=0.5,
                patience=1,
            ),
            # 不再改善时中断训练
            keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=3,
            )]
        model.fit(dataset, epochs=params['epochs'], callbacks=callbacks_list)
    else:
        model = load_model(os.path.join('.','model','LSTM.h5'))
        start_index = np.random.randint(0, len(article)-maxlen-1)
        generated_text = article[start_index: start_index + maxlen]
        with open('make_sense.txt','w') as file:
            for i in range(text_len):
                text_cut = jieba.cut(generated_text)
                result = []
                for j in text_cut:
                    if j in tokens_indices:
                        result.append(tokens_indices[j])
                    else:
                        result.append(0)

                # 预测，采样，生成下一个 token
                preds = model.predict(result, verbose=0)[0]
                next_index = sample(preds)
                next_token = tokens[next_index]
                
                file.write(next_token)

                generated_text = generated_text[1:] + next_token
