# majority voting system (eaqul votes): linear SVC, LR, GloVe_LSTM
# Jinghua Xu

import random
import re
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from keras import backend as K
from keras.layers import *
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from numpy import array, asarray, zeros
from scipy.sparse import hstack
from sklearn import preprocessing, tree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.utils import class_weight
from tqdm import tqdm_notebook

from resources.dont_patronize_me import DontPatronizeMe


def get_f1(y_true, y_pred):  # taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


def labels2file(p, outf_path):
    with open(outf_path, 'w') as outf:
        for pi in p:
            outf.write(','.join([str(k) for k in pi])+'\n')


test_data_path = "./data/task4_test.tsv"
data_path = './data'
glove_path = './resources/glove.6B.100d.txt'

# load test data
tedf = pd.read_csv(test_data_path, sep='\t', header=None)
text_test = tedf[tedf.columns[4]]


# load task 1 training data
dpm = DontPatronizeMe(data_path, '')
# This method loads the subtask 1 data
dpm.load_task1()
training_set1 = dpm.train_task1_df


class Encoding:

    def __init__(self, gold_tok):
        self.gold_tok = gold_tok

    def word_ngrams(self, paragraph, min=1, max=3):
        word_vect = TfidfVectorizer(
            analyzer='word', token_pattern=r'\w{1,}', ngram_range=(min, max))
        word_vect.fit(self.gold_tok)
        word_ngrams = word_vect.transform(paragraph)
        # print(word_vect.get_feature_names())
        return word_ngrams

    def char_ngrams(self, paragraph, min=1, max=5):
        char_vector = TfidfVectorizer(analyzer='char', ngram_range=(min, max))
        char_vector.fit(self.gold_tok)
        char_ngrams = char_vector.transform(paragraph)
        # print(char_vector.get_feature_names())
        return char_ngrams


if __name__ == '__main__':

    text_train = training_set1.text
    y_train = training_set1.label

    # Encode text column
    # WORD & CHAR N-GRAMS FEATURES
    feats = Encoding(text_train)
    word_ngrams_train = feats.word_ngrams(text_train)
    word_ngrams_test = feats.word_ngrams(text_test)
    # print(word_ngrams_train.shape)
    # print(word_ngrams_test.shape)

    char_ngrams_train = feats.char_ngrams(text_train)
    char_ngrams_test = feats.char_ngrams(text_test)
    # print(char_ngrams_train.shape)
    # print(char_ngrams_test.shape)

    # stack all features
    feats_train = hstack([word_ngrams_train, char_ngrams_train])
    feats_test = hstack([word_ngrams_test, char_ngrams_test])

    print()
    print('Logistic Regression')
    print()
    lr = LogisticRegression(C=1, class_weight='balanced',
                            max_iter=10000, multi_class='ovr', solver='sag')
    lr.fit(feats_train, y_train)
    lr_preds = lr.predict(feats_test)

    print()
    print('Linear SVC')
    print()
    lsvc = LinearSVC(C=1, class_weight='balanced',
                     dual=False, max_iter=2000, penalty='l1')
    lsvc.fit(feats_train, y_train)

    # generate_model_report(lr, feats_test, y_test)
    lsvc_preds = lsvc.predict(feats_test)

    # LSTM starts here
    train_text = training_set1['text']
    train_label = training_set1['label']

    # prepare tokenizer
    t = Tokenizer()
    t.fit_on_texts(train_text)
    vocab_size = len(t.word_index) + 1

    # integer encode the documents
    encoded_docs = t.texts_to_sequences(train_text)
    encoded_test = t.texts_to_sequences(text_test)

    maxi = 0
    for doc in encoded_docs:
        # print(doc)
        if len(doc) > maxi:
            maxi = len(doc)

    for doc in encoded_test:
        # print(doc)
        if len(doc) > maxi:
            maxi = len(doc)

    # print(f'max_len = maxi = {maxi}')
    max_length = maxi

    # pad documents to a max length
    padded_docs = pad_sequences(
        encoded_docs, maxlen=max_length, padding='post')
    padded_test = pad_sequences(
        encoded_test, maxlen=max_length, padding='post')

    # load the whole embedding into memory
    embeddings_index = dict()
    f = open(glove_path)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))

    # create a weight matrix for words in training docs
    embedding_matrix = zeros((vocab_size, 100))
    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # define model
    model = Sequential()
    model.add(Input(shape=(max_length,), dtype='int32'))
    e = Embedding(vocab_size, 100, weights=[
        embedding_matrix], input_length=max_length, trainable=False)
    model.add(e)
    model.add(LSTM(60, return_sequences=True, name='lstm_layer'))
    model.add(GlobalMaxPool1D())
    model.add(Dropout(0.1))
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))

    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=[get_f1])

    # summarize the model
    print(model.summary())

    class_weights = {0: 1., 1: 10.}
    X_train, X_val, y_train, y_val = train_test_split(
        padded_docs, train_label, test_size=0.1, random_state=42, stratify=train_label)

    # fit the model
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              class_weight=class_weights, epochs=50, verbose=1, batch_size=128, shuffle=True)

    lstm_preds = model.predict(padded_test, batch_size=128, verbose=1)

    lstm_preds[lstm_preds <= 0.5] = 0
    lstm_preds[lstm_preds > 0.5] = 1

    # voting system
    majority_vote = []

    for i in range(len(lstm_preds)):
        votes = []
        votes.append(int(lstm_preds[i][0]))
        votes.append(lr_preds[i])
        votes.append(lsvc_preds[i])
        majority_vote.append(max(set(votes), key=votes.count))

    preds_path = './predictions/voting_sys/task1.txt'

    labels2file([[k] for k in majority_vote], preds_path)
