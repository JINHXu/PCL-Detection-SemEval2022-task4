# Baseline Linear SVC (hstacked features)
# Diana, Jinghua

import random
import re
import warnings
import numpy as np
import pandas as pd
from numpy import array
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.utils import class_weight
from resources.dont_patronize_me import DontPatronizeMe


def labels2file(p, outf_path):
    with open(outf_path, 'w') as outf:
        for pi in p:
            outf.write(','.join([str(k) for k in pi])+'\n')


# data path
test_data_path = "./data/task4_test.tsv"
data_path = './data'

# load test data
tedf = pd.read_csv(test_data_path, sep='\t', header=None)
text_test = tedf[tedf.columns[4]]


# load task 1 training data
dpm = DontPatronizeMe(data_path, '')
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
        return word_ngrams

    def char_ngrams(self, paragraph, min=1, max=5):
        char_vector = TfidfVectorizer(analyzer='char', ngram_range=(min, max))
        char_vector.fit(self.gold_tok)
        char_ngrams = char_vector.transform(paragraph)
        return char_ngrams


if __name__ == '__main__':

    text_train = training_set1.text
    y_train = training_set1.label

    # WORD & CHAR N-GRAMS FEATURES
    feats = Encoding(text_train)
    word_ngrams_train = feats.word_ngrams(text_train)
    word_ngrams_test = feats.word_ngrams(text_test)
    char_ngrams_train = feats.char_ngrams(text_train)
    char_ngrams_test = feats.char_ngrams(text_test)

    # stack features
    feats_train = hstack([word_ngrams_train, char_ngrams_train])
    feats_test = hstack([word_ngrams_test, char_ngrams_test])

    print()
    print('Linear SVC')
    print()
    lsvc = LinearSVC(C=1, class_weight='balanced',
                     dual=False, max_iter=1500, penalty='l1')
    lsvc.fit(feats_train, y_train)
    lsvc_preds = lsvc.predict(feats_test)
    preds_path = './predictions/LSVC/task1.txt'
    labels2file([[k] for k in lsvc_preds], preds_path)
