"""
    Authors:      Diana Hoefels, Jinghua Xu
    Task:         Ensemble Models using Logistic Regression and LinearSVC as base classifiers
"""

import pandas as pd
from scipy.sparse import hstack
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.svm import LinearSVC
from dont_patronize_me import DontPatronizeMe


def labels2file(p, output_file):
    with open(output_file, 'w') as prediction_file:
        for pi in p:
            prediction_file.write(','.join([str(k) for k in pi]) + '\n')


class Encoding:

    def __init__(self, gold_tok):
        self.gold_tok = gold_tok

    def word_ngrams(self, paragraph, min_range=1, max_range=3):
        word_vect = TfidfVectorizer(
            analyzer='word', token_pattern=r'\w{1,}', ngram_range=(min_range, max_range))
        word_vect.fit(self.gold_tok)
        word_ngrams = word_vect.transform(paragraph)
        return word_ngrams

    def char_ngrams(self, paragraph, min_range=1, max_range=5):
        char_vector = TfidfVectorizer(analyzer='char', ngram_range=(min_range, max_range))
        char_vector.fit(self.gold_tok)
        char_ngrams = char_vector.transform(paragraph)
        return char_ngrams


if __name__ == '__main__':

    # Load the subtask 1 data
    dpm = DontPatronizeMe('.', 'dont_patronize_me.py')
    dpm.load_task1()
    dataset = dpm.train_task1_df.copy()

    train_ids = pd.read_csv('train_semeval_parids-labels.csv')
    dev_ids = pd.read_csv('dev_semeval_parids-labels.csv')

    train_ids.par_id = train_ids.par_id.astype(str)
    dev_ids.par_id = dev_ids.par_id.astype(str)

    # Rebuild training set (Task 1)

    rows = []
    for idx in range(len(train_ids)):
        parid = train_ids.par_id[idx]
        text = dpm.train_task1_df.loc[dpm.train_task1_df.par_id == parid].text.values[0]
        label = dpm.train_task1_df.loc[dpm.train_task1_df.par_id == parid].label.values[0]
        rows.append({
            'par_id': parid,
            'text': text,
            'label': label
        })

    train = pd.DataFrame(rows)

    # Rebuild test set (Task 1)

    rows = []  # will contain par_id, label and text
    for idx in range(len(dev_ids)):
        parid = dev_ids.par_id[idx]
        text = dpm.train_task1_df.loc[dpm.train_task1_df.par_id == parid].text.values[0]
        label = dpm.train_task1_df.loc[dpm.train_task1_df.par_id == parid].label.values[0]
        rows.append({
            'par_id': parid,
            'text': text,
            'label': label
        })

    dev = pd.DataFrame(rows)

    X_train = train.text
    y_train = train.label
    X_test = dev.text
    y_test = dev.label

    # Encode paragraphs

    # word and char n-gram features
    feats = Encoding(X_train)
    word_ngrams_train = feats.word_ngrams(X_train)
    word_ngrams_test = feats.word_ngrams(X_test)
    char_ngrams_train = feats.char_ngrams(X_train)
    char_ngrams_test = feats.char_ngrams(X_test)

    # stack features
    feats_train = hstack([word_ngrams_train, char_ngrams_train])
    feats_test = hstack([word_ngrams_test, char_ngrams_test])

    print('Logistic Regression')
    print()

    clf_one = LogisticRegression(C=2, class_weight='balanced', multi_class='ovr', penalty='l2', solver='sag')
    clf_two = LinearSVC(C=1, class_weight='balanced', dual=False, penalty='l1')

    model = BaggingClassifier(base_estimator=clf_one, random_state=0)
    model.fit(feats_train, y_train)
    yhat = model.predict(feats_test)
    pr = precision_score(yhat, y_test)
    re = recall_score(yhat, y_test)
    f_score = f1_score(yhat, y_test)
    print('Precision: {:10.4f}'.format(pr))
    print('Recall: {:10.4f}'.format(re))
    print("F1 score: {:10.4f}".format(f_score))

    predictions_path = 'task1_bagging_log_reg.txt'
    labels2file([[k] for k in yhat], predictions_path)

    print()
    print('LinearSVC')
    print()
    lr = LogisticRegression(C=1, class_weight='balanced',
                            max_iter=2000, multi_class='ovr', solver='sag')
    model = BaggingClassifier(base_estimator=clf_two, random_state=0)
    model.fit(feats_train, y_train)
    yhat = model.predict(feats_test)
    pr = precision_score(yhat, y_test)
    re = recall_score(yhat, y_test)
    f_score = f1_score(yhat, y_test)
    print('Precision: {:10.4f}'.format(pr))
    print('Recall: {:10.4f}'.format(re))
    print("F1 score: {:10.4f}".format(f_score))

    predictions_path = '.task1_bagging_svc.txt'
    labels2file([[k] for k in yhat], predictions_path)
