# ensemble model: lr+lstm
# Jinghua, Diana

import os 

def file2label(fn):
    with open(fn, 'r') as f:
        lines = f.readlines()
        ret = []
        for line in lines:
            ret.append(int(line.strip()))
        # print(ret)
        return ret


def labels2file(p, outf_path):
    with open(outf_path, 'w') as outf:
        for pi in p:
            outf.write(','.join([str(k) for k in pi])+'\n')


lsvc_preds = file2label('./predictions/LSVC/task1.txt')
lstm_preds = file2label('./predictions/GloVe_LSTM/task1.txt')

preds = []
zipped_preds = zip(lstm_preds, lsvc_preds)
for lstm_pred, lsvc_pred in zipped_preds:
    if lstm_pred == 1 and lsvc_pred == 1:
        preds.append(1)
    else:
        preds.append(0)

os.mkdir('./predictions/fun_algo/')
preds_path = './predictions/fun_algo/task1.txt'
labels2file([[k] for k in preds], preds_path)
