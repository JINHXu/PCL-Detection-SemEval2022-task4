# majority voting system (eaqul votes): linear SVC, LR, GloVe_LSTM
# Jinghua Xu


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


lr_preds = file2label('./predictions/LR/task1.txt')
lsvc_preds = file2label('./predictions/LSVC/task1.txt')
lstm_preds = file2label('./predictions/GloVe_LSTM/task1.txt')
# ann_preds = file2label('./predictions/ANN/task1.txt')

# voting system, equal weights
majority_vote = []

for i in range(len(lstm_preds)):
    votes = []
    votes.append(lstm_preds[i])
    votes.append(lr_preds[i])
    votes.append(lsvc_preds[i])
    # votes.append(ann_preds[i])
    majority_vote.append(max(set(votes), key=votes.count))

preds_path = './predictions/voting4/task1.txt'
labels2file([[k] for k in majority_vote], preds_path)
