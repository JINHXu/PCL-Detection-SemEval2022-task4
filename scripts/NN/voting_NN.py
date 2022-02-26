# majority voting system NN
# Jinghua Xu


def file2labels(fn):
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


ann1_preds = file2labels('./predictions/ANN1/task1.txt')
ann2_preds = file2labels('./predictions/ANN2/task1.txt')
lstm1_preds = file2labels('./predictions/LSTM1/task1.txt')
lstm2_preds = file2labels('./predictions/LSTM2/task1.txt')

# voting system, equal weights
majority_vote = []

for i in range(len(ann1_preds)):
    votes = []
    votes.append(lstm1_preds[i])
    votes.append(lstm2_preds[i])
    votes.append(ann1_preds[i])
    votes.append(ann2_preds[i])
    majority_vote.append(max(set(votes), key=votes.count))

preds_path = './predictions/voting_NN/task1.txt'
labels2file([[k] for k in majority_vote], preds_path)
