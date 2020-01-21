from sklearn.metrics import jaccard_similarity_score, roc_auc_score, precision_score, recall_score, f1_score, average_precision_score
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import warnings
import dill
from collections import Counter
import os
import os.path as osp
import logging
import copy
from tqdm import tqdm
from itertools import combinations
from collections import defaultdict, OrderedDict

warnings.filterwarnings('ignore')
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# --- hyperparameter ---


class KeyObt:
    KEY_ID = 'patient_id'
    KEY_DX = 'dx'
    KEY_RX = 'rx'
    KEY_DX_CODE = 'diag_cd'
    KEY_RX_CODE = 'ndc'
    KEY_LABEL = 'cohort'
    KEY_DATE = 'svc_dt'
    KEY_DAYS = 'days_supply_cnt'


class Voc(object):
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}

    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)


# rel_dict = {
#     '<': 0, '>': 1,  # before
#     '=': 2,  # equal
#     'm': 3, 'mi': 4,  # meet
#     'o': 5, 'oi': 6,  # overlap
#     'c': 7, 'ci': 8,  # contain
#     's': 9, 'si': 10,  # start
#     'f': 11, 'fi': 12,  # finish
# }

rel_ls = [
    'unk',  # unknown
    '=',  # equal
    'm',  # meet
    'o',  # overlap
    'c',  # contain
    's',  # start
    'f',  # finish
]
rel_voc = Voc()
rel_voc.add_sentence(rel_ls)


def get_edge_type(a, b):
    # equal
    if a[1] == b[1] and a[2] == b[2]:
        return rel_voc.word2idx['=']
    # meet
    if a[2] == b[1]:
        return rel_voc.word2idx['m']
    if a[1] == b[2]:
        return rel_voc.word2idx['m']
    # overlap
    if a[2] > b[1] and a[2] < b[2] and a[1] < b[1]:
        return rel_voc.word2idx['o']
    if b[2] > a[1] and b[2] < a[2] and b[1] < a[1]:
        return rel_voc.word2idx['o']
    # contain
    if a[1] > b[1] and a[2] < b[2]:
        return rel_voc.word2idx['c']
    if b[1] > a[1] and b[2] < a[2]:
        return rel_voc.word2idx['c']
    # start
    if a[1] == b[1] and a[2] < b[2]:
        return rel_voc.word2idx['s']
    if a[1] == b[1] and a[2] > b[2]:
        return rel_voc.word2idx['s']
    # finish
    if a[2] == b[2] and a[1] < b[1]:
        return rel_voc.word2idx['f']
    if a[2] == b[2] and a[1] > b[1]:
        return rel_voc.word2idx['f']
    return rel_voc.word2idx['unk']


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def binary_metric_report(y_probs, y_preds, y):
    res = OrderedDict()
    res['precision'] = precision_score(y, y_preds)
    res['recall'] = recall_score(y, y_preds)
    res['f1'] = f1_score(y, y_preds)
    res['pr'] = average_precision_score(y, y_probs)
    try:
        res['roc'] = roc_auc_score(y, y_probs)
    except Exception as e:
        res['roc'] = 0

    return res


def load_data(data_dir='./data', test_size=0.33, dataset_prefix='ad'):
    data_path = os.path.join(data_dir, '{}_data.pkl'.format(dataset_prefix))
    if os.path.exists(data_path):
        res = dill.load(open(data_path, 'rb'))
        return res['data'], res['dx_voc'], res['rx_voc']

    # --- read csv --
    dx_pd = pd.read_csv(os.path.join(
        data_dir, 'dx_{}.csv'.format(dataset_prefix)))
    rx_pd = pd.read_csv(os.path.join(
        data_dir, 'rx_{}.csv'.format(dataset_prefix)))

    # --- transform csv to list---
    pos_data = []  # df in data list
    neg_data = []
    for id in tqdm(rx_pd[KeyObt.KEY_ID].unique()):
        sample = {}
        sample[KeyObt.KEY_DX] = dx_pd[dx_pd[KeyObt.KEY_ID] == id]
        sample[KeyObt.KEY_RX] = rx_pd[rx_pd[KeyObt.KEY_ID] == id]
        # filter out patients with medication event == 1
        if sample[KeyObt.KEY_RX].shape[0] < 2:
            continue
        if sample[KeyObt.KEY_RX][KeyObt.KEY_LABEL].values[0] == "positive":
            pos_data.append(sample)
        else:
            neg_data.append(sample)

    neg_data.extend(pos_data)
    data = neg_data

    # --- load voc ---
    dx_voc, rx_voc = Voc(), Voc()
    for sample in data:
        if not sample[KeyObt.KEY_DX] is None:
            dx_voc.add_sentence(
                list(sample[KeyObt.KEY_DX][KeyObt.KEY_DX_CODE].values))
        rx_voc.add_sentence(
            list(sample[KeyObt.KEY_RX][KeyObt.KEY_RX_CODE].values))

    # --- save ---
    dill.dump({'data': data, 'dx_voc': dx_voc,
               'rx_voc': rx_voc}, open(data_path, 'wb'))
    return data, dx_voc, rx_voc


if __name__ == "__main__":
    load_data()
