import dill
from datetime import timedelta, datetime
from tqdm import tqdm
import pandas as pd
from torch.nn import Transformer
from util import get_edge_type, load_data, KeyObt, Voc, rel_voc
from datetime import datetime, timedelta
import numpy as np
import os.path as osp
import copy
import os
from collections import Counter


def extract_r_x(code_ls, has_value=False):
    r = np.zeros((len(code_ls), len(code_ls)))
    for i, code_i in enumerate(code_ls):
        for j in range(0, i+1):  # include self
            v = get_edge_type(code_i, code_ls[j])
            if v != 0:
                r[i, j] = v

    # --- reduce code_ls to only keep the voc index
    code_ls = [code[0] for code in code_ls]
    code_va = [(code[2]-code[1]).days for code in code_ls]
    if has_value:
        return code_ls, r, code_va
    else:
        return code_ls, r


def pprint(str, file):
    print(str)
    print(str, file=file)


def dataset_stat(dataset_prefix, truncation=512, ratio=None):
    # load dataset
    data, dx_voc, rx_voc = load_transform_data(dataset_prefix=dataset_prefix)
    Y = []
    for sample in data:
        Y.append(sample['y'])
    if ratio is not None:
        for sample in data:
            sample['x'] = sample['x'][:max(
                min(truncation, int(ratio*len(sample['x']))), 2)]

    with open('./data/{}_stat_trn_{}_ratio_{}.txt'.format(dataset_prefix, truncation, ratio), 'w') as fout:

        pprint('-'*10 + 'label dist' + '-'*10, fout)
        label_dist = Counter(Y)
        for k, v in label_dist.items():
            pprint('label:{}, count:{}'.format(k, v), fout)

        pprint('-'*10 + 'unique code' + '-'*10, fout)
        pprint('dx:{} , rx:{}'.format(
            len(dx_voc.word2idx), len(rx_voc.word2idx)), fout)

        pprint('-'*10 + 'code length stat per patient' + '-'*10, fout)
        lengths = list(map(lambda sample: len(sample['x']), data))
        pprint('mean:{}, max:{}, min:{}'.format(
            sum(lengths)/len(lengths),
            max(lengths),
            min(lengths)
        ), fout)

        # extract rel count
#        pprint('-'*10 + 'rel dist per patient' + '-'*10, fout)
#        rel_list = []
#        for sample in data:
#            for i, code_i in enumerate(sample['x']):
#                for j in range(0, i):  # include self
#                    v = get_edge_type(code_i, sample['x'][j])
#                    if v != 0:
#                        rel_list.append(rel_voc.idx2word[v])
#        pprint('rel/interval:{}'.format(len(rel_list)/sum(lengths)), fout)
#        rel_dist = Counter(rel_list)
#        for k, v in rel_dist.items():
#            pprint('rel:{}, ratio:{}'.format(k, v/len(rel_list)), fout)


def stat_gap(dataset_prefix):
    res = dill.load(open('./data/{}_data.pkl'.format(dataset_prefix), 'rb'))
    data = res['data']
    event_gaps = []

    cnt = 0
    # read diff of date
    for sample in tqdm(data, desc='read data'):
        sample[KeyObt.KEY_RX] = sample[KeyObt.KEY_RX].sort_values(
            by=KeyObt.KEY_DATE)
        dates = [datetime.strptime(date, '%Y-%m-%d')
                 for date in sample[KeyObt.KEY_RX][KeyObt.KEY_DATE].values]
        gap = [(dates[i+1]-dates[i]).days for i in range(len(dates)-1)]

        if len(gap) >= 1:
            event_gaps.append(gap)
            cnt += 1

    # stat per patient
    min_gap = None
    max_gap = None
    mean_gap = 0
    for gap in tqdm(event_gaps, desc='stat'):
        if min_gap is None or min_gap > min(gap):
            min_gap = min(gap)
        if max_gap is None or max_gap < max(gap):
            max_gap = max(gap)
        mean_gap += sum(gap)/len(gap)

    print('min_gap:%.2f, max_gap:%.2f, mean_gap:%.2f' % (
        min_gap,
        max_gap,
        mean_gap/cnt,
    ))


def transform_data_sample(sample, interval_gap, default_days, dx_voc, rx_voc):
    # --- extract label & fuse dx and rx sort by date---
    code_ls = []

    for _, row in sample[KeyObt.KEY_RX].iterrows():
        code = row[KeyObt.KEY_RX_CODE]
        code_idx = 1 + len(dx_voc.word2idx) + rx_voc.word2idx[code]
        start_date = datetime.strptime(
            row[KeyObt.KEY_DATE], '%Y-%m-%d')
        supply_days = row[KeyObt.KEY_DAYS]
        end_date = start_date + \
            timedelta(days=default_days if np.isnan(
                supply_days) else supply_days)
        code_ls.append([code_idx, start_date, end_date])

    for _, row in sample[KeyObt.KEY_DX].iterrows():
        code = row[KeyObt.KEY_DX_CODE]
        code_idx = 1 + dx_voc.word2idx[code]
        start_date = datetime.strptime(
            row[KeyObt.KEY_DATE], '%Y-%m-%d')
        end_date = start_date + interval_gap
        code_ls.append([code_idx, start_date, end_date])
    # sort code_ls by start date
    code_ls = sorted(code_ls, key=lambda x: x[1])

    return code_ls

# inner_data.pkl: AD disease
# nash_data.pkl
# idf_data.pkl


def load_transform_data(interval_gap=timedelta(days=90), default_days=10, dataset_prefix='ad'):
    data, dx_voc, rx_voc = load_data(dataset_prefix=dataset_prefix)
    data_path = os.path.join(
        './data', '{}_day_{}_trans_data.pkl'.format(dataset_prefix, interval_gap.days))
    if os.path.exists(data_path):
        new_data = dill.load(open(data_path, 'rb'))
        return new_data, dx_voc, rx_voc

    new_data = []
    label_dict = {'positive': 1}
    for sample in tqdm(data):
        x = transform_data_sample(
            sample, interval_gap, default_days, dx_voc, rx_voc)
        y = label_dict.get(sample[KeyObt.KEY_RX]
                           [KeyObt.KEY_LABEL].values[0], 0)
        new_data.append({'x': x, 'y': y})
    dill.dump(new_data, open(data_path, 'wb'))
    return new_data, dx_voc, rx_voc


def load_trans4retain(window_size=timedelta(days=90), truncation=512, truncation_offset=0, dataset_prefix='ad', ratio=None):
    if ratio is None or np.isnan(ratio):
        data_path = os.path.join(
            './data', '{}_trans_retain_data_truncation:{}_offset:{}.pkl'.format(dataset_prefix, truncation, truncation_offset))
    else:
        data_path = os.path.join(
            './data', '{}_trans_retain_data_truncation:{}_ratio:{}.pkl'.format(dataset_prefix, truncation, ratio))

    data, dx_voc, rx_voc = load_transform_data(dataset_prefix=dataset_prefix)
    if os.path.exists(data_path):
        new_data = dill.load(open(data_path, 'rb'))
        return new_data, dx_voc, rx_voc

    new_data = []
    for sample in data:
        if ratio is not None and not np.isnan(ratio):
            code_ls = sample['x'][:max(
                min(truncation, int(ratio*len(sample['x']))), 2)]
        elif len(sample['x']) <= truncation:
            code_ls = sample['x']
        else:
            code_ls = sample['x'][-(truncation +
                                    truncation_offset):-truncation_offset]

        x = []
        visit_x = [code_ls[0][0]]
        left_p = 0
        for right_p in range(1, len(code_ls)):
            if code_ls[right_p][1] - code_ls[left_p][1] <= window_size:
                # in the same window size
                visit_x.append(code_ls[right_p][0])
            else:
                x.append(visit_x)
                visit_x = [code_ls[right_p][0]]
                left_p = right_p
        new_data.append({'x': x+[visit_x], 'y': sample['y']})

    stat_file = open(data_path+'_stat.txt', 'w')
    visit_len = list(map(lambda x: len(x['x']), new_data))
    print('mean visit len:', sum(visit_len)/len(new_data), file=stat_file)
    print('max visit len:', max(visit_len), file=stat_file)
    print('min visit len:', min(visit_len), file=stat_file)

    dill.dump(new_data, open(data_path, 'wb'))
    return new_data, dx_voc, rx_voc


if __name__ == "__main__":
    dataset_stat('ad', ratio=0.1)
    dataset_stat('ad', ratio=0.3)
    dataset_stat('nash', ratio=0.1)
    dataset_stat('nash', ratio=0.3)
