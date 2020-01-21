from collections import defaultdict, Counter
import pandas as pd
import datetime
import os
import os.path as osp
import copy
from tqdm import tqdm
import dill as pickle
import functools
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

from util import Voc, binary_metric_report
from data_util import load_transform_data



# parameters
seed = 1
data_dir = './data'
DATASET_PREFIX = 'ad'
RATIO = None # for early prediction
truncation = 128
truncation_offset = 1
THREHOLD = 0.5
min_support_ratio = 0.023 # 0.05
max_gap = datetime.timedelta(days=200)
K = 100
FILTER_TOP = 50
RUN_KARMALEGO = True
IS_VALUE = True
np.random.seed(seed)

# data structure
Relation = {
    '<': 0,  # before
    '=': 1,  # equal
    'm': 2,  # meet
    'o': 3,  # overlap
    'c': 4,  # contain
    's': 5,  # start
    'f': 6,  # finish
}
Transition_Table = [
    [['<'],['<'],['<'],['<'],['<'],['<'],['<']],
    [['<'],['='],['m'],['o'],['c'],['s'],['f']],
    [['<'],['m'],['<'],['<'],['<'],['m'],['<']],
    [['<'],['o'],['<'],['<','o','m'],['<','o','m','c','f'],['o'],['<','o','m']],
    [['<','o','m','c','f'],['c'],['c','o','f'],['c','o','f'],['c'],['c','o','f'],['c']],
    [['<'],['s'],['<'],['<','o','m'],['<','o','m','c','f'],['s'],['<','o','m']],
    [['<'],['f'],['m'],['o'],['c'],['o'],['f']],
]

def pk_save(obj, path):
    pickle.dump(obj, open(path, 'wb'))


def pk_load(path):
    return pickle.load(open(path, 'rb'))


def rel_func(A, B, epislon, max_gap):
    # before
    if B.s - A.e > epislon and B.s - A.e < max_gap:
        return Relation['<']
    # equal
    if abs(B.s-A.s) <= epislon and abs(B.e-A.e) <= epislon:
        return Relation['=']
    # meet
    if abs(B.s-A.e) <= epislon:
        return Relation['m']
    # overlap
    if B.s-A.s > epislon and A.e-B.s > epislon:
        return Relation['o']
    # contain
    if B.s-A.s > epislon and A.e-B.e > epislon:
        return Relation['c']
    # start
    if abs(B.s-A.s) <= epislon and B.e - A.e > epislon:
        return Relation['s']
    # finish
    if B.s-A.s > epislon and abs(B.e-A.e) <= epislon:
        return Relation['f']
    print(A, B)
    raise ValueError('relation not found!')


class Entity(object):
    def __init__(self, id, I, label):
        self.id = id
        self.I = I
        self.label = label

    def __repr__(self):
        return 'id:{} size:{} I:{}'.format(self.id, len(self.I), self.I)


class Interval(object):
    def __init__(self, sym, s, e):
        self.sym = sym
        self.s = s
        self.e = e

    def __repr__(self):
        return 'sym:{} s:{} e:{}'.format(self.sym, self.s, self.e)


class Instance(object):
    def __init__(self, e_idx, e_id, ti):
        self.e_idx = e_idx
        self.e_id = e_id
        self.ti = ti


class TIRP(object):
    def __init__(self, sym, tr, insts):
        self.sym = sym
        self.size = len(sym)

        self.tr = tr
        self.tr_size = len(tr)

        assert self.size*(self.size-1)//2 == self.tr_size

        self.insts = insts

    def expand(self):
        new_t = copy.deepcopy(self)

        new_t.sym.append(-1)  # -1 for placeholder
        new_t.size = len(new_t.sym)

        new_t.tr.extend([-1]*(new_t.size-1))
        new_t.tr_size = len(new_t.tr)

        for inst in new_t.insts:
            inst.ti.append(-1)

        assert new_t.size*(new_t.size-1)//2 == new_t.tr_size
        return new_t

def extract_top_k_frequent_code(data, k, filter_top_k = 100):
    code_ls = []
    for sample in data:
        for code in sample['x']:
            code_ls.append(code[0])
    counter = Counter(code_ls)
    dist = counter.most_common(k+filter_top_k)[filter_top_k:]
    codes, _ = zip(*dist)
    return codes  

def read_db(dataset_prefix='nash', k=100, filter_top_k=100):
#    data_path = os.path.join(data_dir, 'dataset:{}_k:{}_filter_k:{}_truncation:{}_karmalego_db.pkl'.format(DATASET_PREFIX, k, filter_top_k, truncation))
#    if osp.exists(data_path):
#        DB, X, Y = pk_load(data_path)
#        return DB, X, Y

    data, dx_voc, rx_voc = load_transform_data(dataset_prefix=DATASET_PREFIX)
    new_voc = Voc()
    for sample in data:
        if RATIO is not None:
            sample_x = sample['x'][:max(min(truncation, int(RATIO*len(sample['x']))),2)]
        else:
            if len(sample['x']) <= truncation:
                sample_x = sample['x']
            else:
                sample_x = sample['x'][-(truncation+truncation_offset):-truncation_offset]
        new_voc.add_sentence(list(map(lambda x:x[0], sample_x)))
        for code in sample_x:
            code[0] = new_voc.word2idx[code[0]]
        sample['x'] = sample_x

    filter_code_ls = extract_top_k_frequent_code(data, k, filter_top_k)

    X = np.zeros((len(data), len(new_voc.word2idx)))
    Y = []
    DB = []  # dataset of entities
    for i, sample in enumerate(data):
        label = sample['y']
        Y.append(label) # y
        intervals = []
        for code in sample['x']:
            if IS_VALUE:
                X[i, code[0]] = max(X[i, code[0]], (code[2]-code[1]).days)# x
            else:
                X[i, code[0]] += 1 # x
            if code[0] not in filter_code_ls:
                continue
            interval = Interval(sym=code[0], s=code[1], e=code[2])
            intervals.append(interval)
        if len(intervals) < 2:
            continue
        DB.append(Entity(id=i, I=intervals, label=label))


    def sorted_key(i, j):
        if i.s < j.s:
            return -1
        elif i.s > j.s:
            return 1

        # i.s == i.j
        if i.e < j.e:
            return -1
        elif i.e > j.e:
            return 1

        # i.e == j.e
        if i.sym < j.sym:
            return -1
        else:
            return 1

    for entity in DB:
        entity.I = sorted(entity.I, key=functools.cmp_to_key(sorted_key))
    #pk_save([DB, X, Y], data_path)
    return DB, X, Y


class KarmaLego():
    def __init__(self, db, min_support_ratio, max_gap, epsilon=datetime.timedelta(days=0)):
        self.db = db
        self.min_support_ratio = min_support_ratio
        self.max_gap = max_gap
        self.epsilon = epsilon
        self.T1 = []
        # <sym_i, sym_j, r>: Instance list
        self.T2 = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.T = []  # simple list for storing TIRPs above 2 level
    
    def satify_cnt(self, insts):
        unique_e_id = set()
        for inst in insts:
            unique_e_id.add(inst.e_idx)
        return len(unique_e_id)

    def run(self):
        self.karma()
        # transform self.T2 to TIRP form
        T2_TIRPs = []
        for i in self.T2.keys():
            for j in self.T2[i].keys():
                for r in self.T2[i][j].keys():
                    sym = [i, j]
                    tr = [r]
                    insts = self.T2[i][j][r]
                    T2_TIRPs.append(TIRP(sym, tr, insts))
        
        for t in tqdm(T2_TIRPs, desc='lego'):
            self.lego(t)
        return self.T

    def karma(self):
        # construct T2
        for e_idx, e in enumerate(tqdm(self.db, desc='karma')):
            I = e.I
            for i in range(len(I)-1):
                for j in range(i+1, len(I)):
                    if (I[j].s - I[i].e) >= self.max_gap:
                        break
                    r = rel_func(I[i], I[j], self.epsilon, self.max_gap)
                    ti = [i, j]
                    support_instance = Instance(e_idx, e.id, ti)
                    self.T2[I[i].sym][I[j].sym][r].append(support_instance)
        # prune
        new_T2 = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        T2_cnt = 0
        new_T2_cnt = 0
        for i in self.T2.keys():
            for j in self.T2[i].keys():
                for r in self.T2[i][j].keys():
                    T2_cnt += 1
                    if self.satify_cnt(self.T2[i][j][r])/len(self.db) >= self.min_support_ratio:
                        new_T2[i][j][r].extend(self.T2[i][j][r])
                        new_T2_cnt += 1
                        self.T1.extend([i, j])
        self.T1 = list(set(self.T1))
        print('origin_T2_cnt:%d, prune_T2_cnt:%d, T1_cnt:%d' %
              (T2_cnt, new_T2_cnt, len(self.T1)))
        self.T2 = new_T2

    def lego(self, t: TIRP):
        for sym in self.T1:
            for r in Relation.values():
                new_t = t.expand()
                new_t.sym[-1] = sym
                new_t.tr[-1] = r
                C = []
                self.generate_candidate_TIRPs(new_t, 1, C)
                for c in C:
                    support_cnt = self.search_supporting_instances(c)
                    if support_cnt/len(self.db) > self.min_support_ratio:
                        self.T.append(c)
                        self.lego(c)

    def generate_candidate_TIRPs(self, c, rIdx, C):
        first_rel = (c.size-rIdx)*(c.size-rIdx-1)//2 - 1
        second_rel = c.tr_size - rIdx
        trans_rels = [Relation[rel] for rel in Transition_Table[c.tr[first_rel]][c.tr[second_rel]]]
        for rel in trans_rels:
            new_c = copy.deepcopy(c)
            new_c.tr[-rIdx-1] = rel
            if first_rel > 0:
                self.generate_candidate_TIRPs(new_c,rIdx+1,C)
            else:
                C.append(new_c)

    def search_supporting_instances(self, c) -> int:
        next_sym = c.sym[-1]
        res_insts = []
        for inst in c.insts:
            rel = c.tr[-1]
            sym = c.sym[-2]
            next_stis = self.get_next_stis(inst.ti[-2], self.T2[sym][next_sym][rel])
            for next_sti in next_stis:
                new_inst = copy.deepcopy(inst)
                new_inst.ti[-1] = next_sti
                has_inst = True
                for i in range(1, c.size-1):
                    rel = c.tr[c.tr_size-1-i]
                    sym = c.sym[c.size-2-i]
                    if self.is_no_inst(new_inst.e_id, new_inst.ti[c.size-2-i], next_sti, self.T2[sym][next_sym][rel]):
                        has_inst = False
                        break
                if has_inst:
                    res_insts.append(new_inst)
        c.insts = res_insts
        return self.satify_cnt(c.insts)
                
    def get_next_stis(self, latest_sti, insts)->list:
        next_stis = []
        for inst in insts:
            if inst.ti[0] == latest_sti:
                next_stis.append(inst.ti[1])
        return next_stis

    def is_no_inst(self, e_id, cur_sti, next_sti, insts)->bool:
        for inst in insts:
            if inst.e_id == e_id and inst.ti[0]==cur_sti and inst.ti[1]==next_sti:
                return False
        return True

if __name__ == "__main__":

    DB, X, Y = read_db(dataset_prefix=DATASET_PREFIX, k=K, filter_top_k=FILTER_TOP)
    # downsampling
#    DB_pos = []
#    for idx in np.where(y_gt==1)[0]:
#        DB_pos.append(DB[idx])
#    DB_neg = []
#    for idx in np.where(y_gt==0)[0]:
#        DB_neg.append(DB[idx])
#
#    sample_size = 20
#    new_DB = []
#    new_DB.extend(DB_pos[:sample_size])
#    new_DB.extend(DB_neg[:sample_size])
#    print('neg:%d, pos:%d' %(len(DB_neg), len(DB_pos)))
#    print('new:')
#    new_y_gt = np.concatenate([np.ones(sample_size),np.zeros(sample_size)])
#
#    # replace
#    DB = new_DB
#    y_gt = new_y_gt
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, stratify=Y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.5, stratify=y_test)
    
    # LR
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    y_probs = lr_model.predict_proba(X_test)[:,1]
    y_preds = copy.deepcopy(y_probs)
    y_preds[y_preds>THREHOLD] = 1
    y_preds[y_preds<THREHOLD] = 0
    metrics_report = binary_metric_report(y_probs, y_preds, y_test)
    save_path = 'saved/lr_point_seed{}/{}_{}_ratio_{}'.format(seed, DATASET_PREFIX, truncation, 0 if RATIO is None  else RATIO)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fout = open(os.path.join(save_path, 'metrics.txt'), 'w')
    for k, v in metrics_report.items():
        print('k:%s, v:%.4f' % (k, v), file=fout)
        print('Point k:%s, v:%.4f' % (k, v))
    # close
    fout.close()
    
    if RUN_KARMALEGO:
        # Temporal feature
        model = KarmaLego(DB, min_support_ratio, max_gap)
        T = model.run()
    #    pk_save(model, osp.join(data_dir, '{}_karmalego.pkl'.format(DATASET_PREFIX)))

        interval_feature = np.zeros((len(X), len(T)))
        for t_idx, t in enumerate(T):
            for inst in t.insts:
                interval_feature[inst.e_idx, t_idx] = 1
        concat_feature = np.concatenate([X, interval_feature], axis=-1)
        # concat_feature = interval_feature
        X_train, X_test, y_train, y_test = train_test_split(
            concat_feature, Y, test_size=0.3, stratify=Y)
        X_val, X_test, y_val, y_test = train_test_split(
            X_test, y_test, test_size=0.5, stratify=y_test)
        
        lr_model = LogisticRegression()
        lr_model.fit(X_train, y_train)
        y_probs = lr_model.predict_proba(X_test)[:,1]
        y_preds = copy.deepcopy(y_probs)
        y_preds[y_preds>THREHOLD] = 1
        y_preds[y_preds<THREHOLD] = 0
        metrics_report = binary_metric_report(y_probs, y_preds, y_test)
        save_path = 'saved/lr_interval/{}_{}_min_{}'.format(DATASET_PREFIX, truncation, min_support_ratio)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        fout = open(os.path.join(save_path, 'metrics.txt'), 'w')
        for k, v in metrics_report.items():
            print('k:%s, v:%.4f' % (k, v), file=fout)
            print('Interval k:%s, v:%.4f' % (k, v))
        # close
        fout.close()


