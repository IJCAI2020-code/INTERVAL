import os
from collections import OrderedDict
import torch.nn as nn
import torch
import torch.nn.functional as F
from test_tube import HyperOptArgumentParser, Experiment
from torch import optim
import numpy as np
import dill
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import pytorch_lightning as pl
from pytorch_lightning.models.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
# from models import AttnIntervalRNN as MyNet
# from models import T_LSTM_Model as MyNet multi-hot version
from models import TLSTM as MyNet
from data_util import load_mimic_voc, load_transform_data
from util import binary_metric_report, KeyObt, load_data

torch.manual_seed(1)
np.random.seed(1)

def map_to_log_elapse(et_s, et_e):
    return 1/np.log(np.e + (et_e - et_s).days)


class RNNDataset(Dataset):
    def __init__(self, data, hparams):
        self.data = data
        self.truncation = hparams.truncation
        self.truncation_offset = hparams.truncation_offset # the end distance to the end
        self.ratio = hparams.ratio

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        truncation = self.truncation

        lengths = []
        X = []  # code (B, L)
        Y = []  # label (B)
        T = torch.ones(len(batch), truncation-1,
                       dtype=torch.float)   # time elaplse

        # cal length first
        for sample in batch:
            if self.ratio is not None and not np.isnan(self.ratio):
                lengths.append(max(min(truncation, int(self.ratio*len(sample['x']))), 2))
            else:
                if len(sample['x']) <= truncation:
                    lengths.append(len(sample['x']))
                else:
                    lengths.append(len(sample['x'][-(truncation+self.truncation_offset):-self.truncation_offset]))

        max_len = max(lengths)

        for i, sample in enumerate(batch):
            Y.append(sample['y'])
            if self.ratio is not None and not np.isnan(self.ratio):
                sample_x = sample['x'][:lengths[i]]
            else:
                if len(sample['x']) <= truncation:
                    sample_x = sample['x']
                else:
                    sample_x = sample['x'][-(truncation+self.truncation_offset):-self.truncation_offset]
            
            X.append(list(map(lambda x: x[0], sample_x)) + [0]*(max_len-lengths[i]))
            precode = None
            for j, code in enumerate(sample_x):
                if j == 0:
                    precode = code
                    continue
                T[i, j-1] = map_to_log_elapse(precode[1], code[1])
                precode = code

        return torch.LongTensor(X), torch.LongTensor(Y), T, torch.LongTensor(lengths)-1

#    def collate_fn_multi_hot(self, batch):
#        """
#        multi_hot version
#        """
#        visit_len = list(map(lambda sample: len(sample['x']), batch))
#        max_visit_len = max(visit_len)
#
#        # code (B, V, C)
#        X = torch.zeros((len(batch), max_visit_len, self.i_dim))
#        Y = torch.zeros(len(batch), dtype=torch.long)  # label (B)
#        T = torch.ones(len(batch), max_visit_len-1,
#                       dtype=torch.float)   # time elaplse
#        for i, sample in enumerate(batch):
#            Y[i] = sample['y']
#            for j, visit in enumerate(sample['x']):
#                for code_idx in visit:
#                    X[i, j, code_idx] = 1
#                if j > 0:
#                    T[i, j -
#                        1] = map_to_log_elapse(sample['t'][j-1], sample['t'][j])
#
#        return X, Y, T, torch.LongTensor(visit_len) - 1


class MyModel(pl.LightningModule):
    def __init__(self, hparams):
        super(MyModel, self).__init__()
        self.hparams = hparams
        self.batch_size = hparams.batch_size

        self._create_dataset()
        # self.model = MyNet(len(self.dx_voc.word2idx)+len(
        #     self.rx_voc.word2idx), hparams.h_dim, hparams.heads, hparams.has_attn, hparams.has_hidden_attn)
        # self.i_dim = len(self.dx_voc.word2idx)+len(self.rx_voc.word2idx)
        # self.model = MyNet(
        #     self.i_dim,
        #     hparams.h_dim
        # )
        self.model = MyNet(len(self.dx_voc.word2idx) +
                           len(self.rx_voc.word2idx)+1, hparams.h_dim)

    def training_step(self, batch, batch_i):
        y_hat, loss = self.model(batch)
        output = OrderedDict({
            'loss': loss
        })
        return output

    def validation_step(self, batch, batch_i):
        _, y, _, _ = batch
        y_hat, loss = self.model(batch)
        y_prob = y_hat[:, 1]

        output = OrderedDict({
            'val_loss': loss,
            'y_prob': y_prob.detach().cpu().data.numpy(),
            'y': y.detach().cpu().data.numpy()
        })
        return output

    def validation_end(self, outputs):
        y_probs = []
        y_preds = []
        y = []
        val_loss_mean = 0
        for output in outputs:
            y_probs.extend(list(output['y_prob']))
            y_preds.extend(
                [0 if x < self.hparams.threhold else 1 for x in list(output['y_prob'])])
            y.extend(output['y'])
            val_loss_mean += output['val_loss']

        tqdm_dic = binary_metric_report(y_probs, y_preds, y)
        tqdm_dic['val_loss'] = val_loss_mean/len(outputs)
        print('threhold_{}'.format(self.hparams.threhold))
        print(np.array(y_probs)[np.array(y).nonzero()[0]])
        print(confusion_matrix(y, y_preds))
        print(tqdm_dic)
        return tqdm_dic

    def test_step(self, batch, batch_i):
        return self.validation_step(batch, batch_i)

    def test_end(self, outputs):
        return self.validation_end(outputs)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    def _create_dataset(self):
        # if 'mimic' in self.hparams.data_path:
        #     dx_voc, rx_voc = load_mimic_voc()
        # else:
        #     _, dx_voc, rx_voc = load_data()
        # transformed_data = dill.load(open(self.hparams.data_path, 'rb'))
        transformed_data, dx_voc, rx_voc = load_transform_data(
            dataset_prefix=self.hparams.dataset_prefix)
        Y = []
        for sample in transformed_data:
            Y.append(sample['y'])
        x_train, x_val, _, y_val = train_test_split(
            transformed_data, Y, test_size=0.3, stratify=Y)
        x_val, x_test, _, _ = train_test_split(
            x_val, y_val, test_size=0.5, stratify=y_val)
        self.train_data = x_train
        self.val_data = x_val
        self.test_data = x_test
        self.dx_voc = dx_voc
        self.rx_voc = rx_voc
        print('train:%d, val:%d, test:%d' %
              (len(x_train), len(x_val), len(x_test)))

    @pl.data_loader
    def tng_dataloader(self):
        dataset = RNNDataset(self.train_data, self.hparams)
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=dataset.collate_fn
        )
        return loader

    @pl.data_loader
    def val_dataloader(self):
        dataset = RNNDataset(self.val_data, self.hparams)
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=dataset.collate_fn
        )
        return loader

    @pl.data_loader
    def test_dataloader(self):
        dataset = RNNDataset(self.test_data, self.hparams)
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=dataset.collate_fn
        )
        return loader

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):
        parser = HyperOptArgumentParser(
            strategy=parent_parser.strategy, parents=[parent_parser])

        # network params
        parser.add_argument('--h_dim', default=32, type=int)
        parser.add_argument('--truncation', default=64, type=int)
        parser.add_argument('--truncation_offset', default=1, type=int)
        parser.add_argument('--ratio', default=None, type=float, help="ratio for early prediction")

        # training params (opt)
        parser.opt_list('--threhold', default=0.5, type=float,
                        options=[0.3, 0.4, 0.5, 0.6], tunable=False)
        parser.opt_list('--learning_rate', default=5e-5, type=float,
                        options=[0.0001, 0.0005, 0.001, 0.005],
                        tunable=False)
        parser.opt_list('--optimizer_name', default='adam', type=str,
                        options=['adam'], tunable=False)
        parser.opt_list('--batch_size', default=8, type=int)
        return parser


def main(hparams):
    # load model
    model = MyModel(hparams)

    # init experiment
    exp = Experiment(
        name=hparams.experiment_name,
        save_dir=hparams.test_tube_save_path,
        autosave=False,
        description='baseline tlstm'
    )

    exp.argparse(hparams)
    exp.save()

    # define callbackes
    model_save_path = '{}/{}/{}'.format(hparams.model_save_path,
                                        exp.name, exp.version)
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=True,
        mode='min'
    )

    checkpoint = ModelCheckpoint(
        filepath=model_save_path,
        save_best_only=True,
        verbose=True,
        monitor='pr',
        mode='max'
    )

    # init trainer
    trainer = Trainer(
        experiment=exp,
        checkpoint_callback=checkpoint,
        early_stop_callback=early_stop,
        gpus=hparams.gpus,
        val_check_interval=1
    )

    # start training
    trainer.fit(model)


def run_train():
    # dirs
    root_dir = 'saved_seed2'
    demo_log_dir = os.path.join(root_dir, 'tlstm')
    checkpoint_dir = os.path.join(demo_log_dir, 'model_weights')
    test_tube_dir = os.path.join(demo_log_dir, 'test_tube_data')

    # although we user hyperOptParser, we are using it only as argparse right now
    parent_parser = HyperOptArgumentParser(
        strategy='grid_search', add_help=False)

    # gpu args
    parent_parser.add_argument('--gpus', type=str, default='-1',
                               help='how many gpus to use in the node.'
                                    'value -1 uses all the gpus on the node')
    parent_parser.add_argument('--test_tube_save_path', type=str, default=test_tube_dir,
                               help='where to save logs')
    parent_parser.add_argument('--model_save_path', type=str, default=checkpoint_dir,
                               help='where to save model')
    parent_parser.add_argument('--experiment_name', type=str, default='inner',
                               help='test tube exp name')
    parent_parser.add_argument('--dataset_prefix', type=str, default='ad',
                               help='[ad, idf, nash]')

    # allow model to overwrite or extend args
    parser = MyModel.add_model_specific_args(
        parent_parser, root_dir)
    hyperparams = parser.parse_args()
    hyperparams.experiment_name = hyperparams.dataset_prefix

    # ---------------------
    # RUN TRAINING
    # ---------------------
    # run on HPC cluster
    print(f'RUNNING INTERACTIVE MODE ON GPUS. gpu ids: {hyperparams.gpus}')
    main(hyperparams)


def run_test():
#    model = MyModel.load_from_metrics(
#        weights_path='saved/tlstm/model_weights/nash/4/_ckpt_epoch_22.ckpt',
#        tags_csv='saved/tlstm/test_tube_data/nash/version_4/meta_tags.csv',
#        on_gpu=True,
#        map_location=torch.device('cuda')
#    )
#    model = MyModel.load_from_metrics(
#        weights_path='saved/tlstm/model_weights/ad/0/_ckpt_epoch_8.ckpt',
#        tags_csv='saved/tlstm/test_tube_data/ad/version_0/meta_tags.csv',
#        on_gpu=True,
#        map_location=torch.device('cuda')
#    )
#    model = MyModel.load_from_metrics(
#        weights_path='saved/tlstm/model_weights/nash/8/_ckpt_epoch_6.ckpt',
#        tags_csv='saved/tlstm/test_tube_data/nash/version_10/meta_tags.csv',
#        on_gpu=True,
#        map_location=torch.device('cuda')
#    )
#    trainer = Trainer()
#    trainer.test(model)
#    model = MyModel.load_from_metrics(
#        weights_path='saved/tlstm/model_weights/nash/9/_ckpt_epoch_6.ckpt',
#        tags_csv='saved/tlstm/test_tube_data/nash/version_9/meta_tags.csv',
#        on_gpu=True,
#        map_location=torch.device('cuda')
#    )
#    trainer = Trainer()
#    trainer.test(model)
    model = MyModel.load_from_metrics(
        weights_path='saved_seed1/tlstm/model_weights/ad/2/_ckpt_epoch_7.ckpt',
        tags_csv='saved_seed1/tlstm/test_tube_data/ad/version_2/meta_tags.csv',
        on_gpu=True,
        map_location=torch.device('cuda')
    )
    trainer = Trainer()
    trainer.test(model)


if __name__ == "__main__":
    #run_train()
	run_test()
