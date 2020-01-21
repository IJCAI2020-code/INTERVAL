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
from models import Retain as MyNet
from data_util import load_mimic_voc, load_trans4retain
from util import binary_metric_report, KeyObt


torch.manual_seed(1)
np.random.seed(1)


class RNNDataset(Dataset):
    def __init__(self, data, reverse=True):
        self.data = data
        self.reverse = reverse

    def __getitem__(self, index):
        data = self.data[index]
        if self.reverse:
            data['x'] = data['x'][::-1]

        return data

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    max_visit_len = max(list(map(lambda sample: len(sample['x']), batch)))
    max_code_len = max([len(visit_code_ls)
                        for patient in batch
                        for visit_code_ls in patient['x']]
                       )

    # code (B, V, C)
    X = torch.zeros((len(batch), max_visit_len,
                     max_code_len), dtype=torch.long)
    Y = torch.zeros(len(batch), dtype=torch.long)  # label (B)
    M = torch.zeros((len(batch), max_visit_len),
                    dtype=torch.long)  # mask (B, V)
    for i, sample in enumerate(batch):
        Y[i] = sample['y']
        M[i, :len(sample['x'])] = 1
        for j, visit in enumerate(sample['x']):
            # 1 if offset 1 added for padding (idx=0)
            X[i, j, :len(visit)] = torch.LongTensor(visit)

    return X, Y, M


class MyModel(pl.LightningModule):
    def __init__(self, hparams):
        super(MyModel, self).__init__()
        self.hparams = hparams
        self.batch_size = hparams.batch_size

        self._create_dataset()
        # self.model = MyNet(len(self.dx_voc.word2idx)+len(
        #     self.rx_voc.word2idx), hparams.h_dim, hparams.heads, hparams.has_attn, hparams.has_hidden_attn)
        self.model = MyNet(
            len(self.dx_voc.word2idx)+len(self.rx_voc.word2idx)+1,
            hparams.h_dim
        )

    def training_step(self, batch, batch_i):
        y_hat, loss = self.model(batch)
        output = OrderedDict({
            'loss': loss
        })
        return output

    def validation_step(self, batch, batch_i):
        _, y, _ = batch
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
        print(self.hparams.threhold)
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
        transformed_data, dx_voc, rx_voc = load_trans4retain(
            dataset_prefix=self.hparams.dataset_prefix, truncation=self.hparams.truncation, truncation_offset=self.hparams.truncation_offset, 
            ratio=self.hparams.ratio)
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
        dataset = RNNDataset(self.train_data)
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
        return loader

    @pl.data_loader
    def val_dataloader(self):
        dataset = RNNDataset(self.val_data)
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        return loader

    @pl.data_loader
    def test_dataloader(self):
        dataset = RNNDataset(self.test_data)
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=collate_fn
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
                        options=[0.5, 0.6, 0.7, 0.8])
        parser.opt_list('--learning_rate', default=5e-5, type=float,
                        options=[0.0001, 0.0005, 0.001, 0.005])
        parser.opt_list('--optimizer_name', default='adam', type=str,
                        options=['adam'], tunable=False)
        parser.opt_list('--batch_size', default=8, type=int,
                        options=[2, 4, 8, 16])
        return parser


def main(hparams):
    # load model
    model = MyModel(hparams)

    # init experiment
    exp = Experiment(
        name=hparams.experiment_name,
        save_dir=hparams.test_tube_save_path,
        autosave=False,
        description='baseline retain'
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
    demo_log_dir = os.path.join(root_dir, 'retain')
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
    parent_parser.add_argument('--dataset_prefix', type=str, default='nash',
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
    # hyperparams.optimize_parallel_gpu(
    #     main,
    #     max_nb_trials=10,
    #     gpu_ids=['-1', '-1']
    # )
    # hyperparams.optimize_parallel_cpu(
    #     main,
    #     nb_trials=10,
    #     nb_workers=4,
    # )


def run_test():
#    model = MyModel.load_from_metrics(
#        weights_path='saved/retain/model_weights/inner/7/_ckpt_epoch_9.ckpt',
#        tags_csv='saved/retain/test_tube_data/inner/version_7/meta_tags.csv',
#        on_gpu=True,
#        map_location=None,
#    )
#    model = MyModel.load_from_metrics(
#        weights_path='saved/retain/model_weights/nash/2/_ckpt_epoch_22.ckpt',
#        tags_csv='saved/retain/test_tube_data/nash/version_2/meta_tags.csv',
#        on_gpu=True,
#        map_location=None,
#    )
#    model = MyModel.load_from_metrics(
#        weights_path='saved/retain/model_weights/nash/6/_ckpt_epoch_4.ckpt',
#        tags_csv='saved/retain/test_tube_data/nash/version_6/meta_tags.csv',
#        on_gpu=True,
#        map_location=torch.device('cuda'),
#    )
#    trainer = Trainer()
#    trainer.test(model)
#    model = MyModel.load_from_metrics(
#        weights_path='saved/retain/model_weights/nash/7/_ckpt_epoch_4.ckpt',
#        tags_csv='saved/retain/test_tube_data/nash/version_7/meta_tags.csv',
#        on_gpu=True,
#        map_location=torch.device('cuda'),
#    )
#    trainer = Trainer()
#    trainer.test(model)
    model = MyModel.load_from_metrics(
        weights_path='saved_seed1/retain/model_weights/ad/1/_ckpt_epoch_8.ckpt',
        tags_csv='saved_seed1/retain/test_tube_data/ad/version_1/meta_tags.csv',
        on_gpu=True,
        map_location=torch.device('cuda'),
    )
    trainer = Trainer()
    trainer.test(model)


if __name__ == "__main__":
    #run_train()
    run_test()
