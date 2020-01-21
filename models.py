from modules import TransformerBlock, LayerNorm, TransformerBlockSimple
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init
# from torch_geometric.nn import GCNConv, RGCNConv
import numpy as np
import math
from util import rel_voc


def init_weights(module):
    """ Initialize the weights.
    """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        init.xavier_normal_(module.weight.data)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()
    if isinstance(module, nn.GRU):
        for param in module.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    if isinstance(module, LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


class VanilaRNN(nn.Module):
    def __init__(self, dx_size, rx_size, h_dim):
        super(VanilaRNN, self).__init__()
        self.dx_embedding = nn.Embedding(dx_size+1, h_dim, padding_idx=dx_size)
        self.rx_embedding = nn.Embedding(rx_size+1, h_dim, padding_idx=rx_size)

        self.dx_rnn = nn.GRU(h_dim, h_dim, batch_first=True, dropout=0.3)
        self.rx_rnn = nn.GRU(h_dim, h_dim, batch_first=True, dropout=0.3)

        self.classifer = nn.Sequential(
            nn.Linear(2*h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim//2),
            nn.ReLU(),
            nn.Linear(h_dim//2, 2)
        )

        self.apply(init_weights)

    def forward(self, batch):
        dx, rx, y = batch  # dx:[B,Td] rx:[B,Tr] y:[B,1]
        dx_emb = self.dx_embedding(dx)
        rx_emb = self.rx_embedding(rx)

        _, h_d = self.dx_rnn(dx_emb)
        _, h_r = self.rx_rnn(rx_emb)

        concat_feat = torch.cat(
            [h_d, h_r], dim=-1).squeeze(dim=0)  # [B,2*h_dim]
        output = self.classifer(concat_feat)
        y_hat = F.softmax(output, dim=-1)
        return y_hat, F.cross_entropy(output, y.squeeze(dim=-1))


class RNNSeq(nn.Module):
    def __init__(self, emb_num, h_dim):
        super(RNNSeq, self).__init__()
        self.emb = nn.Embedding(emb_num, h_dim)

        self.gru = nn.GRU(h_dim, h_dim, batch_first=True, dropout=0.3)

        self.classifer = nn.Sequential(
            nn.Linear(h_dim, h_dim//2),
            nn.ReLU(),
            nn.Linear(h_dim//2, 2)
        )

        self.apply(init_weights)

    def forward(self, code_idx, y):
        code_idx = torch.LongTensor([code_idx]).to(self.emb.weight.device)
        code_emb = self.emb(code_idx)

        _, h = self.gru(code_emb)

        output = self.classifer(h.squeeze(dim=0))
        y_hat = F.softmax(output, dim=-1)
        return y_hat, F.cross_entropy(output, torch.LongTensor([y]).to(self.emb.weight.device))


"""
IntervalRNN
"""


class Attention(nn.Module):
    def __init__(self, h_dim):
        super(Attention, self).__init__()
        self.h_dim = h_dim
        self.K = nn.Linear(h_dim, h_dim, bias=False)
        self.V = nn.Linear(h_dim, h_dim, bias=False)
        self.A = nn.Linear(h_dim, h_dim, bias=False)

    def forward(self, k, v, r):
        a = torch.cat([k, v])  # (N+1, dim)
        k = self.K(k)  # (1, dim)
        v = self.V(a)  # (N+1, dim)
        a = self.A(a)  # (N+1, dim)
        attn_weight = torch.matmul(k, v.transpose(
            0, 1)) / math.sqrt(self.h_dim)  # (1,N)
        attn_weight = F.softmax(attn_weight, dim=-1)
        return torch.matmul(attn_weight, a)


"""
Interval 
"""


class AttentionBatch(nn.Module):
    def __init__(self, h_dim, dropout=0.2):
        super(AttentionBatch, self).__init__()
        self.output_linear = nn.Linear(h_dim, h_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, R=None):
        mask = (R > 0) if R is not None else None  # (B,L,L)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
            / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)

        output = self.output_linear(torch.matmul(p_attn, value))
        return output, p_attn


class RelationAttentionBatch(nn.Module):
    def __init__(self, h_dim, dropout=0.2):
        super(RelationAttentionBatch, self).__init__()

        relation_num = len(rel_voc.word2idx)
        self.relation_embedding = nn.Embedding(
            relation_num, h_dim*h_dim, padding_idx=0)
        # self.relation_embedding = nn.Embedding(
        #     relation_num, h_dim, padding_idx=0)

        self.output_linear = nn.Linear(h_dim, h_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, R=None):
        """
        Q,K,V: (B,L,D)
        R: (B,L,L)
        """
        # not batch x^TRx
        # scores = []
        # for i in range(R.size(1)):
        #     rel_emb = self.relation_embedding(R[:, i, :].long())  # (B,L,D*D)
        #     rel_emb = rel_emb.view(R.size(0), R.size(
        #         2), query.size(-1), -1)  # (B,L,D,D)
        #     score = torch.matmul(rel_emb, key.unsqueeze(
        #         dim=-1)).squeeze(dim=-1)  # (B,L,D)
        #     q = query[:, i, :].unsqueeze(dim=-1)  # (B,D,1)
        #     score = torch.matmul(score, q).transpose(-2, -1)  # (B,1,L)
        #     scores.append(score)

        # scores = torch.cat(scores, dim=1) \
        #     / math.sqrt(query.size(-1))  # (B,L,L)

        # batch x^TRx
        rel_emb = self.relation_embedding(R.long())  # (B,L,L,D*D)
        rel_emb = rel_emb.view(R.size(0), R.size(1), R.size(
            2), query.size(-1), -1)  # (B,L,L,D,D)
        rel_key = torch.matmul(rel_emb, key.unsqueeze(
            dim=1).unsqueeze(dim=-1)).squeeze(dim=-1)  # (B,L,L,D)
        scores = torch.matmul(rel_key, query.unsqueeze(
            dim=-1)).squeeze(dim=-1)  # (B,L,L)

        mask = (R > 0)  # (B,L,L)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)

        return self.output_linear(torch.matmul(p_attn, value)), p_attn


class INTERVAL(nn.Module):
    def __init__(self, emb_num, h_dim, n_layers=1, param_reg=0.1):
        super(INTERVAL, self).__init__()
        self.n_layers = n_layers
        self.param_reg = param_reg

        self.emb = nn.Embedding(emb_num+1, h_dim, padding_idx=0)
        self.attns = nn.ModuleList([RelationAttentionBatch(h_dim)
                                    for i in range(0, n_layers)])
        self.rnns = nn.ModuleList(
            [nn.GRU(h_dim, h_dim, batch_first=True, dropout=0.2) for i in range(0, n_layers)])

        self.output_rnn = nn.GRU(h_dim, 1, batch_first=True, dropout=0.2)
        #self.output_attn = nn.Linear(h_dim, 1)
        self.classifer = nn.Sequential(
            nn.Linear(h_dim, h_dim//2),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(h_dim//2, 2)
        )

        self.apply(init_weights)

    def forward(self, batch):
        """
        x: code index
        y: label
        r: relation matrix (B,L,L)
        l: lengths
        """
        x, y, r, l = batch
        x = self.emb(x)  # (B,L,D)

        hidden_outputs = []
        mask = torch.zeros((x.size(0), x.size(1)))  # (B,L)
        for i in range(x.size(0)):
            mask[:, :(l[i]+1)] = 1
        mask = mask.to(self.emb.weight.device)
        o = x
        for i in range(self.n_layers):
            # modified rnn
            o, local_attn = self.attns[i](o, o, o, r)
            o, _ = self.rnns[i](o)  # (B,L,D)

            # attn sum -> o
            global_attn, _ = self.output_rnn(o)  # (B,L,1)
            global_attn = global_attn.squeeze(dim=-1)  # (B,L)
            # global_attn = self.output_attn(o).squeeze(dim=-1) # (B,L)
            global_attn = global_attn.masked_fill(mask == 0, -1e9)
            global_attn = F.softmax(global_attn, dim=-1)

            c = global_attn.unsqueeze(dim=1).bmm(o).squeeze(dim=1)  # (B,D)
            # h = o[torch.arange(o.size(0)), l, :]  # (B, D)
            output = self.classifer(c)
            y_hat = F.softmax(output, dim=-1)
            hidden_outputs.append(output)
        loss = self.get_loss(hidden_outputs, y)

        return y_hat, loss

    def get_loss(self, o, y):
        """
        o : list (n_layer, B, h_dim)
        """
        regularization_loss = 0
        # for i in range(self.n_layers-1):
        #    regularization_loss += 1/self.n_layers * F.cross_entropy(o[i], y)
        loss = F.cross_entropy(o[-1], y) + regularization_loss
        return loss


class AttnRNN(nn.Module):
    def __init__(self, h_dim, n_layers=1, is_multi=False):
        super(AttnRNN, self).__init__()
        self.n_layers = n_layers

        if is_multi:
            self.attns = nn.ModuleList([RelationAttentionBatch(h_dim)
                                        for i in range(0, n_layers)])
        else:
            self.attns = nn.ModuleList([AttentionBatch(h_dim)
                                        for i in range(0, n_layers)])
        self.rnns = nn.ModuleList(
            [nn.GRU(h_dim, h_dim, batch_first=True, dropout=0.2) for i in range(0, n_layers)])

    def forward(self, x, mask):
        o = x
        for i in range(0, self.n_layers):
            o, p_attn = self.attns[i](o, o, o, mask)
            o, _ = self.rnns[i](o)
        return o


class MultiAttnRNN(nn.Module):
    def __init__(self, emb_num, h_dim=200, n_layers=1, only_rnn=False, is_multi=False):
        super(MultiAttnRNN, self).__init__()
        self.only_rnn = only_rnn

        self.emb = nn.Embedding(emb_num+1, h_dim, padding_idx=0)

        self.li = nn.Linear(h_dim+1, h_idm)

        if self.only_rnn:
            self.hidden_layer = nn.GRU(
                h_dim, h_dim, num_layers=n_layers, batch_first=True, dropout=0.2)
        else:
            self.hidden_layer = AttnRNN(h_dim, n_layers, is_multi)
        self.classifer = nn.Sequential(
            nn.Linear(h_dim, h_dim//2),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(h_dim//2, 2)
        )

        self.apply(init_weights)

    def forward(self, batch):
        """
        x: code index
        y: label
        r: relation matrix (B,L,L)
        l: lengths
        """
        if self.only_rnn:
            x, y, l = batch
            x = self.emb(x)  # (B,L,D)
            o, _ = self.hidden_layer(x)
        else:
            x, y, r, l = batch[0], batch[1], batch[2], batch[3]
            x = self.emb(x)  # (B,L,D)
            if len(batch) > 4:
                va = batch[4]
                x = self.li(torch.cat([x, va], dim=-1))
            o = self.hidden_layer(x, r)

        h = o[torch.arange(o.size(0)), l, :]  # (B, D)
        output = self.classifer(h)
        y_hat = F.softmax(output, dim=-1)
        return y_hat, F.cross_entropy(output, y)


class AttnIntervalRNN(nn.Module):
    def __init__(self, emb_num, h_dim=200, heads=4, has_attn=False, is_multi_rel=False, has_hidden_attn=False):
        super(AttnIntervalRNN, self).__init__()
        self.has_attn = has_attn
        self.has_hidden_attn = has_hidden_attn
        self.is_multi_rel = is_multi_rel
        self.emb = nn.Embedding(emb_num+1, h_dim, padding_idx=0)
        if has_attn:
            self.attn = TransformerBlockSimple(
                h_dim, heads, h_dim, dropout=0.2)

        self.rnn = nn.GRU(h_dim, h_dim, batch_first=True, dropout=0.3)
        if self.has_hidden_attn:
            self.rnn2 = nn.GRU(h_dim, h_dim, batch_first=True, dropout=0.3)

        self.classifer = nn.Sequential(
            nn.Linear(h_dim, h_dim//2),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(h_dim//2, 2)
        )

        self.apply(init_weights)

    def forward(self, batch):
        """
        x: code index
        y: label
        r: relation matrix (B,L,L)
        l: lengths
        """
        x, y, r, l = batch

        x = self.emb(x)  # (B,L,D)
        mask = (r > 0).unsqueeze(dim=1)  # (B,1,L,L)
        if self.has_attn:
            x = self.attn(x, mask)  # (B,L,D)
        o, _ = self.rnn(x)  # (B, L, D)
        if self.has_hidden_attn:
            o = self.attn(o, mask)
            o, _ = self.rnn2(o)

        h = o[torch.arange(o.size(0)), l, :]  # (B, D)
        output = self.classifer(h)
        y_hat = F.softmax(output, dim=-1)
        return y_hat, F.cross_entropy(output, y)


"""
Retain
"""


class Retain(nn.Module):
    def __init__(self, emb_num, h_dim):
        super(Retain, self).__init__()
        self.emb_num = emb_num
        self.h_dim = h_dim

        self.emb = nn.Sequential(
            nn.Embedding(emb_num, h_dim, padding_idx=0),
            nn.Dropout(0.2)
        )
        self.alpha_gru = nn.GRU(
            h_dim, h_dim, batch_first=True, dropout=0.2)
        self.beta_gru = nn.GRU(
            h_dim, h_dim, batch_first=True, dropout=0.2)

        self.alpha_li = nn.Linear(h_dim, 1)
        self.beta_li = nn.Linear(h_dim, h_dim)

        self.classifer = nn.Sequential(
            nn.Linear(h_dim, h_dim//2),
            nn.Dropout(0.2),
            nn.ReLU(),
            # nn.BatchNorm1d(h_dim//2),
            nn.Linear(h_dim//2, 2)
        )

        self.apply(init_weights)

    def forward(self, batch):
        """
        x: (B,V,C) # V for Visit_len, C for codes_len
        y: (B)
        mask: (B,V)
        """
        x, y, mask = batch
        visit_emb = self.emb(x)
        visit_emb = visit_emb.sum(dim=-2)  # (B, V, D) is the h_dim

        g, _ = self.alpha_gru(visit_emb)
        h, _ = self.beta_gru(visit_emb)

        attn_alpha = self.alpha_li(g).squeeze(dim=-1)  # (B,V)
        attn_alpha = attn_alpha.masked_fill(mask == 0, -1e9)
        attn_alpha = F.softmax(attn_alpha, dim=-1)

        attn_h = F.tanh(self.beta_li(h))  # (B,V,D)

        c = attn_alpha.unsqueeze(dim=1).bmm(
            attn_h * visit_emb
        ).squeeze(dim=1)  # (B,D)

        output = self.classifer(c)
        y_hat = F.softmax(output, dim=-1)
        return y_hat, F.cross_entropy(output, y)


"""
T-LSTM
"""


class TLSTM_Cell(nn.Module):
    def __init__(self, i_dim, h_dim):
        super(TLSTM_Cell, self).__init__()
        self.lstm = nn.LSTMCell(i_dim, h_dim)
        self.w_d = nn.Linear(h_dim, h_dim)

    def forward(self, X, T):
        """
        X: (B,V,D)
        T: (B,V-1)
        """
        O = []
        c_t, h_t = None, None
        for i in range(X.size(1)):
            if c_t is not None:
                # adjust previous memory
                c_t_S = F.tanh(self.w_d(c_t))  # (B,D)
                c_t_S_star = c_t_S * T[:, i-1:i]  # (B,1)
                c_t_T = c_t - c_t_S
                c_t_star = c_t_T + c_t_S_star
                c_t = c_t_star
                c_t, h_t = self.lstm(X[:, i, :], (c_t, h_t))
            else:
                c_t, h_t = self.lstm(X[:, i, :])
            O.append(h_t)
        return torch.stack(O, dim=1), h_t


class TLSTM(nn.Module):
    """
    Embedding version
    """

    def __init__(self, emb_num, h_dim):
        super(TLSTM, self).__init__()
        self.emb_num = emb_num
        self.h_dim = h_dim

        self.emb = nn.Sequential(
            nn.Embedding(emb_num, h_dim, padding_idx=0),
            nn.Dropout(0.2)
        )
        self.rnn = TLSTM_Cell(h_dim, h_dim)

        self.classifer = nn.Sequential(
            nn.Linear(h_dim, h_dim//2),
            nn.Dropout(0.2),
            nn.ReLU(),
            # nn.BatchNorm1d(h_dim//2),
            nn.Linear(h_dim//2, 2)
        )

        self.apply(init_weights)

    def forward(self, batch):
        """
        X: (B, L)
        y: (B)
        T: (B, L-1)
        V: (B)
        """
        x, y, T, V = batch
        x = self.emb(x)  # (B,L,D)
        o, _ = self.rnn(x, T)

        h = o[torch.arange(o.size(0)), V, :]  # (B, D)
        output = self.classifer(h)
        y_hat = F.softmax(output, dim=-1)
        return y_hat, F.cross_entropy(output, y)


class T_LSTM_Model(nn.Module):
    """
    multi-hot version
    """

    def __init__(self, i_dim, h_dim):
        super(T_LSTM_Model, self).__init__()
        self.i_dim = i_dim
        self.h_dim = h_dim

        self.rnn = TLSTM_Cell(i_dim, h_dim)
        # self.rnn = nn.GRU(i_dim, h_dim, batch_first=True)

        self.classifer = nn.Sequential(
            nn.Linear(h_dim, h_dim//2),
            nn.ReLU(),
            # nn.BatchNorm1d(h_dim//2),
            nn.Linear(h_dim//2, 2)
        )

        self.apply(init_weights)

    def forward(self, batch):
        """
        X: (B, V, L)
        y: (B)
        T: (B, V-1)
        V: (B)
        """
        X, y, T, V = batch
        o, _ = self.rnn(X, T)  # (B,V,L)
        h = o[torch.arange(o.size(0)), V, :]  # (B,L)
        output = self.classifer(h)
        y_hat = F.softmax(output, dim=-1)
        return y_hat, F.cross_entropy(output, y)
