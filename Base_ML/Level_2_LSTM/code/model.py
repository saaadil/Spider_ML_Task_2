import torch
import torch.nn as nn
from config import CONFIG

class MyLSTMCell(nn.Module):  # custom cell
    def __init__(self, inp_sz, hid_sz):
        super().__init__()
        self.h_size = hid_sz  # save size
        comb = inp_sz + hid_sz  # total size
        self.wi = nn.Linear(comb, hid_sz)  # input gate
        self.wf = nn.Linear(comb, hid_sz)  # forget gate
        self.wo = nn.Linear(comb, hid_sz)  # output gate
        self.wg = nn.Linear(comb, hid_sz)  # candidate gate
        nn.init.constant_(self.wf.bias, 1.0)  # keep memory

    def forward(self, xt, h_prev, c_prev):  # math step
        c = torch.cat([xt, h_prev], dim=1)  # merge in
        it = torch.sigmoid(self.wi(c))  # input calc
        ft = torch.sigmoid(self.wf(c))  # forget calc
        ot = torch.sigmoid(self.wo(c))  # output calc
        gt = torch.tanh(self.wg(c))  # candidate calc
        ct = ft * c_prev + it * gt  # update cell
        ht = ot * torch.tanh(ct)  # update hidden
        return ht, ct  # return states

    def make_hidden(self, bs, dev):  # empty states
        h = torch.zeros(bs, self.h_size, device=dev)
        c = torch.zeros(bs, self.h_size, device=dev)
        return h, c

class MyLSTMLayer(nn.Module):  # unroll loop
    def __init__(self, inp, hid, drop_p=0.0):
        super().__init__()
        self.h = hid
        self.c = MyLSTMCell(inp, hid)
        self.drop = nn.Dropout(p=drop_p)  # dropout logic

    def forward(self, x, state=None):
        bs, slen, _ = x.shape
        if state is None:  # start blank
            ht, ct = self.c.make_hidden(bs, x.device)
        else:
            ht, ct = state
            
        h_list = []  # track steps
        for t in range(slen):  # loop time
            ht, ct = self.c(x[:, t, :], ht, ct)
            h_list.append(ht)
            
        out = torch.stack(h_list, dim=1)  # stack steps
        out = self.drop(out)
        return out, (ht, ct)

class LSTMStack(nn.Module):  # layer wrapper
    def __init__(self, in_sz, h_sz, nl=2, dp=0.3):
        super().__init__()
        layrs = []  # empty list
        for i in range(nl):  # loop layers
            cur_in = in_sz if i == 0 else h_sz
            d = dp if i < nl - 1 else 0.0
            layrs.append(MyLSTMLayer(cur_in, h_sz, dropout=d))
        self.lyrs = nn.ModuleList(layrs)  # register layers

    def forward(self, x):  # pass through
        curr = x
        for l in self.lyrs:
            curr, _ = l(curr)
        return curr

class WxModel(nn.Module):  # final model
    def __init__(self, inp, hid, n_lay, horiz, drp):
        super().__init__()
        self.lstm = LSTMStack(inp, hid, n_lay, drp)
        self.norm = nn.LayerNorm(hid)  # norm state
        self.do = nn.Dropout(p=drp)
        self.linear = nn.Linear(hid, horiz)  # output layer
        nn.init.xavier_uniform_(self.linear.weight)  # smart init
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        out = self.lstm(x)
        last_h = out[:, -1, :]  # grab last
        last_h = self.norm(last_h)
        last_h = self.do(last_h)
        res = self.linear(last_h)  # predict
        return res