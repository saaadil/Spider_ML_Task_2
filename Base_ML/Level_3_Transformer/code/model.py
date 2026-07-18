import math
import torch
import torch.nn as nn
from config import CONFIG

class SinusoidalPositionalEncoding(nn.Module):  # pos encoder
    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)  # dropout logic
        pe = torch.zeros(max_len, d_model)  # blank matrix
        position = torch.arange(0, max_len).unsqueeze(1).float()  # positions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # frequencies
        pe[:, 0::2] = torch.sin(position * div_term)  # apply sin
        pe[:, 1::2] = torch.cos(position * div_term)  # apply cos
        self.register_buffer('pe', pe.unsqueeze(0))  # save buffer

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # math pass
        x = x + self.pe[:, :x.size(1), :]  # add encoding
        return self.dropout(x)  # return dropped

class CustomLayerNorm(nn.Module):  # manual norm
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))  # scale var
        self.beta = nn.Parameter(torch.zeros(d_model))  # shift var

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # math pass
        mean = x.mean(dim=-1, keepdim=True)  # get mean
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # get var
        out = (x - mean) / torch.sqrt(var + self.eps)  # normalize it
        return self.gamma * out + self.beta  # scale shift

class MultiHeadAttention(nn.Module):  # mha block
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # head size
        self.q_linear = nn.Linear(d_model, d_model)  # query weights
        self.k_linear = nn.Linear(d_model, d_model)  # key weights
        self.v_linear = nn.Linear(d_model, d_model)  # value weights
        self.out_linear = nn.Linear(d_model, d_model)  # out weights
        self.dropout = nn.Dropout(dropout)  # drop logic

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # math pass
        bs, seq_len, _ = x.size()
        q = self.q_linear(x).view(bs, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # build q
        k = self.k_linear(x).view(bs, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # build k
        v = self.v_linear(x).view(bs, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # build v
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # dot product
        attn = self.dropout(torch.softmax(scores, dim=-1))  # get probs
        context = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bs, seq_len, self.d_model)  # apply attn
        return self.out_linear(context)  # project out

class EncoderBlock(nn.Module):  # trans layer
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads, dropout)  # init mha
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )  # init ffn
        self.norm1 = CustomLayerNorm(d_model)  # first norm
        self.norm2 = CustomLayerNorm(d_model)  # second norm
        self.dropout1 = nn.Dropout(dropout)  # drop one
        self.dropout2 = nn.Dropout(dropout)  # drop two

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # math pass
        nx = self.norm1(x)  # pre norm
        x = x + self.dropout1(self.mha(nx))  # mha skip
        nx2 = self.norm2(x)  # pre norm
        x = x + self.dropout2(self.ffn(nx2))  # ffn skip
        return x  # return out

class WeatherTransformer(nn.Module):  # main transformer
    def __init__(self, input_size: int, d_model: int, n_heads: int, n_encoder_layers: int, d_ff: int, horizon: int, seq_len: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)  # linear projection
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=seq_len, dropout=dropout)  # pass seq len
        self.layers = nn.ModuleList([EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_encoder_layers)])  # stack layers
        self.norm = CustomLayerNorm(d_model)  # final norm
        self.fc = nn.Linear(d_model, horizon)  # output prediction
        nn.init.xavier_uniform_(self.fc.weight)  # smart init
        nn.init.zeros_(self.fc.bias)  # zero bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # model pass
        x = self.input_proj(x)  # project input
        x = self.pos_enc(x)  # add positional
        for layer in self.layers:  # loop blocks
            x = layer(x)
        x = x[:, -1, :]  # fix: last token extraction
        x = self.norm(x)  # normalize state
        return self.fc(x)  # predict output

    def count_parameters(self) -> int:  # count weights
        return sum(p.numel() for p in self.parameters() if p.requires_grad)  # sum trainable

class myLSTMcell(nn.Module):  # custom cell
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size  # save size
        self.wi = nn.Linear(input_size, hidden_size)  # in weights
        self.ui = nn.Linear(hidden_size, hidden_size, bias=False)  # in hidden
        self.wf = nn.Linear(input_size, hidden_size)  # forget weights
        self.uf = nn.Linear(hidden_size, hidden_size, bias=False)  # forget hidden
        self.wo = nn.Linear(input_size, hidden_size)  # out weights
        self.uo = nn.Linear(hidden_size, hidden_size, bias=False)  # out hidden
        self.wc = nn.Linear(input_size, hidden_size)  # cand weights
        self.uc = nn.Linear(hidden_size, hidden_size, bias=False)  # cand hidden
        nn.init.constant_(self.wf.bias, 1.0)  # keep memory

    def forward(self, x: torch.Tensor, states: tuple) -> tuple:  # math pass
        h, c = states
        i = torch.sigmoid(self.wi(x) + self.ui(h))  # calc in
        f = torch.sigmoid(self.wf(x) + self.uf(h))  # calc forget
        o = torch.sigmoid(self.wo(x) + self.uo(h))  # calc out
        c_tilde = torch.tanh(self.wc(x) + self.uc(h))  # calc cand
        c_next = f * c + i * c_tilde  # update cell
        h_next = o * torch.tanh(c_next)  # update hidden
        return h_next, c_next  # return states

class ComparisonLSTM(nn.Module):  # baseline lstm
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, horizon: int, dropout: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cells = nn.ModuleList()  # stack cells
        self.cells.append(myLSTMcell(input_size, hidden_size))  # first lay
        for _ in range(1, num_layers):
            self.cells.append(myLSTMcell(hidden_size, hidden_size))  # hidden lay
        self.norm = CustomLayerNorm(hidden_size)  # state norm
        self.dropout = nn.Dropout(p=dropout)  # drop logic
        self.fc = nn.Linear(hidden_size, horizon)  # out layer
        nn.init.xavier_uniform_(self.fc.weight)  # init weights
        nn.init.zeros_(self.fc.bias)  # init bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # model pass
        bs, seq_len, _ = x.size()
        dev = x.device
        h = [torch.zeros(bs, self.hidden_size, device=dev) for _ in range(self.num_layers)]  # blank h
        c = [torch.zeros(bs, self.hidden_size, device=dev) for _ in range(self.num_layers)]  # blank c
        for t in range(seq_len):  # time loop
            xt = x[:, t, :]
            for layer in range(self.num_layers):  # lay loop
                h[layer], c[layer] = self.cells[layer](xt, (h[layer], c[layer]))
                xt = h[layer]
        h_T = h[-1]  # get last
        h_T = self.norm(h_T)  # norm state
        h_T = self.dropout(h_T)  # apply drop
        return self.fc(h_T)  # predict output

    def count_parameters(self) -> int:  # count weights
        return sum(p.numel() for p in self.parameters() if p.requires_grad)  # sum trainable