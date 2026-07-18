import urllib.request
import zipfile
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from config import DATA_DIR, CONFIG, N_WORKERS

warnings.filterwarnings('ignore')  # hide warnings

def get_data():  # fetch dataset
    csv_file = DATA_DIR / "jena_climate_2009_2016.csv"
    if not csv_file.exists():  # if missing
        url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"  # download link
        zpath = DATA_DIR / "jena_climate.zip"  # temp zip
        urllib.request.urlretrieve(url, zpath)  # download it
        with zipfile.ZipFile(zpath, 'r') as z:  # open zip
            z.extractall(DATA_DIR)  # extract stuff
        zpath.unlink()  # delete zip
    return csv_file  # return path

def load_data(c_path): # load and shrink
    df = pd.read_csv(c_path)  # read csv
    if 'Date Time' in df.columns:  # check strings
        df = df.drop(columns=['Date Time'])  # drop it
    df = df.select_dtypes(include=[np.number])  # keep only numbers
    idx = np.arange(len(df)) // 6  # group by hour
    df_hourly = df.groupby(idx, sort=True).mean().reset_index(drop=True)  # take average
    return df_hourly  # return small df

def make_windows(arr, s_len, h, t_col):  # slice windows
    x_data, y_data = [], []  # empty lists
    tot = s_len + h  # window size
    for i in range(len(arr) - tot + 1):  # loop data
        x_data.append(arr[i : i + s_len])  # get input
        y_data.append(arr[i + s_len : i + s_len + h, t_col])  # get target
    return np.array(x_data, dtype=np.float32), np.array(y_data, dtype=np.float32)  # to arrays

def prep_data(df_in):  # scale and split
    cols = list(df_in.columns)  # get columns
    t_idx = cols.index('T (degC)')  # target index
    num_f = len(cols)  # feature count
    raw_data = df_in.values.astype(np.float32)  # to float
    n = len(raw_data)
    
    n_test = int(n * CONFIG['test_ratio'])  # test size
    n_val = int(n * CONFIG['val_ratio'])  # val size
    n_train = n - n_val - n_test  # train size
    
    tr_raw = raw_data[:n_train]  # train slice
    v_raw = raw_data[n_train : n_train + n_val]  # val slice
    te_raw = raw_data[n_train + n_val:]  # test slice
    
    sc = StandardScaler()  # make scaler
    tr_sc = sc.fit_transform(tr_raw)  # fit train
    v_sc = sc.transform(v_raw)  # scale val
    te_sc = sc.transform(te_raw)  # scale test
    
    s_len = CONFIG['seq_len']
    hor = CONFIG['horizon']
    
    x_tr, y_tr = make_windows(tr_sc, s_len, hor, t_idx)  # train windows
    x_v, y_v = make_windows(v_sc, s_len, hor, t_idx)  # val windows
    x_te, y_te = make_windows(te_sc, s_len, hor, t_idx)  # test windows
    
    return (x_tr, y_tr, x_v, y_v, x_te, y_te, sc, t_idx, num_f)  # return all

def invert_temp(y_sc, scaler, t_idx, num_feat):  # reverse scaling
    f = y_sc.flatten()  # flatten
    dum = np.zeros((len(f), num_feat), dtype=np.float32)  # fake matrix
    dum[:, t_idx] = f  # insert preds here
    return scaler.inverse_transform(dum)[:, t_idx].reshape(y_sc.shape)  # invert and reshape

class WxData(Dataset):  # dataset wrapper
    def __init__(self, x_in, y_in):
        self.x = torch.from_numpy(x_in)
        self.y = torch.from_numpy(y_in)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

def build_loaders(xtr, ytr, xv, yv, xte, yte):  # make loaders
    args = dict(
        batch_size=CONFIG['batch_size'],
        num_workers=N_WORKERS,
        pin_memory=True,
        persistent_workers=(N_WORKERS > 0),
    )
    tr_loader = DataLoader(WxData(xtr, ytr), shuffle=True, **args)
    v_loader = DataLoader(WxData(xv, yv), shuffle=False, **args)
    te_loader = DataLoader(WxData(xte, yte), shuffle=False, **args)
    return tr_loader, v_loader, te_loader