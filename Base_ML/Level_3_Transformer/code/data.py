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

def download_jena() -> Path:  # fetch dataset
    csv_path = DATA_DIR / "jena_climate_2009_2016.csv"
    if not csv_path.exists():  # if missing
        url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"  # download link
        zip_path = DATA_DIR / "jena_climate.zip"  # temp zip
        print("[Data] Downloading Jena Climate Dataset ...")
        urllib.request.urlretrieve(url, zip_path)  # download it
        with zipfile.ZipFile(zip_path, 'r') as z:  # open zip
            z.extractall(DATA_DIR)  # extract stuff
        zip_path.unlink()  # delete zip
        print("[Data] Download complete.")
    return csv_path  # return path

def load_and_downsample(csv_path: Path) -> pd.DataFrame:  # load and shrink
    print("[Data] Loading CSV ...")
    df = pd.read_csv(csv_path)  # read csv
    if 'Date Time' in df.columns:  # check strings
        df = df.drop(columns=['Date Time'])  # drop it
    df = df.select_dtypes(include=[np.number])  # keep only numbers
    print(f"[Data] Raw shape: {df.shape}")
    print(f"[Data] Features: {list(df.columns)}")
    group_idx = np.arange(len(df)) // 6  # group by hour
    df_hourly = df.groupby(group_idx, sort=True).mean().reset_index(drop=True)  # take average
    print(f"[Data] Hourly shape: {df_hourly.shape}")
    return df_hourly  # return small df

def _make_seqs(arr: np.ndarray, seq_len: int, horizon: int, tcol: int):  # slice windows
    xs, ys = [], []  # empty lists
    total = seq_len + horizon  # window size
    for i in range(len(arr) - total + 1):  # loop data
        xs.append(arr[i : i + seq_len])  # get input
        ys.append(arr[i + seq_len : i + seq_len + horizon, tcol])  # get target
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)  # to arrays

def prepare_data(df_hourly: pd.DataFrame):  # scale and split
    feature_cols = list(df_hourly.columns)  # get columns
    temp_col_idx = feature_cols.index('T (degC)')  # target index
    n_features = len(feature_cols)  # feature count
    data = df_hourly.values.astype(np.float32)  # to float
    n = len(data)
    
    n_test = int(n * CONFIG['test_ratio'])  # test size
    n_val = int(n * CONFIG['val_ratio'])  # val size
    n_train = n - n_val - n_test  # train size
    
    train_raw = data[:n_train]  # train slice
    val_raw = data[n_train : n_train + n_val]  # val slice
    test_raw = data[n_train + n_val:]  # test slice
    print(f"\n[Split] Train: {n_train:,} h | Val: {n_val:,} h | Test: {n_test:,} h")
    
    scaler = StandardScaler()  # make scaler
    train_sc = scaler.fit_transform(train_raw)  # fit train
    val_sc = scaler.transform(val_raw)  # scale val
    test_sc = scaler.transform(test_raw)  # scale test
    
    seq_len = CONFIG['seq_len']
    horizon = CONFIG['horizon']
    
    X_train, y_train = _make_seqs(train_sc, seq_len, horizon, temp_col_idx)  # train windows
    X_val, y_val = _make_seqs(val_sc, seq_len, horizon, temp_col_idx)  # val windows
    X_test, y_test = _make_seqs(test_sc, seq_len, horizon, temp_col_idx)  # test windows
    print(f"[Seqs] Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    
    return (X_train, y_train, X_val, y_val, X_test, y_test, scaler, temp_col_idx, n_features)  # return all

def inv_temp(y_scaled: np.ndarray, scaler: StandardScaler, temp_idx: int, n_feat: int) -> np.ndarray:  # reverse scaling
    flat = y_scaled.flatten()  # flatten
    dummy = np.zeros((len(flat), n_feat), dtype=np.float32)  # fake matrix
    dummy[:, temp_idx] = flat  # insert preds here
    return scaler.inverse_transform(dummy)[:, temp_idx].reshape(y_scaled.shape)  # invert and reshape

class WeatherDataset(Dataset):  # dataset wrapper
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test):  # make loaders
    _kw = dict(
        batch_size=CONFIG['batch_size'],
        num_workers=N_WORKERS,
        pin_memory=True,
        persistent_workers=(N_WORKERS > 0),
    )
    train_loader = DataLoader(WeatherDataset(X_train, y_train), shuffle=True, **_kw)
    val_loader = DataLoader(WeatherDataset(X_val, y_val), shuffle=False, **_kw)
    test_loader = DataLoader(WeatherDataset(X_test, y_test), shuffle=False, **_kw)
    print(f"\n[Data] Batches - Train: {len(train_loader):,} | Val: {len(val_loader):,} | Test: {len(test_loader):,}")
    return train_loader, val_loader, test_loader