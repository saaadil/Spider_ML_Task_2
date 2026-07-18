import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix
from config import TEMP_BINS

def categorize_temp(t_arr):  # bin temps
    b = np.zeros(t_arr.shape, dtype=int)  # empty array
    b[t_arr < 0] = 0  # very cold
    b[(t_arr >= 0) & (t_arr < 10)] = 1
    b[(t_arr >= 10) & (t_arr < 20)] = 2
    b[t_arr >= 20] = 3
    return b  # return bins

def do_plots(h, p):  # plot loss
    eps = range(1, len(h['tr']) + 1)  # x axis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(eps, h['tr'], label='Train', color='steelblue', lw=2)
    ax1.plot(eps, h['val'], label='Val', color='tomato', lw=2)
    ax1.set_title('Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2.plot(eps, h['val'], color='tomato', lw=2, label='Val MSE')
    ax2.set_title('Validation Loss')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(p, dpi=150, bbox_inches='tight')  # save png
    plt.close()  # free memory

def do_preds(t, p_arr, pth, n=500):  # plot timeline
    t1 = t[:n, 0]  # first hour
    p1 = p_arr[:n, 0]
    
    fig, axs = plt.subplots(2, 1, figsize=(14, 8))
    axs[0].plot(t1, label='Actual', color='steelblue', lw=1.5, alpha=0.9)
    axs[0].plot(p1, label='Predicted', color='tomato', lw=1.5, alpha=0.8, ls='--')
    axs[0].legend()
    axs[0].grid(alpha=0.3)
    
    axs[1].scatter(t1, p1, alpha=0.25, s=8, color='steelblue')
    mn = min(t1.min(), p1.min()) - 1  # abs min
    mx = max(t1.max(), p1.max()) + 1  # abs max
    axs[1].plot([mn, mx], [mn, mx], 'r--', lw=2, label='Perfect prediction')
    axs[1].legend()
    axs[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(pth, dpi=150, bbox_inches='tight')
    plt.close()

def do_ex(x_sc, yt, yp, sc, tidx, nf, pth, n=5):  # plot windows
    fig, axs = plt.subplots(n, 1, figsize=(14, 4 * n))
    idx_list = np.linspace(0, len(yt) - 1, n, dtype=int)  # evenly spaced
    
    for i, idx in enumerate(idx_list):
        ax = axs[i]
        act = sc.inverse_transform(x_sc[idx])  # unscale context
        c_temp = act[-24:, tidx]  # recent day
        t12 = yt[idx]
        p12 = yp[idx]
        cx = np.arange(-24, 0)  # x past
        fx = np.arange(0, 12)  # x future
        
        ax.plot(cx, c_temp, color='steelblue', lw=2, label='context')
        ax.plot(fx, t12, color='green', lw=2, marker='o', label='actual')
        ax.plot(fx, p12, color='tomato', lw=2, marker='s', ls='--', label='pred')
        ax.legend()
        ax.grid(alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(pth, dpi=150, bbox_inches='tight')
    plt.close()

def do_horiz(yt, yp, pth):  # plot degradation
    s = np.arange(1, yt.shape[1] + 1)
    m = [mean_absolute_error(yt[:, h], yp[:, h]) for h in range(len(s))]  # mae array
    r = [np.sqrt(mean_squared_error(yt[:, h], yp[:, h])) for h in range(len(s))]  # rmse array
    
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    a1.bar(s, m, color='steelblue')
    a1.set_title('MAE')
    a2.bar(s, r, color='tomato')
    a2.set_title('RMSE')
    
    plt.tight_layout()
    plt.savefig(pth, dpi=150, bbox_inches='tight')
    plt.close()

def do_cm(l, p, nms, pth):  # plot cm
    cm = confusion_matrix(l, p, labels=TEMP_BINS)
    rs = cm.sum(axis=1, keepdims=True)  # sum rows
    rs[rs == 0] = 1  # no zero div
    pct = cm.astype(float) / rs * 100  # row percents
    
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=nms, yticklabels=nms, ax=a1)
    sns.heatmap(pct, annot=True, fmt='.1f', cmap='Greens', xticklabels=nms, yticklabels=nms, ax=a2)
    
    plt.tight_layout()
    plt.savefig(pth, dpi=150, bbox_inches='tight')
    plt.close()