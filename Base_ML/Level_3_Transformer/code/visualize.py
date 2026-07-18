import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix
from sklearn.preprocessing import StandardScaler

from config import CONFIG, TEMP_BINS, TEMP_LABELS

def bin_temperature(temps: np.ndarray) -> np.ndarray:  # cat temps
    b = np.zeros(temps.shape, dtype=int)  # zero arr
    b[temps < 0] = 0  # v cold
    b[(temps >= 0) & (temps < 10)] = 1  # cold
    b[(temps >= 10) & (temps < 20)] = 2  # mild
    b[temps >= 20] = 3  # warm
    return b  # ret bins

def plot_training_curves(history: dict, path: Path):  # plot loss
    epochs = range(1, len(history['train_loss']) + 1) # get eps
    best_e = int(np.argmin(history['val_loss'])) + 1  # get best
    best_v = min(history['val_loss']) # get min
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))  # canvas
    ax1.plot(epochs, history['train_loss'], label='Train', color='steelblue', lw=2)  # plot tr
    ax1.plot(epochs, history['val_loss'], label='Val', color='tomato', lw=2)  # plot vl
    for m in CONFIG['milestones']:  # loop drps
        ax1.axvline(m, color='gray', ls=':', lw=1, alpha=0.6)  # draw line
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2.plot(epochs, history['val_loss'], color='tomato', lw=2, label='Val MSE')  # plot vl
    for m in CONFIG['milestones']:  # loop drps
        ax2.axvline(m, color='gray', ls=':', lw=1, alpha=0.6)  # draw line
    ax2.axvline(best_e, color='green', ls='--', alpha=0.8, lw=1.5)  # draw best
    ax2.annotate(f'Best: {best_v:.5f}\n@ epoch {best_e}', xy=(best_e, best_v), xytext=(best_e + 2, best_v * 1.5), fontsize=9, color='green')  # add txt
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Val MSE Loss')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')  # save png
    plt.close()

def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, path: Path, n_show: int = 500):  # plot pds
    true_h1 = y_true[:n_show, 0]  # get t h1
    pred_h1 = y_pred[:n_show, 0]  # get p h1
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))  # canvas
    axes[0].plot(true_h1, label='Actual', color='steelblue', lw=1.5, alpha=0.9)  # plot t
    axes[0].plot(pred_h1, label='Predicted', color='tomato', lw=1.5, alpha=0.8, ls='--')  # plot p
    axes[0].set_xlabel('Time Step (hours)')
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].scatter(true_h1, pred_h1, alpha=0.25, s=8, color='steelblue')  # plot sct
    mn = min(true_h1.min(), pred_h1.min()) - 1  # get min
    mx = max(true_h1.max(), pred_h1.max()) + 1  # get max
    axes[1].plot([mn, mx], [mn, mx], 'r--', lw=2, label='Perfect prediction')  # draw line
    axes[1].set_xlabel('Actual (°C)')
    axes[1].set_ylabel('Predicted (°C)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')  # save png
    plt.close()

def plot_forecast_examples(X_test_scaled: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, scaler: StandardScaler, temp_idx: int, n_feat: int, path: Path, n: int = 5):  # plot ex
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n))  # canvas
    indices = np.linspace(0, len(y_true) - 1, n, dtype=int)  # get idcs
    
    for k, idx in enumerate(indices):  # loop idx
        ax = axes[k]
        x_actual = scaler.inverse_transform(X_test_scaled[idx])  # uns ctx
        context_temp = x_actual[-72:, temp_idx] # get ctx
        true_24h = y_true[idx]  # get t
        pred_24h = y_pred[idx]  # get p
        context_x = np.arange(-72, 0)  # x past
        forecast_x = np.arange(0, 24)  # x fut
        
        ax.plot(context_x, context_temp, color='steelblue', lw=2, label='Input context')  # draw ctx
        ax.axvline(0, color='black', ls=':', lw=1.2, alpha=0.6)  # draw line
        ax.plot(forecast_x, true_24h, color='green', lw=2, marker='o', label='Actual')  # draw t
        ax.plot(forecast_x, pred_24h, color='tomato', lw=2, marker='s', ls='--', label='Predicted')  # draw p
        
        mae_ex = np.mean(np.abs(true_24h - pred_24h))  # calc mae
        ax.set_title(f'Example {k+1} — Forecast MAE: {mae_ex:.2f}°C')
        ax.set_ylabel('Temperature (°C)')
        ax.legend(loc='best', fontsize=8)
        ax.grid(alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')  # save png
    plt.close()

def plot_horizon_metrics(y_true: np.ndarray, y_pred: np.ndarray, path: Path):  # plot hm
    steps = np.arange(1, y_true.shape[1] + 1)  # get steps
    maes = [mean_absolute_error(y_true[:, h], y_pred[:, h]) for h in range(len(steps))]  # get mae
    rmses = [np.sqrt(mean_squared_error(y_true[:, h], y_pred[:, h])) for h in range(len(steps))]  # get rmse
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))  # canvas
    ax1.bar(steps, maes, color='steelblue')  # draw mae
    ax1.set_xlabel('Hours Ahead')
    ax1.set_ylabel('MAE (°C)')
    ax1.grid(alpha=0.3, axis='y')
    
    ax2.bar(steps, rmses, color='tomato')  # draw rmse
    ax2.set_xlabel('Hours Ahead')
    ax2.set_ylabel('RMSE (°C)')
    ax2.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')  # save png
    plt.close()

def plot_confusion_matrix(labels: np.ndarray, preds: np.ndarray, class_names: list, path: Path):  # plot cm
    cm = confusion_matrix(labels, preds, labels=TEMP_BINS)  # get ints
    row_sums = cm.sum(axis=1, keepdims=True)  # get sums
    row_sums[row_sums == 0] = 1  # stop 0
    cm_pct = cm.astype(float) / row_sums * 100  # get pct
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))  # canvas
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax1)  # draw ints
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Greens', xticklabels=class_names, yticklabels=class_names, ax=ax2)  # draw flt
    
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')  # save png
    plt.close()

def plot_comparison(y_true: np.ndarray, y_pred_trans: np.ndarray, y_pred_lstm: np.ndarray, path: Path, n_show: int = 200):  # plot comp
    true_h1 = y_true[:n_show, 0]  # get t
    trans_h1 = y_pred_trans[:n_show, 0]  # get tr
    lstm_h1 = y_pred_lstm[:n_show, 0]  # get ls
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))  # canvas
    axes[0].plot(true_h1, label='Actual', color='black', lw=1.8, alpha=0.9)  # plot t
    axes[0].plot(trans_h1, label='Transformer', color='steelblue', lw=1.5, ls='--')  # plot tr
    axes[0].plot(lstm_h1, label='LSTM', color='tomato', lw=1.5, ls=':')  # plot ls
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(np.abs(trans_h1 - true_h1), label='Trans Error', color='steelblue', lw=1.5)  # err tr
    axes[1].plot(np.abs(lstm_h1 - true_h1), label='LSTM Error', color='tomato', lw=1.5)  # err ls
    axes[1].set_ylabel('|Error| (°C)')
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')  # save png
    plt.close()

def plot_comparison_training(history_t: dict, history_l: dict, path: Path):  # plot trc
    ep_t = range(1, len(history_t['val_loss']) + 1)  # get t ep
    ep_l = range(1, len(history_l['val_loss']) + 1)  # get l ep
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))  # canvas
    ax1.plot(ep_t, history_t['train_loss'], label='Trans Train', color='steelblue', lw=2)  # plot tr t
    ax1.plot(ep_t, history_t['val_loss'], label='Trans Val', color='steelblue', lw=2, ls='--')  # plot vl t
    ax1.plot(ep_l, history_l['train_loss'], label='LSTM Train', color='tomato', lw=2)  # plot tr l
    ax1.plot(ep_l, history_l['val_loss'], label='LSTM Val', color='tomato', lw=2, ls='--')  # plot vl l
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2.plot(ep_t, history_t['val_loss'], label='Trans Val', color='steelblue', lw=2)  # plot vl t
    ax2.plot(ep_l, history_l['val_loss'], label='LSTM Val', color='tomato', lw=2)  # plot vl l
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Val MSE Loss')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')  # save png
    plt.close()