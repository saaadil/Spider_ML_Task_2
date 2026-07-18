import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import classification_report

from config import CONFIG, WDIR, ODIR, TEMP_NAMES, device
from data import download_jena, load_and_downsample, prepare_data, get_dataloaders, inv_temp
from model import WeatherTransformer, ComparisonLSTM
from visualize import (plot_training_curves, plot_predictions,
                       plot_forecast_examples, plot_horizon_metrics,
                       plot_confusion_matrix, plot_comparison,
                       plot_comparison_training, bin_temperature)

def train_epoch(model, loader, criterion, optimizer, device, grad_clip):  # one pass
    model.train()  # turn on drop
    total_loss = 0.0  # reset loss
    n = 0  # reset count
    for X_b, y_b in loader:  # loop batch
        X_b = X_b.to(device, non_blocking=True)  # to gpu
        y_b = y_b.to(device, non_blocking=True)  # to gpu
        optimizer.zero_grad(set_to_none=True)  # trash grads
        preds = model(X_b)  # forward pass
        loss = criterion(preds, y_b)  # calc err
        loss.backward()  # calc math
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)  # stop exp
        optimizer.step()  # update weights
        total_loss += loss.item() * X_b.size(0)  # add err
        n += X_b.size(0)  # track n
    return total_loss / n  # ret avg

@torch.inference_mode()
def evaluate(model, loader, criterion, device):  # validate model
    model.eval()  # turn off drop
    total_loss = 0.0  # reset loss
    n = 0  # reset count
    preds_list = []  # empty pds
    labels_list = []  # empty lbls
    for X_b, y_b in loader:  # loop batch
        X_b = X_b.to(device, non_blocking=True)  # to gpu
        y_b = y_b.to(device, non_blocking=True)  # to gpu
        preds = model(X_b)  # forward pass
        loss = criterion(preds, y_b)  # calc err
        total_loss += loss.item() * X_b.size(0)  # add err
        n += X_b.size(0)  # track n
        preds_list.append(preds.cpu().numpy())  # to cpu
        labels_list.append(y_b.cpu().numpy())  # to cpu
    return (total_loss / n, np.vstack(preds_list), np.vstack(labels_list))  # ret arrays

def run_training(model, train_loader, val_loader, criterion, optimizer, scheduler, device, cfg, wdir, name):  # master loop
    history = {'train_loss': [], 'val_loss': [], 'lr': []}  # track info
    best_val_loss = float('inf')  # set worst
    best_epoch = 0  # set zero
    
    print(f"\n[{name}] Training for {cfg['num_epochs']} epochs")
    print(f"[{name}] LR: {optimizer.param_groups[0]['lr']} -> drops at {cfg['milestones']}\n")
    
    for epoch in range(1, cfg['num_epochs'] + 1):  # loop eps
        t0 = time.time()  # start time
        tr_loss = train_epoch(model, train_loader, criterion, optimizer, device, cfg['grad_clip'])  # train ep
        vl_loss, _, _ = evaluate(model, val_loader, criterion, device)  # eval ep
        current_lr = optimizer.param_groups[0]['lr']  # get lr
        scheduler.step()  # step lr
        
        history['train_loss'].append(tr_loss)  # add tl
        history['val_loss'].append(vl_loss)  # add vl
        history['lr'].append(current_lr)  # add lr
        
        if vl_loss < best_val_loss:  # if record
            best_val_loss = vl_loss  # set best
            best_epoch = epoch  # set ep
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': vl_loss,
                'config': cfg,
            }, wdir / f'best_{name}.pth')  # save weights
            
        if epoch % cfg['save_every'] == 0:  # check pt
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': vl_loss,
            }, wdir / f'checkpoint_{name}_{epoch:03d}.pth')  # save pt
            
        elapsed = time.time() - t0  # calc time
        if epoch == 1 or epoch % 5 == 0 or epoch in cfg['milestones']:  # print iter
            star = " (best)" if epoch == best_epoch else ""
            print(f"Epoch {epoch} | T-Loss: {tr_loss:.6f} | V-Loss: {vl_loss:.6f} | LR: {current_lr:.2e} | {elapsed:.1f}s{star}")
            
    torch.save(model.state_dict(), wdir / f'final_{name}_weights.pth')  # save fin
    print(f"\n[{name}] Best Val Loss: {best_val_loss:.6f} at epoch {best_epoch}")
    
    ckpt = torch.load(wdir / f'best_{name}.pth', map_location=device, weights_only=False)  # load best
    model.load_state_dict(ckpt['model_state_dict'])  # set weights
    print(f"[{name}] Loaded best model")
    
    return history, best_val_loss, best_epoch

if __name__ == '__main__':  # exec logic
    csv_path = download_jena()  # get data
    df_hourly = load_and_downsample(csv_path)  # shrink data
    
    (X_train, y_train, X_val, y_val, X_test, y_test, scaler, temp_col_idx, n_features) = prepare_data(df_hourly)  # split sets
    train_loader, val_loader, test_loader = get_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test)  # get engines
    
    trans_model = WeatherTransformer(  # init trans
        input_size=n_features,
        d_model=CONFIG['d_model'],
        n_heads=CONFIG['n_heads'],
        n_encoder_layers=CONFIG['n_encoder_layers'],
        d_ff=CONFIG['d_ff'],
        horizon=CONFIG['horizon'],
        seq_len=CONFIG['seq_len'],
        dropout=CONFIG['dropout']
    ).to(device)
    
    criterion = nn.MSELoss()  # set loss
    trans_optimizer = optim.Adam(trans_model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])  # set adam
    trans_scheduler = optim.lr_scheduler.MultiStepLR(trans_optimizer, milestones=CONFIG['milestones'], gamma=CONFIG['gamma'])  # set sched
    
    hist_trans, best_val_trans, _ = run_training(  # train trans
        trans_model, train_loader, val_loader, criterion, trans_optimizer,
        trans_scheduler, device, CONFIG, WDIR, "transformer"
    )
    
    lstm_model = ComparisonLSTM(  # init lstm
        input_size=n_features,
        hidden_size=CONFIG['lstm_hidden'],
        num_layers=CONFIG['lstm_layers'],
        horizon=CONFIG['horizon'],
        dropout=CONFIG['lstm_dropout']
    ).to(device)
    
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=CONFIG['lstm_lr'], weight_decay=CONFIG['weight_decay'])  # set adam
    lstm_scheduler = optim.lr_scheduler.MultiStepLR(lstm_optimizer, milestones=CONFIG['milestones'], gamma=CONFIG['gamma'])  # set sched
    
    hist_lstm, best_val_lstm, _ = run_training(  # train lstm
        lstm_model, train_loader, val_loader, criterion, lstm_optimizer,
        lstm_scheduler, device, CONFIG, WDIR, "lstm_baseline"
    )
    
    _, y_pred_trans_scaled, y_true_scaled = evaluate(trans_model, test_loader, criterion, device)  # eval trans
    _, y_pred_lstm_scaled, _ = evaluate(lstm_model, test_loader, criterion, device)  # eval lstm
    
    y_pred_trans = inv_temp(y_pred_trans_scaled, scaler, temp_col_idx, n_features)  # uns trans
    y_pred_lstm = inv_temp(y_pred_lstm_scaled, scaler, temp_col_idx, n_features)  # uns lstm
    y_true = inv_temp(y_true_scaled, scaler, temp_col_idx, n_features)  # uns true
    
    true_bins = bin_temperature(y_true.flatten())  # bin true
    pred_bins = bin_temperature(y_pred_trans.flatten())  # bin preds
    
    print("\nCLASSIFICATION REPORT (Transformer)")
    print(classification_report(true_bins, pred_bins, target_names=TEMP_NAMES, digits=4))  # print clf
    
    print("[Plot] Generating Visualization Suite...")
    plot_training_curves(hist_trans, ODIR / 'transformer_training_curves.png')  # plot curves
    plot_predictions(y_true, y_pred_trans, ODIR / 'transformer_predictions.png')  # plot pds
    plot_forecast_examples(X_test, y_true, y_pred_trans, scaler, temp_col_idx, n_features, ODIR / 'transformer_examples.png')  # plot ex
    plot_horizon_metrics(y_true, y_pred_trans, ODIR / 'transformer_horizon_metrics.png')  # plot horz
    plot_confusion_matrix(true_bins, pred_bins, TEMP_NAMES, ODIR / 'transformer_confusion_matrix.png')  # plot cm
    plot_comparison(y_true, y_pred_trans, y_pred_lstm, ODIR / 'model_comparison.png')  # plot comp
    plot_comparison_training(hist_trans, hist_lstm, ODIR / 'training_comparison.png')  # plot tr
    
    print("[Done] Pipeline Execution Complete.")