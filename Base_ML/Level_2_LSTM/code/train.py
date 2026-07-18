import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import classification_report

from config import CONFIG, WDIR, ODIR, TEMP_LABELS, device
from data import get_data, load_data, prep_data, build_loaders, invert_temp
from model import WxModel
from visualize import categorize_temp, do_plots, do_preds, do_ex, do_horiz, do_cm

def train_one_epoch(mod, ldr, crit, opt, dev, clip):  # one pass
    mod.train()  # dropout on
    tot_l = 0.0
    c = 0
    for bx, by in ldr:  # loop batches
        bx = bx.to(dev, non_blocking=True)  # to gpu
        by = by.to(dev, non_blocking=True)
        opt.zero_grad(set_to_none=True)  # trash grads
        p = mod(bx)  # forward pass
        l = crit(p, by)  # calc error
        l.backward()  # do math
        nn.utils.clip_grad_norm_(mod.parameters(), max_norm=clip)  # stop explosion
        opt.step()  # update weights
        tot_l += l.item() * bx.size(0)  # add error
        c += bx.size(0)
    return tot_l / c

def eval_model(mod, ldr, crit, dev):  # validate only
    mod.eval()  # dropout off
    tot_l = 0.0
    c = 0
    plist = []
    tlist = []
    with torch.no_grad():  # no grad
        for bx, by in ldr:
            bx = bx.to(dev, non_blocking=True)
            by = by.to(dev, non_blocking=True)
            p = mod(bx)
            l = crit(p, by)
            tot_l += l.item() * bx.size(0)
            c += bx.size(0)
            plist.append(p.cpu().numpy())  # to cpu
            tlist.append(by.cpu().numpy())
    return (tot_l / c, np.vstack(plist), np.vstack(tlist))

def main():  # main loop
    c_path = get_data()  # get data
    df = load_data(c_path)  # clean data
    
    (xtr, ytr, xv, yv, xte, yte, sc, tidx, nf) = prep_data(df)
    trl, vl, tel = build_loaders(xtr, ytr, xv, yv, xte, yte)  # build loaders
    
    net = WxModel(  # init model
        inp=nf,
        hid=CONFIG['lstm_hidden'],
        n_lay=CONFIG['lstm_layers'],
        horiz=CONFIG['horizon'],
        drp=CONFIG['lstm_dropout']
    ).to(device)
    
    loss_fn = nn.MSELoss()  # mse loss
    op = optim.Adam(net.parameters(), lr=CONFIG['lstm_lr'], weight_decay=CONFIG['weight_decay'])  # adam opt
    sched = optim.lr_scheduler.MultiStepLR(op, milestones=CONFIG['milestones'], gamma=CONFIG['gamma'])  # drop lr
    
    hist = {'tr': [], 'val': []}  # track metrics
    b_val = float('inf')
    
    print("training starting")
    for e in range(1, CONFIG['num_epochs'] + 1):  # loop epochs
        tl = train_one_epoch(net, trl, loss_fn, op, device, CONFIG['grad_clip'])  # train epoch
        vl_loss, _, _ = eval_model(net, vl, loss_fn, device)  # eval val
        hist['tr'].append(tl)
        hist['val'].append(vl_loss)
        sched.step()  # update lr
        
        if vl_loss < b_val:  # new record
            b_val = vl_loss
            torch.save({  # save weights
                'epoch': e,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': op.state_dict(),
                'val_loss': vl_loss,
            }, WDIR / 'best_lstm_model.pth')
            s = " (best)"
        else:
            s = ""
            
        if e % 5 == 0 or e == 1:
            print(f"Epoch {e}/{CONFIG['num_epochs']} | tr: {tl:.4f} | val: {vl_loss:.4f}{s}")
            
    cp = torch.load(WDIR / 'best_lstm_model.pth', map_location=device, weights_only=False)  # load best
    net.load_state_dict(cp['model_state_dict'])
    
    t_loss, p_sc, t_sc = eval_model(net, tel, loss_fn, device)  # test set
    p_raw = invert_temp(p_sc, sc, tidx, nf)  # unscale preds
    t_raw = invert_temp(t_sc, sc, tidx, nf)  # unscale true
    
    tbins = categorize_temp(t_raw.flatten())  # bin true
    pbins = categorize_temp(p_raw.flatten())  # bin preds
    
    print("\nREPORT\n")
    print(classification_report(tbins, pbins, target_names=TEMP_LABELS, digits=4))
    
    print("plotting...")
    do_plots(hist, ODIR / 'lstm_training_curves.png')  # loss curves
    do_preds(t_raw, p_raw, ODIR / 'lstm_predictions.png')
    do_ex(xte, t_raw, p_raw, sc, tidx, nf, ODIR / 'lstm_examples.png')
    do_horiz(t_raw, p_raw, ODIR / 'lstm_horizon_metrics.png')
    do_cm(tbins, pbins, TEMP_LABELS, ODIR / 'lstm_confusion_matrix.png')

if __name__ == '__main__':
    main()