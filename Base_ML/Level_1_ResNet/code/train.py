import os
import platform
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report

from model import myResNet
from dataset import get_dataloaders, classes
from utils import plt_c, plt_cm
from config import my_conf, w_dir, OutDir, data_path, device, seed_val

def set_seed(s):  # set seeds
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    np.random.seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(seed_val)  # apply seeds

def train_e(mod, ldr, crit, opt, dev):  # train pass
    mod.train()  # turn on train
    tot_l = 0.0  # reset loss
    corr = 0  # reset acc
    n = 0  # reset count
    
    loop = tqdm(ldr, leave=False, desc="Training")  # loader loop
    for ix, lx in loop:  # batch loop
        ix = ix.to(dev, non_blocking=True)  # imgs to gpu
        lx = lx.to(dev, non_blocking=True)  # labels to gpu
        
        opt.zero_grad(set_to_none=True)  # reset grads
        
        lg = mod(ix)  # forward pass
        l = crit(lg, lx)  # loss calc
        l.backward()  # do math
        opt.step()  # update weights
        
        tot_l += l.item() * ix.size(0)  # add loss
        corr += lg.argmax(dim=1).eq(lx).sum().item()  # add acc
        n += ix.size(0)  # add count
        
        loop.set_postfix(loss=l.item())  # show loss
        
    return tot_l / n, 100.0 * corr / n  # return metrics

def eval_m(mod, ldr, crit, dev):  # eval pass
    mod.eval()  # turn on eval
    tot_l = 0.0  # reset loss
    corr = 0  # reset acc
    n = 0  # reset count
    p_lst = []  # empty preds
    l_lst = []  # empty labels
    
    with torch.inference_mode():  # no grad math
        for ix, lx in ldr:  # batch loop
            ix = ix.to(dev, non_blocking=True)  # imgs to gpu
            lx = lx.to(dev, non_blocking=True)  # labels to gpu
            
            lg = mod(ix)  # forward pass
            l = crit(lg, lx)  # loss calc
            
            tot_l += l.item() * ix.size(0)  # add loss
            pr = lg.argmax(dim=1)  # get pred
            corr += pr.eq(lx).sum().item()  # add acc
            n += ix.size(0)  # add count
            
            p_lst.extend(pr.cpu().numpy())  # append preds
            l_lst.extend(lx.cpu().numpy())  # append labels
            
    return (tot_l / n, 100.0 * corr / n, np.array(p_lst), np.array(l_lst))  # return arrays

def main():  # main logic
    tr_l, v_l, te_l = get_dataloaders(my_conf, data_path, seed_val)  # get loaders
    
    net = myResNet(
        s_chans=my_conf['stage_channels'], 
        blks=my_conf['blocks_per_stage'], 
        n_cls=my_conf['num_classes']
    ).to(device)  # to gpu
    
    loss_f = nn.CrossEntropyLoss(label_smoothing=my_conf['label_smooth'])  # set loss
    op = optim.Adam(net.parameters(), lr=my_conf['lr'], weight_decay=my_conf['weight_decay'])  # set optim
    sched = optim.lr_scheduler.MultiStepLR(op, milestones=my_conf['milestones'], gamma=my_conf['gamma'])  # set sched
    
    hist = {'tr_l': [], 'v_l': [], 'tr_a': [], 'v_a': [], 'lr': []}  # empty dict
    b_acc = 0.0  # bad acc
    b_ep = 0  # bad ep
    
    print(f"Starting training on {device}...")  # start string
    
    for e in range(1, my_conf['num_epochs'] + 1):  # loop eps
        tl, ta = train_e(net, tr_l, loss_f, op, device)  # train ep
        vl, va, _, _ = eval_m(net, v_l, loss_f, device)  # eval ep
        
        c_lr = op.param_groups[0]['lr']  # get curr lr
        sched.step()  # step lr
        
        hist['tr_l'].append(tl)  # add tl
        hist['v_l'].append(vl)  # add vl
        hist['tr_a'].append(ta)  # add ta
        hist['v_a'].append(va)  # add va
        hist['lr'].append(c_lr)  # add lr
        
        if va > b_acc:  # if best
            b_acc = va  # save best
            b_ep = e  # save best
            torch.save({
                'epoch': e,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': op.state_dict(),
                'val_acc': va,
                'config': my_conf
            }, w_dir / 'best_model.pth')  # save dict
            
        if e % my_conf['save_every'] == 0:  # checkpoint logic
            torch.save(net.state_dict(), w_dir / f'checkpoint_epoch_{e:03d}.pth')  # save chkpnt
            
        print(f"Epoch {e}/{my_conf['num_epochs']} | Train Loss: {tl:.4f} | Val Acc: {va:.2f}% | LR: {c_lr:.1e}")  # print pass
        
    torch.save(net.state_dict(), w_dir / 'final_model_weights.pth')  # save final
    
    cp = torch.load(w_dir / 'best_model.pth', map_location=device, weights_only=False)  # load best
    net.load_state_dict(cp['model_state_dict'])  # load weights
    
    te_l, te_a, te_p, te_t = eval_m(net, te_l, loss_f, device)  # test eval
    
    print("\n--- Final Test Results ---")  # start string
    print(f"Loss: {te_l:.4f}")  # print tl
    print(f"Accuracy: {te_a:.2f}%")  # print ta
    print(classification_report(te_t, te_p, target_names=classes, digits=4))  # print clf
    
    plt_c(hist, OutDir / 'training_curves.png')  # plot curves
    plt_cm(te_t, te_p, classes, OutDir / 'confusion_matrix.png')  # plot cm

if __name__ == '__main__':
    main() # run file