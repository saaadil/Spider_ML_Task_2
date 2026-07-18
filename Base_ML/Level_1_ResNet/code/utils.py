import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

def plt_c(h, p):  # plot loss curves
    eps = range(1, len(h['tr_l']) + 1)  # get epochs
    b_ep = int(np.argmax(h['v_a'])) + 1  # get best
    
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))  # canvas
    
    a1.plot(eps, h['tr_l'], label='Train', color='steelblue', lw=2)  # plot tl
    a1.plot(eps, h['v_l'], label='Val', color='tomato', lw=2)  # plot vl
    a1.set_xlabel('Epoch')
    a1.set_ylabel('Loss')
    a1.legend()
    a1.grid(alpha=0.3)
    
    a2.plot(eps, h['tr_a'], label='Train', color='steelblue', lw=2)  # plot ta
    a2.plot(eps, h['v_a'], label='Val', color='tomato', lw=2)  # plot va
    a2.axvline(b_ep, color='green', ls='--', alpha=0.8, lw=1.5)  # draw line
    a2.set_xlabel('Epoch')
    a2.set_ylabel('Accuracy (%)')
    a2.legend()
    a2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(p, dpi=150, bbox_inches='tight')  # save png
    plt.close()

def plt_cm(lbl, pds, nms, p):  # plot conf matrix
    c = confusion_matrix(lbl, pds)  # gen matrix
    c_p = c.astype(float) / c.sum(axis=1, keepdims=True) * 100  # gen percents
    
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(18, 7))  # canvas
    
    sns.heatmap(c, annot=True, fmt='d', cmap='Blues', xticklabels=nms, yticklabels=nms, ax=a1)  # draw ints
    sns.heatmap(c_p, annot=True, fmt='.1f', cmap='Greens', xticklabels=nms, yticklabels=nms, ax=a2)  # draw floats
    
    plt.tight_layout()
    plt.savefig(p, dpi=150, bbox_inches='tight')  # save png
    plt.close()