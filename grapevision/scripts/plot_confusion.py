"""Plot detection confusion matrices (TP/FN/FP, IoU >= 0.5) from outputs/eval_counts_*.json.

Run from grapevision/outputs: python ../scripts/plot_confusion.py
"""

import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
runs=[("full_farm","Base, full"),("sahi_farm","Base, SAHI"),("full_farm_ft","Fine-tuned, full"),("sahi_farm_ft","Fine-tuned, SAHI")]
fig,axs=plt.subplots(1,4,figsize=(16,4.2))
for ax,(k,t) in zip(axs,runs):
    d=json.load(open(f"eval_counts_{k}.json"));b=d["boxes"]
    tot=b["tp"]+b["fn"]
    m=np.array([[b["tp"],b["fn"]],[b["fp"],np.nan]])
    norm=np.array([[b["tp"]/tot,b["fn"]/tot],[0,0]])
    ax.imshow(np.nan_to_num(m),cmap="Blues",vmax=max(b["tp"],b["fn"],b["fp"],1))
    for i in range(2):
        for j in range(2):
            if i==1 and j==1: ax.text(j,i,"TN\n(n/a)",ha="center",va="center",color="gray")
            else:
                v=int(m[i,j]); ax.text(j,i,f"{v}",ha="center",va="center",fontsize=16,color="white" if v>0.55*max(b["tp"],b["fn"],b["fp"],1) else "black")
    ax.set_xticks([0,1],["Bunch","Background"]);ax.set_yticks([0,1],["Bunch","Background"])
    ax.set_xlabel("Predicted");ax.set_ylabel("Actual");ax.set_title(f'{t} (thr {d["threshold"]})\nP={b["precision"]:.3f} R={b["recall"]:.3f} F1={b["f1"]:.3f}')
fig.suptitle("Farm test set (5 images, 219 bunches): detection confusion matrices (IoU >= 0.5)")
plt.tight_layout();plt.savefig("confusion_matrix_farm.png",dpi=150)
