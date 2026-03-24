#!/usr/bin/env python3
"""DrAgnes v2 -- FDA-Grade Clinical Metrics (PPV/NPV/NNB/LR/F1/MCC/ECE/AUROC/failure modes).
Evaluates on ISIC 2019 15% stratified holdout (akinsanyaayomide, seed=42).
Usage: python scripts/clinical-metrics-analysis.py
"""
import json, time, warnings, os
from pathlib import Path
from collections import Counter
import numpy as np, torch
from PIL import Image
from sklearn.metrics import (confusion_matrix, matthews_corrcoef, f1_score,
                             roc_auc_score, roc_curve, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from datasets import load_dataset, concatenate_datasets
from transformers import ViTForImageClassification, ViTImageProcessor

warnings.filterwarnings("ignore"); os.environ["TOKENIZERS_PARALLELISM"] = "false"
SD = Path(__file__).resolve().parent
MD = SD / "dragnes-classifier-v2" / "best"
CLS = ["akiec","bcc","bkl","df","mel","nv","vasc"]
FN = {"akiec":"Actinic Keratosis","bcc":"Basal Cell Carcinoma","bkl":"Benign Keratosis",
      "df":"Dermatofibroma","mel":"Melanoma","nv":"Melanocytic Nevus","vasc":"Vascular Lesion"}
NC, MEL, BS, CANCER = 7, 4, 32, {0,1,4}
I2H = {0:0,1:1,2:2,3:3,4:4,5:5,6:0,7:6}

def img(row):
    for k in ("image","img"):
        if k in row:
            v = row[k]
            if isinstance(v,Image.Image): return v.convert("RGB")
            if isinstance(v,dict) and "bytes" in v:
                import io; return Image.open(io.BytesIO(v["bytes"])).convert("RGB")
            return v.convert("RGB")
    raise ValueError(f"no image: {list(row.keys())}")

def op_points(yb, yp, specs, senss):
    fpr,tpr,_ = roc_curve(yb,yp); sp = 1-fpr; d = {}
    for s in specs:
        m = sp>=s; d[f"sens@{int(s*100)}%spec"] = round(float(tpr[m].max()),4) if m.any() else None
    for s in senss:
        m = tpr>=s; d[f"spec@{int(s*100)}%sens"] = round(float(sp[m].max()),4) if m.any() else None
    return d

def main():
    t0 = time.time()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[1/5] Loading model on {dev}")
    model = ViTForImageClassification.from_pretrained(str(MD)).to(dev).eval()
    proc = ViTImageProcessor.from_pretrained(str(MD))
    print("[2/5] Loading ISIC 2019 + reproducing holdout (seed=42, 15%)")
    ds = load_dataset("akinsanyaayomide/skin_cancer_dataset_balanced_labels_2")
    full = concatenate_datasets([ds[s] for s in ds])
    labels = [I2H.get(int(r),-1) for r in full["label"]]
    _,tidx = train_test_split(range(len(full)), test_size=0.15, random_state=42, stratify=labels)
    items = [(i,labels[i]) for i in tidx if labels[i]>=0]
    yt = np.array([l for _,l in items])
    print(f"   N={len(items)}  dist={dict(Counter(yt))}")
    print(f"[3/5] Inference (batch={BS}) ...")
    probs = []
    for s in range(0, len(items), BS):
        b = items[s:s+BS]
        inp = proc(images=[img(full[i]) for i,_ in b], return_tensors="pt").to(dev)
        with torch.no_grad(): lo = model(**inp).logits
        probs.append(torch.softmax(lo,dim=-1).cpu().numpy())
        if (s//BS)%30==0: print(f"   {s+len(b)}/{len(items)}",end="\r")
    yp = np.concatenate(probs); ypred = yp.argmax(1)
    print(f"   Done in {time.time()-t0:.0f}s")
    cm = confusion_matrix(yt,ypred,labels=list(range(NC)))
    pr,rc,f1,sup = precision_recall_fscore_support(yt,ypred,labels=list(range(NC)),zero_division=0)
    mcc = round(float(matthews_corrcoef(yt,ypred)),4)
    yb = label_binarize(yt,classes=list(range(NC)))
    print("[4/5] Computing metrics ...")
    pc = {}
    for c in range(NC):
        tp=int(cm[c,c]); fn_=int(cm[c].sum()-tp); fp=int(cm[:,c].sum()-tp); tn=int(cm.sum()-tp-fn_-fp)
        se=tp/(tp+fn_) if tp+fn_ else 0; sp=tn/(tn+fp) if tn+fp else 0
        ppv=tp/(tp+fp) if tp+fp else 0; npv=tn/(tn+fn_) if tn+fn_ else 0
        lrp=se/(1-sp) if sp<1 else float("inf"); lrn=(1-se)/sp if sp>0 else float("inf")
        auc = round(float(roc_auc_score(yb[:,c],yp[:,c])),4) if yb[:,c].sum() not in (0,len(yb)) else None
        e = {"full_name":FN[CLS[c]],"support":int(sup[c]),"sensitivity":round(se,4),"specificity":round(sp,4),
             "PPV":round(ppv,4),"NPV":round(npv,4),"F1":round(float(f1[c]),4),
             "LR+":round(lrp,4) if lrp!=float("inf") else "inf",
             "LR-":round(lrn,4) if lrn!=float("inf") else "inf","AUROC":auc}
        if c in CANCER: e["NNB"] = round(1/ppv,2) if ppv>0 else None
        if c==MEL: e.update(op_points(yb[:,c],yp[:,c],[.85,.90,.95],[.90,.95,.98]))
        pc[CLS[c]] = e
    ov = {"MCC":mcc,"weighted_F1":round(float(f1_score(yt,ypred,average="weighted")),4),
          "macro_F1":round(float(f1_score(yt,ypred,average="macro")),4),
          "macro_AUROC":round(float(roc_auc_score(yb,yp,average="macro",multi_class="ovr")),4),
          "weighted_AUROC":round(float(roc_auc_score(yb,yp,average="weighted",multi_class="ovr")),4)}
    # Calibration ECE
    mx = yp.max(1); cor = (ypred==yt).astype(float); edges = np.linspace(0,1,11); ece=0.0; cb=[]
    for b in range(10):
        lo,hi = edges[b],edges[b+1]
        m = (mx>lo)&(mx<=hi) if b else (mx>=lo)&(mx<=hi); n=m.sum()
        if not n: cb.append({"bin":f"{lo:.1f}-{hi:.1f}","n":0}); continue
        ac,cn = float(cor[m].mean()),float(mx[m].mean()); ece+=abs(ac-cn)*n
        cb.append({"bin":f"{lo:.1f}-{hi:.1f}","n":int(n),"conf":round(cn,4),"acc":round(ac,4)})
    ece/=len(yt)
    # Failure analysis
    mm = yt==MEL; missed = mm&(ypred!=MEL); nm = int(missed.sum()); fails=[]
    if nm:
        for mi in np.where(missed)[0]:
            fails.append({"predicted":CLS[ypred[mi]],"mel_conf":round(float(yp[mi,MEL]),4),
                "pred_conf":round(float(yp[mi,ypred[mi]]),4),
                "top3":[{"cls":CLS[j],"p":round(float(yp[mi,j]),4)} for j in np.argsort(yp[mi])[::-1][:3]]})
    md_ = {CLS[k]:int(v) for k,v in Counter(ypred[missed]).items()}
    fa = {"total_mel":int(mm.sum()),"missed":nm,"miss_rate":round(nm/mm.sum(),4) if mm.sum() else 0,
          "misclassified_as":md_,"details":sorted(fails,key=lambda x:x["mel_conf"],reverse=True)}
    R = {"model":str(MD),"N":len(items),"classes":CLS,"per_class":pc,"overall":ov,
         "calibration":{"ECE":round(ece,4),"bins":cb},"confusion_matrix":{"labels":CLS,"matrix":cm.tolist()},
         "failure_analysis_melanoma":fa}
    OUT = SD/"clinical-metrics-full.json"
    print(f"[5/5] Saving {OUT}")
    with open(OUT,"w") as f: json.dump(R,f,indent=2,default=str)
    # ── Clinical Report ──────────────────────────────────────────────
    el = time.time()-t0
    P = lambda s: print(s)
    P("\n"+"="*72); P("  DrAgnes v2 -- CLINICAL PERFORMANCE REPORT")
    P(f"  ISIC 2019 External Holdout  |  N = {len(items):,}"); P("="*72)
    P(f"\n{'Cls':<7} {'Name':<26} {'Sens':>6} {'Spec':>6} {'PPV':>6} {'NPV':>6} {'F1':>6} {'AUC':>6} {'N':>5}")
    P("-"*72)
    for c in range(NC):
        e=pc[CLS[c]]
        P(f"{CLS[c]:<7} {FN[CLS[c]]:<26} {e['sensitivity']:>6.1%} {e['specificity']:>6.1%} "
          f"{e['PPV']:>6.1%} {e['NPV']:>6.1%} {e['F1']:>6.3f} {e['AUROC'] or 0:>6.3f} {e['support']:>5}")
    P("-"*72)
    P(f"{'Overall':<35} MCC={ov['MCC']:.3f}  wF1={ov['weighted_F1']:.3f}  macAUC={ov['macro_AUROC']:.3f}")
    P("\n  Number Needed to Biopsy:")
    for c in sorted(CANCER):
        n=pc[CLS[c]].get("NNB"); P(f"    {CLS[c]}: {n:.1f}" if n else f"    {CLS[c]}: N/A")
    me=pc["mel"]
    P(f"\n  Melanoma Likelihood Ratios:  LR+ = {me['LR+']}  (>10=strong rule-in)   LR- = {me['LR-']}  (<0.1=strong rule-out)")
    P("  Melanoma Operating Points:")
    for k in sorted(k for k in me if "@" in k): P(f"    {k}: {me[k]}")
    P(f"\n  Calibration ECE = {ece:.4f}")
    P(f"  {'Bin':<10} {'N':>6} {'Conf':>7} {'Acc':>7} {'Gap':>7}")
    for b in cb:
        if b["n"]>0: P(f"  {b['bin']:<10} {b['n']:>6} {b['conf']:>7.3f} {b['acc']:>7.3f} {abs(b['acc']-b['conf']):>7.3f}")
    P(f"\n  Confusion Matrix (rows=true, cols=pred):")
    P(f"  {'':>7}"+"".join(f"{c:>7}" for c in CLS))
    for i in range(NC): P(f"  {CLS[i]:>7}"+"".join(f"{cm[i,j]:>7}" for j in range(NC)))
    P(f"\n  Melanoma Failure Analysis: {fa['missed']}/{fa['total_mel']} missed ({fa['miss_rate']:.1%})")
    P(f"  Misclassified as: {fa['misclassified_as']}")
    for d in fa["details"][:5]:
        P(f"    -> pred {d['predicted']} ({d['pred_conf']:.3f}), mel_conf={d['mel_conf']:.3f}")
    P(f"\n  Completed in {el:.0f}s  |  Results: {OUT}"); P("="*72)

if __name__=="__main__": main()
