#!/usr/bin/env python3
"""Dondurulmuş embedding'ler üzerine MLP eğitimi ve değerlendirme (çok etiketli).

Her iki model için aynı hiperparametreler ve aynı seed'ler kullanılır (adil karşılaştırma).
Birden fazla seed verilirse sonuçlar ortalama ± std olarak raporlanır, ayrıca modeller
arasındaki fark seed başına eşleştirilerek verilir.
Metrikler: macro ROC-AUC, macro PR-AUC, macro F1 (eşikler validation üzerinde etiket
başına seçilir, test'e uygulanır). Üç metrik de aynı etiket kümesi üzerinden, yani test'te en az
bir pozitifi olan etiketler için hesaplanır. Ek olarak rastgele baseline PR-AUC (ortalama
prevalence) ve kosinüs benzerliğiyle komşu getirme kalitesi (P@5, P@10) raporlanır.
--model concat: PANNs ve VGGish embedding'lerini track_id üzerinden birleştirir.

Örnek:
    python src/train.py --emb-dir /kaggle/input/musicrec-emb --model panns vggish --seeds 1 2 3 4 5
"""
import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score

from common import SPLITS, read_split, set_seed


def resolve_emb_dir(p: str) -> Path:
    p = Path(p)
    for cand in (p, p / "embeddings"):
        if (cand / "splits").exists():
            return cand
    raise SystemExit(f"{p} altında splits/ klasörü bulunamadı (pipeline.py çıktısını ver).")


def load_embeddings(emb_dir: Path, model: str, layers=None):
    """Eski (şarkı başına vektör) ve pencere bazlı (n_win var) .npz biçimlerini okur.
    Pencere bazlı biçimde pencereler şarkı başına ortalanır; mert gibi (W, katman, boyut) dizilerde
    --layers ile seçilen katmanların ortalaması alınır (verilmezse tüm katmanların)."""
    files = sorted((emb_dir / model).glob("*.npz"))
    if not files:
        raise SystemExit(f"{emb_dir / model} içinde .npz yok.")
    ids, xs = [], []
    for f in files:
        z = np.load(f)
        ids.append(z["ids"])
        e = z["emb"]
        if "n_win" in z.files:
            nw = z["n_win"].astype(np.int64)
            if e.ndim == 3:
                e = e[:, layers].mean(1) if layers else e.mean(1)
            starts = np.concatenate([[0], np.cumsum(nw)[:-1]])
            e = np.add.reduceat(e.astype(np.float32), starts, axis=0) / nw[:, None]
        xs.append(e.astype(np.float32))
    return np.concatenate(ids), np.concatenate(xs)


def load_combined(emb_dir: Path, model: str, layers=None):
    """model: tek model adı (panns, vggish, mert, clap), 'a+b+c' biçiminde birleşim (ör. panns+vggish+mert)
    veya 'concat' (= panns+vggish). Birleşimde embedding'ler ortak track_id'ler üzerinden yan yana eklenir."""
    if model == "concat":
        model = "panns+vggish"
    parts = model.split("+")
    ids, x = load_embeddings(emb_dir, parts[0], layers)
    for name in parts[1:]:
        ids_n, x_n = load_embeddings(emb_dir, name, layers)
        pos = {int(t): i for i, t in enumerate(ids_n)}
        keep = [i for i, t in enumerate(ids) if int(t) in pos]
        ids = ids[keep]
        x = np.concatenate([x[keep], x_n[[pos[int(t)] for t in ids]]], axis=1)
    return ids, x


def retrieval_eval(x, y, ks=(5, 10)):
    """Kosinüs benzerliğiyle komşu getirme: en yakın k komşudan kaçı sorguyla en az bir etiket paylaşıyor (P@k).

    p_at_k_random: komşuyu rastgele seçseydik beklenen oran (baseline). Etiketsiz sorgular dışarıda bırakılır.
    """
    y = y.astype(np.float32)
    has = y.sum(1) > 0
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    share = (y @ y.T) > 0
    np.fill_diagonal(share, False)
    order = np.argsort(-sim, axis=1)[:, :max(ks)]
    out = {}
    for k in ks:
        hit = np.take_along_axis(share, order[:, :k], axis=1).mean(1)
        out[f"p_at_{k}"] = float(hit[has].mean())
    out["p_at_k_random"] = float(share[has].sum(1).mean() / (len(y) - 1))
    return out


def make_xy(rows, id2idx, X, tag2idx):
    keep = [r for r in rows if r["track_id"] in id2idx]
    x = X[[id2idx[r["track_id"]] for r in keep]]
    y = np.zeros((len(keep), len(tag2idx)), dtype=np.float32)
    for i, r in enumerate(keep):
        for t in r["tags"]:
            if t in tag2idx:
                y[i, tag2idx[t]] = 1.0
    return x, y


def valid_mask(y, min_pos=1):
    """Test'te en az min_pos pozitifi olan (ve hepsi pozitif olmayan) etiketler."""
    pos = y.sum(0)
    return (pos >= min_pos) & (pos < len(y))


def macro_metrics(y, p, min_pos=1):
    valid = valid_mask(y, min_pos)
    roc = roc_auc_score(y[:, valid], p[:, valid], average="macro")
    pr = average_precision_score(y[:, valid], p[:, valid], average="macro")
    return float(roc), float(pr)


def best_thresholds(y, p):
    thr = np.full(y.shape[1], 0.5)
    for j in range(y.shape[1]):
        if y[:, j].sum() == 0:
            continue
        prec, rec, th = precision_recall_curve(y[:, j], p[:, j])
        if len(th) == 0:
            continue
        f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
        thr[j] = th[int(np.argmax(f1))]
    return thr


@torch.no_grad()
def predict(net, x, device, bs=2048):
    net.eval()
    out = []
    for i in range(0, len(x), bs):
        out.append(torch.sigmoid(net(x[i:i + bs].to(device))).cpu())
    return torch.cat(out).numpy()


def prepare(model_name, args, device):
    emb_dir = resolve_emb_dir(args.emb_dir)
    vocab = json.loads((emb_dir / "vocab.json").read_text())
    tag2idx = {t: i for i, t in enumerate(vocab)}
    ids, X = load_combined(emb_dir, model_name, args.layers)
    id2idx = {int(t): i for i, t in enumerate(ids)}

    data = {}
    for s in SPLITS:
        rows = read_split(emb_dir / "splits" / f"{s}.tsv")
        data[s] = make_xy(rows, id2idx, X, tag2idx)
        print(f"[{model_name}] {s}: {len(data[s][0])}/{len(rows)} şarkı embedding'e sahip")

    mean = data["train"][0].mean(0, keepdims=True)
    std = data["train"][0].std(0, keepdims=True) + 1e-6
    xt = {s: torch.tensor((data[s][0] - mean) / std, dtype=torch.float32) for s in SPLITS}
    yt = {s: data[s][1] for s in SPLITS}
    return {"vocab": vocab, "xt": xt, "yt": yt, "mean": mean, "std": std}


def train_one(model_name, prep, args, device, seed, save_dir=None):
    set_seed(seed)
    vocab, xt, yt = prep["vocab"], prep["xt"], prep["yt"]
    d, n_tags = xt["train"].shape[1], len(vocab)
    net = nn.Sequential(
        nn.Linear(d, args.hidden), nn.ReLU(), nn.Dropout(args.dropout), nn.Linear(args.hidden, n_tags)
    ).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    pos_w = None
    if args.pos_weight:
        pos = yt["train"].sum(0)
        neg = len(yt["train"]) - pos
        w = np.clip(neg / np.maximum(pos, 1), 1.0, args.pos_weight_max)
        pos_w = torch.tensor(w, dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    x_train = xt["train"].to(device)
    y_train = torch.tensor(yt["train"], device=device)
    g = torch.Generator().manual_seed(seed)
    best_pr, best_state, bad, best_epoch = -1.0, None, 0, 0
    for epoch in range(1, args.epochs + 1):
        net.train()
        perm = torch.randperm(len(x_train), generator=g).to(device)
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size]
            opt.zero_grad()
            loss = loss_fn(net(x_train[idx]), y_train[idx])
            loss.backward()
            opt.step()
        roc, pr = macro_metrics(yt["validation"], predict(net, xt["validation"], device))
        if pr > best_pr:
            best_pr, best_epoch, bad = pr, epoch, 0
            best_state = copy.deepcopy(net.state_dict())
        else:
            bad += 1
        if epoch % 10 == 0 or epoch == 1:
            print(f"[{model_name} seed {seed}] epoch {epoch:3d} val ROC-AUC {roc:.4f} PR-AUC {pr:.4f}")
        if bad >= args.patience:
            print(f"[{model_name} seed {seed}] erken durdu (en iyi epoch {best_epoch})")
            break

    net.load_state_dict(best_state)
    pv, pt = predict(net, xt["validation"], device), predict(net, xt["test"], device)
    thr = best_thresholds(yt["validation"], pv)
    y_te = yt["test"]
    f1_tag = f1_score(y_te, (pt >= thr).astype(int), average=None, zero_division=0)
    valid = valid_mask(y_te)
    roc, pr = macro_metrics(y_te, pt)
    f1 = float(f1_tag[valid].mean())  # ROC/PR-AUC ile aynı etiket kümesi
    prevalence = y_te.mean(0)
    result = {"model": model_name, "seed": seed, "roc_auc": roc, "pr_auc": pr, "f1_macro": f1,
              "pr_auc_baseline": float(prevalence[valid].mean()),
              "best_epoch": best_epoch, "embedding_dim": int(d), "n_test": int(len(pt)),
              "n_tags_eval": int(valid.sum())}
    if args.min_pos > 1:
        roc_m, pr_m = macro_metrics(y_te, pt, args.min_pos)
        vm = valid_mask(y_te, args.min_pos)
        result.update({"roc_auc_minpos": roc_m, "pr_auc_minpos": pr_m,
                       "f1_macro_minpos": float(f1_tag[vm].mean()), "n_tags_minpos": int(vm.sum())})
    result.update(retrieval_eval(xt["test"].numpy(), y_te))
    print(f"[{model_name} seed {seed}] TEST ROC-AUC {roc:.4f} PR-AUC {pr:.4f} (baseline {result['pr_auc_baseline']:.4f}) "
          f"F1 {f1:.4f} P@10 {result['p_at_10']:.4f} (rastgele {result['p_at_k_random']:.4f})")

    if save_dir is not None:
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(json.dumps(result, indent=1))
        torch.save({"state_dict": best_state, "mean": prep["mean"], "std": prep["std"],
                    "vocab": vocab, "thr": thr}, out / "model.pt")
        with open(out / "per_tag.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tag", "n_test_pos", "prevalence", "roc_auc", "pr_auc", "f1", "thr"])
            for j, t in enumerate(vocab):
                npos = int(y_te[:, j].sum())
                if 0 < npos < len(pt):
                    w.writerow([t, npos, f"{prevalence[j]:.4f}",
                                f"{roc_auc_score(y_te[:, j], pt[:, j]):.4f}",
                                f"{average_precision_score(y_te[:, j], pt[:, j]):.4f}",
                                f"{f1_tag[j]:.4f}", f"{thr[j]:.4f}"])
    return result


METRICS = (("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("f1_macro", "F1"),
           ("p_at_5", "P@5"), ("p_at_10", "P@10"))
METRICS_MINPOS = (("roc_auc_minpos", "ROC-AUC*"), ("pr_auc_minpos", "PR-AUC*"), ("f1_macro_minpos", "F1*"))


def summarize(results, models, seeds):
    """Model başına ortalama ± std ve (>=2 model varsa) seed başına eşleştirilmiş fark."""
    metrics = list(METRICS)
    if results and "roc_auc_minpos" in results[0]:
        metrics += list(METRICS_MINPOS)
    lines = [f"Seed'ler: {', '.join(map(str, seeds))} (std = örneklem std, n-1)", "",
             "| model | " + " | ".join(lbl for _, lbl in metrics) + " |", "|---|" + "---|" * len(metrics)]
    agg = {}
    for m in models:
        rs = [r for r in results if r["model"] == m]
        agg[m] = {}
        cells = []
        for k, _ in metrics:
            v = np.array([r[k] for r in rs])
            sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
            agg[m][k] = {"mean": float(v.mean()), "std": sd, "values": v.tolist()}
            cells.append(f"{v.mean():.4f} ± {sd:.4f}")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    diff = {}
    if len(models) >= 2:
        a, b = models[0], models[1]
        lines += ["", f"Fark ({a} − {b}), seed başına eşleştirilmiş:", "",
                  "| metrik | ort. fark ± std | her seed'de {a} daha iyi mi |".replace("{a}", a), "|---|---|---|"]
        for k, label in metrics:
            dv = np.array(agg[a][k]["values"]) - np.array(agg[b][k]["values"])
            sd = float(dv.std(ddof=1)) if len(dv) > 1 else 0.0
            wins = int((dv > 0).sum())
            diff[k] = {"mean": float(dv.mean()), "std": sd, "wins": wins, "n": len(dv)}
            lines.append(f"| {label} | {dv.mean():+.4f} ± {sd:.4f} | {wins}/{len(dv)} |")
    rb = float(np.mean([r["pr_auc_baseline"] for r in results]))
    rp = float(np.mean([r["p_at_k_random"] for r in results]))
    lines += ["", f"PR-AUC rastgele baseline (ortalama prevalence): {rb:.4f}",
              f"P@k rastgele baseline (komşunun en az bir etiketi paylaşma olasılığı): {rp:.4f}"]
    if len(metrics) > len(METRICS):
        lines.append("* = test'te en az --min-pos pozitifi olan etiketler")
    return agg, diff, "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--model", nargs="+", default=["panns", "vggish"],
                    help="panns, vggish, mert, clap veya birleşim: concat (=panns+vggish), panns+vggish+mert ...")
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="mert için kullanılacak katmanlar (0-12); verilmezse tüm katmanların ortalaması")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--pos-weight", action="store_true", help="BCE'de etiket başına pos_weight kullan")
    ap.add_argument("--pos-weight-max", type=float, default=20.0)
    ap.add_argument("--min-pos", type=int, default=1,
                    help="test'te bu sayıdan az pozitifi olan etiketleri ayrıca dışarıda bırakan ek (*) metrikler")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42], help="örn. --seeds 1 2 3 4 5")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    results = []
    for m in args.model:
        prep = prepare(m, args, device)
        for sd in args.seeds:
            results.append(train_one(m, prep, args, device, sd, out_root / m / f"seed{sd}"))
    agg, diff, md = summarize(results, args.model, args.seeds)
    print("\n=== Test sonuçları (macro) ===")
    print(md)
    (out_root / "summary.json").write_text(
        json.dumps({"runs": results, "aggregate": agg, "paired_diff": diff}, indent=1))
    (out_root / "summary.md").write_text(md + "\n")

if __name__ == "__main__":
    main()
