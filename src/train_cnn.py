#!/usr/bin/env python3
"""Sıfırdan (önceden eğitilmemiş) CNN ile MTG-Jamendo mood/theme çok etiketli sınıflandırma.

Mimari --arch ile seçilir: small (~1.2 M, 4 blok), resnet18 veya resnet34 (bkz. resnet.py),
crnn (~2.9 M, konvolüsyon + çift yönlü GRU, bkz. crnn.py).
ResNet'ler için önerilen ek düzenlileştirme: --ema (ağırlık EMA'sı), --drop-path, --mixup, --gain-db.

Giriş: pipeline.py --models mel çıktısı (pencere başına 64 x 938 log-mel, float16).
Eğitim: her epoch'ta her şarkıdan rastgele bir pencere, ondan rastgele bir --crop-sec saniyelik kesit,
SpecAugment (frekans + zaman maskesi), isteğe bağlı mixup. Değerlendirme: şarkının tüm pencereleri
tam uzunlukta CNN'e verilir, olasılıklar (ve embedding'ler) şarkı başına ortalanır.
Metrikler, eşik seçimi ve özet tablosu train.py ile aynıdır (aynı fonksiyonlar), böylece sonuçlar
MLP/PANNs/VGGish tablosuyla doğrudan karşılaştırılır. Çıktı düzeni de aynıdır:
<out-dir>/<model>/seed<N>/{results.json, per_tag.csv, model.pt}, summary.json, summary.md.

Örnek:
    python src/train_cnn.py --emb-dir /kaggle/working/emb_mel --seeds 1 2 3 4 5
    python src/train_cnn.py --emb-dir $EMB --arch resnet34 --epochs 80 --patience 15 --lr 1e-3 \
        --mixup 0.3 --ema 0.999 --drop-path 0.1 --gain-db 6 --crop-sec 15 --pos-weight --seeds 1
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from common import SPLITS, read_split, set_seed
from crnn import CRNN
from resnet import EMA, AudioResNet
from train import best_thresholds, macro_metrics, resolve_emb_dir, retrieval_eval, summarize, valid_mask


# ----------------------------------------------------------------------------- veri

def load_mel(emb_dir: Path, vocab):
    """mel/*.npz -> split başına bellekte float16 pencere dizisi + şarkı başına pencere aralığı.

    Önce yalnızca ids/n_win okunur (npz üyeleri tembel açılır), her split için bellek ayrılır, sonra
    dosyalar tek tek doldurulur: tepe bellek ~ toplam veri boyutu (kopya yok).
    """
    tag2idx = {t: i for i, t in enumerate(vocab)}
    rows = {s: read_split(emb_dir / "splits" / f"{s}.tsv") for s in SPLITS}
    owner = {r["track_id"]: (s, r) for s in SPLITS for r in rows[s]}
    files = sorted((emb_dir / "mel").glob("*.npz"))
    if not files:
        raise SystemExit(f"{emb_dir / 'mel'} içinde .npz yok (pipeline.py --models mel çalıştırıldı mı?).")

    count = {s: 0 for s in SPLITS}
    shape = None
    for f in files:
        z = np.load(f)
        for tid, nw in zip(z["ids"], z["n_win"]):
            if int(tid) in owner:
                count[owner[int(tid)][0]] += int(nw)
        if shape is None:
            shape = z["emb"].shape[1:]
    data = {s: np.empty((count[s],) + tuple(shape), dtype=np.float16) for s in SPLITS}
    pos = {s: 0 for s in SPLITS}
    songs = {s: {"ids": [], "start": [], "n": [], "y": []} for s in SPLITS}
    for f in files:
        z = np.load(f)
        ids, nw, emb = z["ids"], z["n_win"].astype(np.int64), z["emb"]
        off = np.concatenate([[0], np.cumsum(nw)[:-1]])
        for tid, n, o in zip(ids, nw, off):
            if int(tid) not in owner:
                continue
            s, r = owner[int(tid)]
            data[s][pos[s]:pos[s] + n] = emb[o:o + n]
            y = np.zeros(len(vocab), dtype=np.float32)
            for t in r["tags"]:
                if t in tag2idx:
                    y[tag2idx[t]] = 1.0
            sg = songs[s]
            sg["ids"].append(int(tid)), sg["start"].append(pos[s]), sg["n"].append(int(n)), sg["y"].append(y)
            pos[s] += n
    out = {}
    for s in SPLITS:
        sg = songs[s]
        out[s] = {"x": torch.from_numpy(data[s]), "ids": np.array(sg["ids"]),
                  "start": np.array(sg["start"]), "n": np.array(sg["n"]), "y": np.stack(sg["y"])}
        print(f"[mel] {s}: {len(sg['ids'])}/{len(rows[s])} şarkı, {len(out[s]['x'])} pencere")
    return out


# ----------------------------------------------------------------------------- model

def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    """4 blok (32-64-128-256 kanal), frekans ekseni ortalanır, zaman ekseninde ortalama+max havuzlama.
    ~1.2 M parametre. Embedding (512) = öneri sistemi için şarkı vektörü."""

    def __init__(self, n_tags, width=32, dropout=0.5):
        super().__init__()
        w = width
        self.bn_in = nn.BatchNorm2d(1)
        self.blocks = nn.Sequential(block(1, w), block(w, 2 * w), block(2 * w, 4 * w), block(4 * w, 8 * w))
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(16 * w, n_tags)

    def embed(self, x):  # x: (B, 1, mel, kare)
        h = self.blocks(self.bn_in(x))
        h = h.mean(2)  # frekans
        return torch.cat([h.mean(2), h.amax(2)], dim=1)  # zaman: ortalama + max -> (B, 16w)

    def forward(self, x, return_emb=False):
        e = self.embed(x)
        out = self.fc(self.drop(e))
        return (out, e) if return_emb else out


MODEL_NAME = {"small": "cnn", "resnet18": "resnet18", "resnet34": "resnet34", "crnn": "crnn"}


def build_model(cfg, n_tags):
    """cfg: arch, width, dropout, drop_path, n_mels (model.pt içine de bu sözlük yazılır)."""
    if cfg["arch"] == "small":
        return SmallCNN(n_tags, cfg["width"], cfg["dropout"])
    if cfg["arch"] == "crnn":
        return CRNN(n_tags, cfg["n_mels"], cfg["width"], cfg.get("rnn_hidden", 128), cfg.get("rnn_layers", 2),
                    cfg["dropout"])
    return AudioResNet(n_tags, cfg["arch"], cfg["n_mels"], cfg["width"], cfg["drop_path"], cfg["dropout"])


# ----------------------------------------------------------------------------- eğitim yardımcıları

def spec_augment(x, freq_mask=8, time_mask=40, n_masks=2):
    """x: (B, 1, mel, kare). Her örnek için bağımsız frekans ve zaman maskeleri (maske değeri = 0, normalize sonrası ortalama)."""
    B, _, M, T = x.shape
    for _ in range(n_masks):
        f = torch.randint(0, freq_mask + 1, (B,), device=x.device)
        f0 = (torch.rand(B, device=x.device) * (M - f)).long()
        t = torch.randint(0, time_mask + 1, (B,), device=x.device)
        t0 = (torch.rand(B, device=x.device) * (T - t)).long()
        fr = torch.arange(M, device=x.device)[None, :]
        tr = torch.arange(T, device=x.device)[None, :]
        fm = (fr >= f0[:, None]) & (fr < (f0 + f)[:, None])
        tm = (tr >= t0[:, None]) & (tr < (t0 + t)[:, None])
        x = x.masked_fill((fm[:, :, None] | tm[:, None, :])[:, None], 0.0)
    return x


def sample_batch(split, idx, crop, rng):
    """Seçilen şarkılardan rastgele pencere + rastgele zaman kesiti. Dönen: (B, 1, mel, crop) float32, y."""
    xs = []
    for i in idx:
        w = split["start"][i] + rng.integers(split["n"][i])
        T = split["x"].shape[-1]
        s = rng.integers(0, T - crop + 1)
        xs.append(split["x"][w, :, s:s + crop])
    x = torch.stack(xs).float().unsqueeze(1)
    return x, torch.from_numpy(split["y"][idx])


@torch.no_grad()
def predict_songs(net, split, mean, std, device, bs=32):
    """Şarkı başına tüm pencereler tam uzunlukta; olasılık ve embedding pencere ortalaması."""
    net.eval()
    nw = split["n"]
    probs, embs = [], []
    wp, we = [], []
    for i in range(0, len(split["x"]), bs):
        x = (split["x"][i:i + bs].to(device).float().unsqueeze(1) - mean) / std
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            lo, e = net(x, return_emb=True)
        wp.append(torch.sigmoid(lo.float()).cpu()), we.append(e.float().cpu())
    wp, we = torch.cat(wp).numpy(), torch.cat(we).numpy()
    for s, n in zip(split["start"], nw):
        probs.append(wp[s:s + n].mean(0)), embs.append(we[s:s + n].mean(0))
    return np.stack(probs), np.stack(embs)


def train_one(prep, args, device, seed, save_dir):
    set_seed(seed)
    rng = np.random.default_rng(seed)
    data, vocab, mean, std = prep["data"], prep["vocab"], prep["mean"], prep["std"]
    tr, va, te = data["train"], data["validation"], data["test"]
    name = MODEL_NAME[args.arch]
    cfg = {"arch": args.arch, "width": args.width, "dropout": args.dropout, "drop_path": args.drop_path,
           "n_mels": int(tr["x"].shape[1]), "rnn_hidden": args.rnn_hidden, "rnn_layers": args.rnn_layers}
    net = build_model(cfg, len(vocab)).to(device)
    if device == "cuda":
        net = net.to(memory_format=torch.channels_last)
    ema = EMA(net, args.ema) if args.ema > 0 else None
    n_params = sum(p.numel() for p in net.parameters())
    print(f"[{name} seed {seed}] parametre sayısı: {n_params / 1e6:.2f} M")

    pos = tr["y"].sum(0)
    neg = len(tr["y"]) - pos
    pos_w = None
    if args.pos_weight:
        pos_w = torch.tensor(np.clip(neg / np.maximum(pos, 1), 1.0, args.pos_weight_max), dtype=torch.float32,
                             device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    # BN ve bias parametrelerine weight decay uygulanmaz (yalnızca konv/doğrusal ağırlıklar)
    decay = [p for p in net.parameters() if p.ndim > 1]
    no_decay = [p for p in net.parameters() if p.ndim <= 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
    steps_per_epoch = int(np.ceil(len(tr["ids"]) / args.batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch,
                                                pct_start=0.15)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    crop = int(round(args.crop_sec * tr["x"].shape[-1] / 30))

    best_pr, best_state, bad, best_epoch = -1.0, None, 0, 0
    history = []  # epoch başına ortalama eğitim kaybı ve validation metrikleri (visualize.py çizer)
    t0 = time.time()
    eval_net = net
    if ema:  # EMA ağırlıkları ayrı bir kopyada değerlendirilir; en iyi epoch EMA'ya göre seçilir
        eval_net = build_model(cfg, len(vocab)).to(device)
        if device == "cuda":
            eval_net = eval_net.to(memory_format=torch.channels_last)
    for epoch in range(1, args.epochs + 1):
        net.train()
        perm = rng.permutation(len(tr["ids"]))
        loss_sum, n_batches = 0.0, 0
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size]
            if len(idx) < 2:
                continue
            x, y = sample_batch(tr, idx, crop, rng)
            x, y = x.to(device), y.to(device)
            if args.gain_db > 0:  # dB ölçeğinde rastgele kazanç = sabit kayma
                x = x + (torch.rand(len(x), 1, 1, 1, device=device) * 2 - 1) * args.gain_db
            x = (x - mean) / std
            x = spec_augment(x, args.freq_mask, args.time_mask)
            if args.mixup > 0:
                lam = float(rng.beta(args.mixup, args.mixup))
                p2 = torch.randperm(len(x), device=device)
                x, y = lam * x + (1 - lam) * x[p2], lam * y + (1 - lam) * y[p2]
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                loss = loss_fn(net(x).float(), y)
            scaler.scale(loss).backward()
            if args.clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if ema:
                ema.update(net)
            loss_sum, n_batches = loss_sum + loss.item(), n_batches + 1
        if ema:
            eval_net.load_state_dict(ema.shadow)
        pv, _ = predict_songs(eval_net, va, mean, std, device)
        roc, pr = macro_metrics(va["y"], pv)
        history.append({"epoch": epoch, "train_loss": loss_sum / max(1, n_batches), "val_roc_auc": roc,
                        "val_pr_auc": pr})
        if pr > best_pr:
            best_pr, best_epoch, bad = pr, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in eval_net.state_dict().items()}
        else:
            bad += 1
        print(f"[{name} seed {seed}] epoch {epoch:3d} loss {history[-1]['train_loss']:.4f} val ROC-AUC {roc:.4f} PR-AUC {pr:.4f} "
              f"({(time.time() - t0) / 60:.1f} dk)")
        if bad >= args.patience:
            print(f"[{name} seed {seed}] erken durdu (en iyi epoch {best_epoch})")
            break

    net = eval_net
    net.load_state_dict(best_state)
    pv, _ = predict_songs(net, va, mean, std, device)
    pt, et = predict_songs(net, te, mean, std, device)
    thr = best_thresholds(va["y"], pv)
    y_te = te["y"]
    f1_tag = f1_score(y_te, (pt >= thr).astype(int), average=None, zero_division=0)
    valid = valid_mask(y_te)
    roc, pr = macro_metrics(y_te, pt)
    f1 = float(f1_tag[valid].mean())
    prevalence = y_te.mean(0)
    result = {"model": name, "seed": seed, "roc_auc": roc, "pr_auc": pr, "f1_macro": f1,
              "pr_auc_baseline": float(prevalence[valid].mean()), "best_epoch": best_epoch,
              "embedding_dim": int(et.shape[1]), "n_params": int(n_params), "n_test": int(len(pt)),
              "n_tags_eval": int(valid.sum())}
    result.update(retrieval_eval(et, y_te))
    result["history"] = history
    print(f"[{name} seed {seed}] TEST ROC-AUC {roc:.4f} PR-AUC {pr:.4f} (baseline {result['pr_auc_baseline']:.4f}) "
          f"F1 {f1:.4f} P@10 {result['p_at_10']:.4f} (rastgele {result['p_at_k_random']:.4f})")

    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(result, indent=1))
    torch.save({"state_dict": best_state, "mean": mean, "std": std, "vocab": vocab, "thr": thr,
                "width": args.width, "cfg": cfg}, out / "model.pt")
    np.savez_compressed(out / "test_probs_emb.npz", ids=te["ids"], probs=pt, emb=et.astype(np.float16))
    with open(out / "per_tag.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "n_test_pos", "prevalence", "roc_auc", "pr_auc", "f1", "thr"])
        for j, t in enumerate(vocab):
            npos = int(y_te[:, j].sum())
            if 0 < npos < len(pt):
                w.writerow([t, npos, f"{prevalence[j]:.4f}", f"{roc_auc_score(y_te[:, j], pt[:, j]):.4f}",
                            f"{average_precision_score(y_te[:, j], pt[:, j]):.4f}", f"{f1_tag[j]:.4f}",
                            f"{thr[j]:.4f}"])
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb-dir", required=True, help="pipeline.py --models mel çıktısı (splits/, vocab.json, mel/)")
    ap.add_argument("--out-dir", default="runs_cnn")
    ap.add_argument("--arch", choices=list(MODEL_NAME), default="small",
                    help="small (küçük CNN), resnet18, resnet34, crnn")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--width", type=int, default=None,
                    help="ilk aşama kanal sayısı; varsayılan small: 32 (32-64-128-256), resnet: 64 (64-128-256-512)")
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--crop-sec", type=float, default=10.0, help="eğitimde rastgele kesit uzunluğu (sn)")
    ap.add_argument("--freq-mask", type=int, default=8)
    ap.add_argument("--time-mask", type=int, default=40)
    ap.add_argument("--mixup", type=float, default=0.0, help="mixup beta parametresi (0 = kapalı)")
    ap.add_argument("--drop-path", type=float, default=0.1, help="resnet: stochastic depth oranı (son blok)")
    ap.add_argument("--rnn-hidden", type=int, default=128, help="crnn: GRU gizli boyutu (yön başına)")
    ap.add_argument("--rnn-layers", type=int, default=2, help="crnn: GRU katman sayısı")
    ap.add_argument("--ema", type=float, default=0.0, help="ağırlık EMA katsayısı, ör. 0.999 (0 = kapalı)")
    ap.add_argument("--gain-db", type=float, default=0.0, help="eğitimde ±dB rastgele kazanç (0 = kapalı)")
    ap.add_argument("--clip", type=float, default=0.0, help="gradyan normu kırpma (0 = kapalı)")
    ap.add_argument("--pos-weight", action="store_true", help="BCE'de etiket başına pos_weight (train.py ile aynı)")
    ap.add_argument("--pos-weight-max", type=float, default=20.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--max-train", type=int, default=0, help="duman testi için ilk N eğitim şarkısı")
    args = ap.parse_args()
    if args.width is None:
        args.width = 32 if args.arch == "small" else 64

    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb_dir = resolve_emb_dir(args.emb_dir)
    vocab = json.loads((emb_dir / "vocab.json").read_text())
    data = load_mel(emb_dir, vocab)
    if args.max_train:
        tr = data["train"]
        k = args.max_train
        data["train"] = {**tr, "ids": tr["ids"][:k], "start": tr["start"][:k], "n": tr["n"][:k], "y": tr["y"][:k]}
    # normalizasyon: eğitim kümesinden tek skaler ortalama/std (dB ölçeği zaten mel bantları arasında tutarlı)
    sample = data["train"]["x"][:: max(1, len(data["train"]["x"]) // 2000)].float()
    mean, std = float(sample.mean()), float(sample.std() + 1e-6)
    print(f"normalizasyon: ortalama {mean:.2f} dB, std {std:.2f}")
    prep = {"data": data, "vocab": vocab, "mean": mean, "std": std}

    out_root = Path(args.out_dir)
    results = [train_one(prep, args, device, sd, out_root / MODEL_NAME[args.arch] / f"seed{sd}") for sd in args.seeds]
    agg, diff, md = summarize(results, [MODEL_NAME[args.arch]], args.seeds)
    print("\n=== Test sonuçları (macro) ===")
    print(md)
    (out_root / "summary.json").write_text(json.dumps({"runs": results, "aggregate": agg}, indent=1))
    (out_root / "summary.md").write_text(md + "\n")


if __name__ == "__main__":
    main()
