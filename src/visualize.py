#!/usr/bin/env python3
"""Veri seti ve deney sonuçlarının görselleştirilmesi: PNG grafikler + tek dosyalık HTML rapor.

Veri bölümü (--splits-dir): split boyutları, etiket sıklıkları ve split'ler arası dağılım, şarkı başına
etiket sayısı, etiket birlikte görülme ısı haritası, şarkı süreleri. --mel-dir verilirse örnek log-mel
spektrogramları, etiket başına ortalama mel spektrumu ve şarkı başına pencere sayısı.

Sonuç bölümü (--runs): train.py / train_cnn.py çıktı klasörleri (summary.json içerenler). Model
karşılaştırma tablosu ve grafikleri (metrik başına ayrı panel), etiket bazında PR-AUC / ROC-AUC,
PR-AUC'nin etiket sıklığıyla ilişkisi, eğitim eğrileri (results.json içinde history varsa) ve
test embedding'lerinin t-SNE haritası (test_probs_emb.npz olan CNN/ResNet koşuları).

Örnekler:
    python src/visualize.py --splits-dir /kaggle/working/emb_mel/splits --mel-dir /kaggle/working/emb_mel/mel \\
        --runs /kaggle/working/runs_v3 /kaggle/working/runs_cnn /kaggle/working/runs_resnet --out viz
    python src/visualize.py --splits-dir mtg-jamendo-dataset/data/splits/split-0 --out viz   # yalnızca veri
--runs öğesi "klasör:etiket" biçiminde etiketlenebilir (ör. runs_v3:MLP). Birden fazla koşu verilirse
model adları "etiket/model" olur.
"""
import argparse
import base64
import html
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from common import SPLITS, read_split  # noqa: E402

# ----------------------------------------------------------------------------- stil
# Kategorik renkler sabit sırayla atanır (model -> renk, sıralamaya göre değil); sıralı ölçek tek ton mavi.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#fcfcfb"] + BLUE)
SPEC = LinearSegmentedColormap.from_list("spec", ["#0d1b2a", "#184f95", "#3987e5", "#9ec5f4", "#fcfcfb"])
SPLIT_TR = {"train": "eğitim", "validation": "doğrulama", "test": "test"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 11,
    "axes.titlelocation": "left", "font.size": 9, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
})


class Report:
    """Grafikleri PNG olarak kaydeder ve HTML rapora (base64 gömülü) bölüm bölüm ekler."""

    def __init__(self, out: Path):
        self.out = out
        (out / "figures").mkdir(parents=True, exist_ok=True)
        (out / "tables").mkdir(exist_ok=True)
        self.parts = []

    def h2(self, text):
        self.parts.append(f"<h2>{html.escape(text)}</h2>")

    def note(self, text):
        self.parts.append(f"<p class='note'>{html.escape(text)}</p>")

    def fig(self, fig, name, caption=""):
        path = self.out / "figures" / f"{name}.png"
        fig.savefig(path)
        plt.close(fig)
        b64 = base64.b64encode(path.read_bytes()).decode()
        cap = f"<figcaption>{html.escape(caption)}</figcaption>" if caption else ""
        self.parts.append(f"<figure><img src='data:image/png;base64,{b64}' alt='{html.escape(name)}'>{cap}</figure>")
        print(f"[viz] {path}")

    def table(self, df: pd.DataFrame, name, caption=""):
        df.to_csv(self.out / "tables" / f"{name}.csv", index=False)
        cap = f"<caption>{html.escape(caption)}</caption>" if caption else ""
        self.parts.append("<div class='tw'>" + df.to_html(index=False, border=0, escape=True).replace(
            "<thead>", cap + "<thead>", 1) + "</div>")

    def write(self, title):
        css = f"""
:root{{--bg:{SURFACE};--ink:{INK};--ink2:{INK2};--grid:{GRID}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:24px 16px 64px}}
h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:19px;margin:40px 0 8px;padding-top:12px;border-top:1px solid var(--grid)}}
.note{{color:var(--ink2);margin:4px 0 12px}}
figure{{margin:16px 0}} img{{max-width:100%;height:auto}} figcaption{{color:var(--ink2);font-size:13px}}
.tw{{overflow-x:auto;margin:12px 0}} table{{border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:13px}}
caption{{text-align:left;color:var(--ink2);padding-bottom:6px}}
th,td{{padding:5px 10px;border-bottom:1px solid var(--grid);text-align:right}} th:first-child,td:first-child{{text-align:left}}
th{{color:var(--ink2);font-weight:600}}
"""
        doc = (f"<!doctype html><html lang='tr'><head><meta charset='utf-8'>"
               f"<meta name='viewport' content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title>"
               f"<style>{css}</style></head><body><main><h1>{html.escape(title)}</h1>"
               + "\n".join(self.parts) + "</main></body></html>")
        path = self.out / "report.html"
        path.write_text(doc, encoding="utf-8")
        print(f"[viz] rapor: {path}")


def hbar(ax, labels, values, color=SERIES[0], fmt="{:.0f}", label_every=1):
    y = np.arange(len(labels))
    ax.barh(y, values, color=color, height=0.72, edgecolor=SURFACE, linewidth=1)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    for i, v in enumerate(values):
        if i % label_every == 0:
            ax.text(v, i, " " + fmt.format(v), va="center", fontsize=7.5, color=INK2)


# ----------------------------------------------------------------------------- veri

def find_split_file(d: Path, split):
    for name in (f"{split}.tsv", f"autotagging_moodtheme-{split}.tsv"):
        if (d / name).exists():
            return d / name
    raise SystemExit(f"{d} içinde {split}.tsv veya autotagging_moodtheme-{split}.tsv yok.")


def data_section(rep: Report, splits_dir: Path, mel_dir: Path | None, top_k: int):
    rows = {s: read_split(find_split_file(splits_dir, s)) for s in SPLITS}
    vocab = sorted({t for s in SPLITS for r in rows[s] for t in r["tags"]})
    rep.h2("1. Veri seti")
    n = {s: len(rows[s]) for s in SPLITS}
    total = sum(n.values())
    rep.table(pd.DataFrame([{"split": SPLIT_TR[s], "şarkı": n[s], "oran": f"{n[s] / total:.1%}",
                             "ort. etiket/şarkı": f"{np.mean([len(r['tags']) for r in rows[s]]):.2f}",
                             "ort. süre (dk)": f"{np.mean([r['duration'] for r in rows[s]]) / 60:.2f}"}
                            for s in SPLITS]), "split_sizes", f"{len(vocab)} etiket, toplam {total} şarkı")

    # etiket sıklığı (eğitim)
    cnt = {s: Counter(t for r in rows[s] for t in r["tags"]) for s in SPLITS}
    order = sorted(vocab, key=lambda t: -cnt["train"][t])
    fig, ax = plt.subplots(figsize=(8, 0.17 * len(vocab) + 1))
    hbar(ax, order, [cnt["train"][t] for t in order])
    ax.set_xlabel("eğitim kümesindeki şarkı sayısı")
    ax.set_title("Etiket sıklığı (eğitim) — uzun kuyruklu dağılım")
    rep.fig(fig, "data_tag_counts", "En sık ve en nadir etiket arasındaki fark, macro metriklerin neden "
                                    "düşük kaldığını açıklar: nadir etiketlerin az örneği var.")

    # split'ler arası prevalence tutarlılığı
    prev = pd.DataFrame({SPLIT_TR[s]: [cnt[s][t] / n[s] for t in order] for s in SPLITS}, index=order)
    fig, ax = plt.subplots(figsize=(8, 0.17 * len(vocab) + 1))
    y = np.arange(len(order))
    for k, s in enumerate(SPLITS):
        ax.scatter(prev[SPLIT_TR[s]], y, s=22, color=SERIES[k], label=SPLIT_TR[s], edgecolor=SURFACE, linewidth=1,
                   zorder=3)
    ax.set_yticks(y, order)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("etiketi taşıyan şarkı oranı")
    ax.set_title("Etiket oranı split'ler arasında")
    ax.legend(loc="lower right", ncols=3)
    rep.fig(fig, "data_tag_prevalence_by_split", "Noktalar üst üste biniyorsa split'ler etiket dağılımı "
                                                 "açısından tutarlıdır.")
    prev.reset_index(names="etiket").round(4).to_csv(rep.out / "tables" / "tag_prevalence.csv", index=False)

    # şarkı başına etiket sayısı ve süre
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    k = Counter(len(r["tags"]) for r in rows["train"])
    xs = sorted(k)
    axes[0].bar(xs, [k[x] for x in xs], color=SERIES[0], width=0.72, edgecolor=SURFACE)
    axes[0].set_xticks(xs)
    axes[0].set_xlabel("şarkı başına etiket sayısı")
    axes[0].set_ylabel("şarkı")
    axes[0].set_title("Etiket sayısı (eğitim)")
    axes[0].grid(axis="x", visible=False)
    dur = np.array([r["duration"] for s in SPLITS for r in rows[s]]) / 60
    axes[1].hist(np.clip(dur, 0, 15), bins=60, color=SERIES[0], edgecolor=SURFACE, linewidth=0.5)
    axes[1].axvline(np.median(dur), color=INK2, linewidth=1, linestyle="--")
    axes[1].text(np.median(dur), axes[1].get_ylim()[1] * 0.95, f"  medyan {np.median(dur):.1f} dk", color=INK2,
                 va="top", fontsize=8)
    axes[1].set_xlabel("süre (dk, 15+ kırpıldı)")
    axes[1].set_title("Şarkı süreleri (tüm split'ler)")
    axes[1].grid(axis="x", visible=False)
    rep.fig(fig, "data_tags_per_song_duration")

    # birlikte görülme: P(B | A), en sık top_k etiket
    top = order[:top_k]
    idx = {t: i for i, t in enumerate(top)}
    co = np.zeros((len(top), len(top)))
    for r in rows["train"]:
        ts = [idx[t] for t in r["tags"] if t in idx]
        for a in ts:
            for b in ts:
                co[a, b] += 1
    cond = co / np.maximum(np.diag(co)[:, None], 1)
    np.fill_diagonal(cond, np.nan)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cond, cmap=SEQ, vmin=0, vmax=np.nanmax(cond))
    ax.set_xticks(range(len(top)), top, rotation=90)
    ax.set_yticks(range(len(top)), top)
    ax.grid(False)
    for a in range(len(top)):
        for b in range(len(top)):
            if not np.isnan(cond[a, b]) and cond[a, b] >= 0.25:
                ax.text(b, a, f"{cond[a, b]:.2f}", ha="center", va="center", fontsize=6,
                        color=SURFACE if cond[a, b] > 0.5 * np.nanmax(cond) else INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.7)
    cb.outline.set_visible(False)
    cb.set_label("P(sütun etiketi | satır etiketi)")
    ax.set_title(f"Etiketlerin birlikte görülmesi (en sık {len(top)} etiket, eğitim)")
    rep.fig(fig, "data_tag_cooccurrence", "Satırdaki etiketi taşıyan şarkıların yüzde kaçı sütundaki etiketi de "
                                          "taşıyor. ≥ 0.25 olan hücreler yazılı.")
    pairs = [(top[a], top[b], int(co[a, b]), cond[a, b]) for a in range(len(top)) for b in range(len(top))
             if a != b and co[a, b] > 0]
    pairs.sort(key=lambda p: -p[3])
    rep.table(pd.DataFrame(pairs[:15], columns=["etiket A", "etiket B", "birlikte", "P(B|A)"]).round(3),
              "top_cooccurrence", "En güçlü 15 koşullu birliktelik")

    if mel_dir is not None:
        mel_section(rep, mel_dir, rows, order)


def mel_section(rep: Report, mel_dir: Path, rows, order):
    files = sorted(mel_dir.glob("*.npz"))
    if not files:
        rep.note(f"{mel_dir} içinde .npz bulunamadı, mel grafikleri atlandı.")
        return
    tags_of = {r["track_id"]: r["tags"] for s in SPLITS for r in rows[s]}
    nwin, examples, sums, counts, used = [], {}, {}, Counter(), set()
    want = order[:3] + order[-3:]  # en sık 3 ve en nadir 3 etiket için örnek
    for f in files:
        z = np.load(f)
        ids, nw = z["ids"], z["n_win"].astype(int)
        off = np.concatenate([[0], np.cumsum(nw)[:-1]])
        emb = z["emb"]
        nwin.extend(nw.tolist())
        for tid, o, k in zip(ids, off, nw):
            tags = tags_of.get(int(tid), [])
            spec = emb[o].astype(np.float32)  # ilk pencere
            prof = spec.mean(1)
            for t in tags:
                sums[t] = sums.get(t, 0) + prof
                counts[t] += 1
                if t in want and t not in examples and int(tid) not in used:
                    examples[t] = (int(tid), spec)
                    used.add(int(tid))
    rep.h2("2. Log-mel spektrogramlar")
    sel = [t for t in want if t in examples]
    if sel:
        fig, axes = plt.subplots(len(sel), 1, figsize=(10, 1.9 * len(sel)), sharex=True)
        axes = np.atleast_1d(axes)
        for ax, t in zip(axes, sel):
            tid, spec = examples[t]
            im = ax.imshow(spec, aspect="auto", origin="lower", cmap=SPEC, extent=[0, 30, 0, spec.shape[0]])
            ax.set_title(f"{t}  (track {tid})", fontsize=9)
            ax.set_ylabel("mel bandı")
            ax.grid(False)
        axes[-1].set_xlabel("zaman (sn, ilk 30 sn'lik pencere)")
        cb = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.01)
        cb.outline.set_visible(False)
        cb.set_label("dB")
        rep.fig(fig, "mel_examples", "Üstte en sık 3, altta en nadir 3 etiketten birer örnek (modelin gördüğü giriş).")

    tags = [t for t in order[:6] if counts[t]]
    if tags:
        fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
        allmean = sum(sums.values()) / sum(counts.values())
        for k, t in enumerate(tags):
            ax.plot(sums[t] / counts[t] - allmean, color=SERIES[k], linewidth=2, label=t)
        ax.axhline(0, color=MUTED, linewidth=1)
        ax.set_xlabel("mel bandı (düşük → yüksek frekans)")
        ax.set_ylabel("dB fark")
        ax.set_title("Etiket başına ortalama spektrum (genel ortalamadan fark)")
        fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncols=len(tags))
        rep.fig(fig, "mel_mean_spectrum", "Etiketin şarkılarının hangi frekans bölgelerinde ortalamadan güçlü/zayıf "
                                          "olduğu (en sık 6 etiket).")

    c = Counter(nwin)
    xs = sorted(c)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(xs, [c[x] for x in xs], color=SERIES[0], width=0.72, edgecolor=SURFACE)
    ax.set_xticks(xs)
    ax.set_xlabel("şarkı başına 30 sn'lik pencere")
    ax.set_ylabel("şarkı")
    ax.set_title("Pencere sayısı")
    ax.grid(axis="x", visible=False)
    rep.fig(fig, "mel_windows_per_song")


# ----------------------------------------------------------------------------- sonuçlar

METRICS = [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("f1_macro", "F1"), ("p_at_5", "P@5"),
           ("p_at_10", "P@10")]


def load_runs(specs):
    """Her koşu klasöründen: model -> {runs: [results], dirs: [seed klasörleri]}."""
    out = {}
    for spec in specs:
        d, _, label = spec.partition(":")
        d = Path(d)
        sj = d / "summary.json"
        if not sj.exists():
            print(f"[viz] uyarı: {sj} yok, atlandı")
            continue
        runs = json.loads(sj.read_text())["runs"]
        label = label or d.name
        for r in runs:
            name = f"{label}/{r['model']}" if len(specs) > 1 else r["model"]
            e = out.setdefault(name, {"runs": [], "dirs": []})
            e["runs"].append(r)
            e["dirs"].append(d / r["model"] / f"seed{r['seed']}")
    return out


def results_section(rep: Report, run_specs):
    models = load_runs(run_specs)
    if not models:
        rep.note("Hiç summary.json bulunamadı; sonuç bölümü boş.")
        return
    names = list(models)
    color = {m: SERIES[i % len(SERIES)] for i, m in enumerate(names)}  # sabit sıra: verilen koşu sırası
    rep.h2("3. Model karşılaştırması (test, macro)")

    rows = []
    for m in names:
        rs = models[m]["runs"]
        row = {"model": m, "seed": len(rs)}
        for k, lbl in METRICS:
            v = np.array([r[k] for r in rs])
            row[lbl] = f"{v.mean():.4f} ± {v.std(ddof=1) if len(v) > 1 else 0:.4f}"
        row["parametre"] = f"{rs[0]['n_params'] / 1e6:.1f} M" if "n_params" in rs[0] else "–"
        row["en iyi epoch"] = f"{np.mean([r['best_epoch'] for r in rs]):.0f}"
        rows.append(row)
    base_pr = float(np.mean([r["pr_auc_baseline"] for m in names for r in models[m]["runs"]]))
    base_pk = float(np.mean([r["p_at_k_random"] for m in names for r in models[m]["runs"]]))
    rep.table(pd.DataFrame(rows), "model_comparison",
              f"Ortalama ± std (seed'ler arası). Rastgele: PR-AUC {base_pr:.4f}, P@k {base_pk:.4f}")

    # metrik başına ayrı panel (tek eksen kuralı: farklı ölçekler aynı eksene konmaz)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(3.0 * len(METRICS), 0.5 * len(names) + 1.6), sharey=True)
    y = np.arange(len(names))
    for ax, (k, lbl) in zip(axes, METRICS):
        mean = np.array([np.mean([r[k] for r in models[m]["runs"]]) for m in names])
        sd = np.array([np.std([r[k] for r in models[m]["runs"]], ddof=1) if len(models[m]["runs"]) > 1 else 0
                       for m in names])
        # eksen sıfırdan başlamadığı için çubuk değil nokta (çubuk boyu yanıltıcı olurdu)
        for i, m in enumerate(names):
            ax.errorbar(mean[i], i, xerr=sd[i], fmt="o", ms=8, color=color[m], mec=SURFACE, mew=1.5,
                        ecolor=INK2, elinewidth=1, capsize=3, zorder=3)
        # eksen rastgele tahmin seviyesinden (F1 için 0) başlar: dar bir aralığa zoom yapılırsa 0.003'lük
        # fark büyük görünür. Farkların büyüklüğü bu yüzden "rastgeleye göre kazanç" ölçeğinde okunur.
        ref = {"pr_auc": base_pr, "p_at_5": base_pk, "p_at_10": base_pk, "roc_auc": 0.5}.get(k)
        hi = max(mean + sd)
        left = ref if ref is not None else 0.0
        span = hi - left
        ax.set_xlim(left - span * 0.04, hi + span * 0.12)
        if ref is not None:
            ax.axvline(ref, color=MUTED, linestyle="--", linewidth=1)
        for i, v in enumerate(mean):
            ax.annotate(f"{v:.3f}", (v, i), xytext=(0, 9), textcoords="offset points", ha="center",
                        fontsize=7.5, color=INK2)
        ax.set_ylim(len(names) - 0.4, -0.6)
        ax.set_title(lbl)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, names)
    rep.fig(fig, "results_model_comparison", "Her panel kendi ölçeğinde ve rastgele tahmin seviyesinden "
                                             "(kesik çizgi; F1 için 0) başlar. Hata çubukları seed'ler arası std. "
                                             "Kesin değerler için tabloya bakın.")

    # seed dağılımı (PR-AUC)
    if any(len(models[m]["runs"]) > 1 for m in names):
        fig, ax = plt.subplots(figsize=(7, 0.45 * len(names) + 1.2))
        for i, m in enumerate(names):
            v = [r["pr_auc"] for r in models[m]["runs"]]
            ax.scatter(v, [i] * len(v), s=36, color=color[m], edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.set_xlabel("test PR-AUC")
        ax.set_title("Seed'ler arası yayılım (her nokta bir seed)")
        ax.grid(axis="y", visible=False)
        rep.fig(fig, "results_seed_spread")

    per_tag_section(rep, models, names, color)
    curves_section(rep, models, names, color)
    tsne_section(rep, models, names)


def read_per_tag(model):
    dfs = [pd.read_csv(d / "per_tag.csv") for d in model["dirs"] if (d / "per_tag.csv").exists()]
    if not dfs:
        return None
    df = pd.concat(dfs).groupby("tag", as_index=False).mean(numeric_only=True)
    return df


def per_tag_section(rep: Report, models, names, color):
    pt = {m: read_per_tag(models[m]) for m in names}
    pt = {m: d for m, d in pt.items() if d is not None}
    if not pt:
        return
    rep.h2("4. Etiket bazında sonuçlar")
    best = max(pt, key=lambda m: pt[m]["pr_auc"].mean())
    df = pt[best].sort_values("pr_auc", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 0.17 * len(df) + 1))
    yy = np.arange(len(df))
    ax.barh(yy, df["pr_auc"], color=color[best], height=0.72, edgecolor=SURFACE, label="PR-AUC")
    ax.scatter(df["prevalence"], yy, marker="|", s=60, color=INK, linewidth=1.5, zorder=3,
               label="rastgele (etiket oranı)")
    ax.set_yticks(yy, df["tag"])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("test PR-AUC (seed ortalaması)")
    ax.set_title(f"Etiket başına PR-AUC — {best}")
    ax.legend(loc="lower right")
    rep.fig(fig, "results_per_tag_pr_auc", "Çubuk ile siyah çizgi arasındaki mesafe, modelin o etikette rastgeleden "
                                           "ne kadar iyi olduğunu gösterir.")

    # ROC-AUC etiket bazında, modeller arası (en fazla 4 model; daha fazlası okunmaz)
    shown = list(pt)[:4]
    tags = pt[best].sort_values("roc_auc", ascending=False)["tag"].tolist()
    fig, ax = plt.subplots(figsize=(8.5, 0.17 * len(tags) + 1.6), layout="constrained")
    for m in shown:
        d = pt[m].set_index("tag")["roc_auc"]
        ax.scatter([d.get(t, np.nan) for t in tags], range(len(tags)), s=20, color=color[m], label=m,
                   edgecolor=SURFACE, linewidth=1, zorder=3)
    ax.axvline(0.5, color=MUTED, linestyle="--", linewidth=1)
    ax.set_yticks(range(len(tags)), tags)
    ax.invert_yaxis()
    ax.set_xlabel("test ROC-AUC")
    ax.set_title("Etiket başına ROC-AUC, modeller karşılaştırmalı")
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncols=min(4, len(shown)))
    rep.fig(fig, "results_per_tag_roc_auc", "0.5 = şans. Sıralama en iyi modele göre.")

    # PR-AUC'nin etiket oranıyla ilişkisi (lift)
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    for m in shown:
        d = pt[m]
        ax.scatter(d["prevalence"], d["pr_auc"], s=26, color=color[m], label=m, edgecolor=SURFACE, linewidth=1,
                   zorder=3, alpha=0.9)
    lim = [0, max(d["prevalence"].max() for d in pt.values()) * 1.05]
    ax.plot(lim, lim, color=MUTED, linestyle="--", linewidth=1)
    ax.text(lim[1], lim[1], "rastgele ", color=MUTED, ha="right", va="bottom", fontsize=7.5)
    # birbirine yakın noktaların etiketleri çakışmasın: piksel uzayında yakınsa sola veya alta kaydır
    fig.canvas.draw()
    placed = []
    for _, r in pt[best].nlargest(6, "pr_auc").iterrows():
        px, py = ax.transData.transform((r["prevalence"], r["pr_auc"]))
        for dx, dy, ha in ((5, 3, "left"), (-5, 3, "right"), (5, -10, "left"), (-5, -10, "right")):
            if all(abs(px + dx - qx) > 55 or abs(py + dy - qy) > 11 for qx, qy in placed):
                break
        placed.append((px + dx, py + dy))
        ax.annotate(r["tag"], (r["prevalence"], r["pr_auc"]), xytext=(dx, dy), textcoords="offset points",
                    fontsize=7.5, color=INK2, ha=ha)
    ax.set_xlabel("etiket oranı (test)")
    ax.set_ylabel("PR-AUC")
    ax.set_title("Sık etiketler daha mı kolay?")
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncols=min(4, len(shown)))
    rep.fig(fig, "results_pr_vs_prevalence", "Kesik çizginin üstündeki her nokta rastgeleden iyi; uzaklık kazancı "
                                             "gösterir.")

    t = pt[best].sort_values("pr_auc", ascending=False)
    cols = ["tag", "n_test_pos", "prevalence", "roc_auc", "pr_auc", "f1", "thr"]
    t = t[[c for c in cols if c in t]].round(4)
    rep.table(pd.concat([t.head(10), t.tail(10)]), "per_tag_best_worst",
              f"{best}: en iyi 10 ve en kötü 10 etiket (PR-AUC'ye göre)")


def curves_section(rep: Report, models, names, color):
    have = [m for m in names if "history" in models[m]["runs"][0]]
    if not have:
        rep.note("Eğitim eğrileri için results.json içinde 'history' yok (eski koşular); yeni koşularda otomatik "
                 "kaydedilir.")
        return
    rep.h2("5. Eğitim eğrileri (ilk seed)")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for m in have:
        r = models[m]["runs"][0]
        h = pd.DataFrame(r["history"])
        axes[0].plot(h["epoch"], h["train_loss"], color=color[m], linewidth=2, label=m)
        axes[1].plot(h["epoch"], h["val_pr_auc"], color=color[m], linewidth=2, label=m)
        be = r["best_epoch"]
        bv = h.loc[h["epoch"] == be, "val_pr_auc"]
        if len(bv):
            axes[1].scatter([be], bv, s=50, color=color[m], edgecolor=SURFACE, linewidth=2, zorder=4)
    for ax in axes:
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    axes[0].set_title("Eğitim kaybı")
    axes[0].set_xlabel("epoch")
    axes[1].set_title("Validation PR-AUC (nokta = seçilen epoch)")
    axes[1].set_xlabel("epoch")
    axes[1].legend(loc="lower right")
    rep.fig(fig, "results_training_curves", "Kayıp düşerken validation PR-AUC düşmeye başlıyorsa model ezberliyor "
                                            "(overfit). Kayıplar farklı kayıp tanımları/mixup nedeniyle modeller "
                                            "arasında doğrudan karşılaştırılamaz.")


def tsne_section(rep: Report, models, names, max_points=3000):
    from sklearn.manifold import TSNE

    cand = [(m, d) for m in names for d in models[m]["dirs"][:1] if (d / "test_probs_emb.npz").exists()]
    if not cand:
        return
    rep.h2("6. Test embedding'leri (t-SNE)")
    for m, d in cand[:3]:
        z = np.load(d / "test_probs_emb.npz")
        emb, probs, ids = z["emb"].astype(np.float32), z["probs"], z["ids"]
        rng = np.random.default_rng(0)
        sel = rng.choice(len(emb), min(max_points, len(emb)), replace=False)
        e = emb[sel]
        e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)  # kosinüs komşuluğu (P@k ile aynı)
        xy = TSNE(n_components=2, init="pca", perplexity=min(30, len(e) - 1), random_state=0).fit_transform(e)
        mp = Path(d) / "model.pt"
        vocab = None
        if mp.exists():
            try:
                import torch
                vocab = torch.load(mp, map_location="cpu", weights_only=False)["vocab"]
            except Exception:
                vocab = None
        # renk: modelin en olası etiketi, en sık 3 tahmin edilen etiket + diğer (dağılım grafiklerinde 3 renk sınırı)
        top1 = probs[sel].argmax(1)
        common = [c for c, _ in Counter(top1).most_common(3)]
        fig, ax = plt.subplots(figsize=(7, 6))
        other = ~np.isin(top1, common)
        ax.scatter(xy[other, 0], xy[other, 1], s=6, color=GRID, label="diğer", linewidth=0)
        for k, c in enumerate(common):
            msk = top1 == c
            lbl = vocab[c] if vocab else f"etiket {c}"
            ax.scatter(xy[msk, 0], xy[msk, 1], s=9, color=SERIES[k], label=lbl, linewidth=0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_title(f"{m}: test şarkılarının embedding haritası")
        ax.legend(loc="upper right", markerscale=2.5, title="en olası etiket")
        rep.fig(fig, f"results_tsne_{m.replace('/', '_')}", "Yakın noktalar modelin benzer bulduğu şarkılar (öneri "
                                                             "sistemi bu komşulukları kullanır).")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits-dir", help="train/validation/test TSV klasörü (ör. <emb-dir>/splits)")
    ap.add_argument("--mel-dir", help="pipeline.py --models mel çıktısındaki mel/ klasörü (isteğe bağlı)")
    ap.add_argument("--runs", nargs="*", default=[], help="summary.json içeren koşu klasörleri, 'klasör:etiket'")
    ap.add_argument("--out", default="viz")
    ap.add_argument("--top-k", type=int, default=25, help="birlikte görülme ısı haritasındaki etiket sayısı")
    ap.add_argument("--title", default="MTG-Jamendo mood/theme: veri ve sonuçlar")
    args = ap.parse_args()
    if not args.splits_dir and not args.runs:
        ap.error("--splits-dir veya --runs verin")
    rep = Report(Path(args.out))
    if args.splits_dir:
        data_section(rep, Path(args.splits_dir), Path(args.mel_dir) if args.mel_dir else None, args.top_k)
    if args.runs:
        results_section(rep, args.runs)
    rep.write(args.title)


if __name__ == "__main__":
    main()
