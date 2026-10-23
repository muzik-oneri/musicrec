"""Ortak yardımcılar: TSV okuma, etiket sözlüğü, tam şarkı yükleme ve pencereleme."""
import csv
import random

import numpy as np
import torch
from torch.utils.data import Dataset

SEED = 42
WIN_SEC = 30
SR_PANNS = 32000
SR_VGGISH = 16000
DATASET = "autotagging_moodtheme"
SPLITS = ("train", "validation", "test")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tag_name(tag: str) -> str:
    """'mood/theme---happy' -> 'happy'"""
    return tag.split("---", 1)[-1]


def read_split(tsv_path):
    """MTG-Jamendo TSV'sini okur.

    Sütunlar: TRACK_ID, ARTIST_ID, ALBUM_ID, PATH, DURATION, TAGS...
    Etiket sayısı satıra göre değiştiği için 5. sütundan sonrası etiket kabul edilir.
    """
    rows = []
    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)  # başlık satırı
        for r in reader:
            if len(r) < 5:
                continue
            rows.append(
                {
                    "track_id": int(r[0].split("_")[-1]),
                    "path": r[3],
                    "duration": float(r[4]),
                    "tags": [tag_name(t) for t in r[5:] if t],
                }
            )
    return rows


def build_vocab(rows_by_split) -> list:
    return sorted({t for rows in rows_by_split.values() for r in rows for t in r["tags"]})


def make_windows(y: np.ndarray, win: int, max_windows: int) -> np.ndarray:
    """Şarkıyı win örneklik pencerelere böler, çıktı (n, win).

    - Şarkı win'den kısaysa tekrarlanarak tek pencereye uzatılır.
    - Aksi halde ceil(uzunluk / win) pencere, başlangıçları eşit aralıklı seçilir:
      şarkının tamamı örtülür, pencereler en fazla birkaç saniye çakışır.
    - max_windows > 0 ve şarkı daha uzunsa pencereler yine eşit aralıkla seyreltilir.
    - Tek pencere kalırsa şarkının ortasındaki pencere alınır (max_windows=1 -> eski 30 sn davranışı).
    """
    n = len(y)
    if n == 0:
        raise ValueError("boş ses")
    if n <= win:
        return np.resize(y, win)[None, :]
    k = int(np.ceil(n / win))
    if max_windows and k > max_windows:
        k = max_windows
    if k == 1:
        starts = np.array([(n - win) // 2])
    else:
        starts = np.linspace(0, n - win, k).round().astype(int)
    return np.stack([y[s:s + win] for s in starts])


def to_16k(w32: np.ndarray) -> np.ndarray:
    """32 kHz pencereleri (n, örnek) tam 2 katı oranla 16 kHz'e indirir."""
    from scipy.signal import resample_poly

    assert SR_PANNS == 2 * SR_VGGISH
    return resample_poly(w32, 1, 2, axis=-1).astype(np.float32)


def resample_windows(w32: np.ndarray, sr: int) -> np.ndarray:
    """32 kHz pencereleri (n, örnek) hedef örnekleme hızına getirir (16, 24, 32 veya 48 kHz)."""
    from scipy.signal import resample_poly

    if sr == SR_PANNS:
        return w32
    ratios = {16000: (1, 2), 24000: (3, 4), 48000: (3, 2)}
    if sr not in ratios:
        raise ValueError(f"desteklenmeyen örnekleme hızı: {sr}")
    up, down = ratios[sr]
    return resample_poly(w32, up, down, axis=-1).astype(np.float32)


def load_track(path) -> np.ndarray:
    """Şarkının tamamını 32 kHz mono olarak çözer (tek decode, tek yeniden örnekleme)."""
    import librosa

    y, _ = librosa.load(str(path), sr=SR_PANNS, mono=True)
    return y.astype(np.float32)


class TrackDataset(Dataset):
    """items: (track_id, dosya_yolu, süre) listesi.

    Her eleman (track_id, pencereler, hata) döner: pencereler (n, 30 sn @ 32 kHz) float32 tensör,
    hata yoksa None, çözme başarısızsa pencereler None ve hata metni dolu.
    """

    def __init__(self, items, max_windows: int):
        self.items = items
        self.max_windows = max_windows

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        tid, path, _dur = self.items[i]
        try:
            y = load_track(path)
            w = make_windows(y, WIN_SEC * SR_PANNS, self.max_windows)
            return tid, torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32)), None
        except Exception as e:  # noqa: BLE001
            return tid, None, f"{type(e).__name__}: {e}"
