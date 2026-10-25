"""LAION CLAP (resmi laion_clap kütüphanesi, müzik checkpoint'i) ile 512 boyutlu embedding çıkarımı.

Not: transformers 5.x'teki ClapModel uyarlaması bu checkpoint'lerde çökmüş çıktı veriyor
(beyaz gürültü bile müzikle 0.97 kosinüs benzerliğinde). Resmi kütüphanede şarkılar arası
benzerlik ~0.4, şarkı-gürültü ~0.08. Bu yüzden resmi kütüphane kullanılıyor.

Giriş: 48 kHz mono pencereler (n, 1440000). Her 30 sn'lik pencere 10 sn'lik parçalara bölünür,
her parça L2 normalize 512 boyutlu ses embedding'i verir, parçaların ortalaması pencere
embedding'idir: (n, 512). Metin sorgusuyla arama için aynı modelin get_text_embedding'i kullanılır.

Kurulum (numpy'ı düşürmemesi için --no-deps):
    pip install --no-deps laion_clap torchlibrosa ftfy braceexpand webdataset progressbar wget
"""
import os
import urllib.request
from pathlib import Path

import numpy as np
import torch

CKPT_URL = "https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt"


def _quantize(x: np.ndarray) -> np.ndarray:
    """Resmi örnekteki gibi int16'ya yuvarlayıp geri çevirir (eğitimdeki ön işlemeyle aynı)."""
    x = np.clip(x, -1.0, 1.0)
    return ((x * 32767.0).astype(np.int16) / 32767.0).astype(np.float32)


class ClapEmbedder:
    dim = 512
    sr = 48000
    chunk_sec = 10

    def __init__(self, device: str, batch: int = 24, name: str = "music_audioset_epoch_15_esc_90.14"):
        import laion_clap

        self.device = device
        self.batch = batch
        ckpt = Path(os.environ.get("CLAP_CKPT", Path.home() / ".cache" / "clap" / f"{name}.pt"))
        if not ckpt.exists():
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            print("CLAP checkpoint indiriliyor:", ckpt)
            urllib.request.urlretrieve(CKPT_URL.replace("music_audioset_epoch_15_esc_90.14", name), ckpt)
        self.model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base", device=device)
        self.model.load_ckpt(str(ckpt), verbose=False)
        self.model.eval()

    @torch.no_grad()
    def windows(self, w48k: np.ndarray) -> np.ndarray:
        """(n, örnek) -> (n, 512) pencere embedding'leri."""
        n, length = w48k.shape
        c = self.chunk_sec * self.sr
        k = length // c
        chunks = _quantize(w48k[:, : k * c].reshape(n * k, c))
        outs = []
        for i in range(0, len(chunks), self.batch):
            x = torch.from_numpy(chunks[i:i + self.batch]).to(self.device)
            e = self.model.get_audio_embedding_from_data(x=x, use_tensor=True)
            outs.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy())
        return np.concatenate(outs).reshape(n, k, self.dim).mean(1).astype(np.float32)

    def __call__(self, w48k: np.ndarray) -> np.ndarray:
        return self.windows(w48k).mean(0)


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x = np.random.randn(2, 48000 * 30).astype(np.float32) * 0.05
    print("CLAP pencere embedding boyutu:", ClapEmbedder(dev).windows(x).shape)
