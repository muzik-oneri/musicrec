"""VGGish ile embedding çıkarımı.

Giriş: 16 kHz mono pencereler, (pencere sayısı, örnek). Her pencere 0.96 sn'lik karelere
bölünür, her kare 128 boyutlu embedding verir. Şarkı vektörü = tüm pencerelerin tüm
karelerinin ortalaması ve standart sapması (128 + 128 = 256 boyut). Ağırlıklar torch.hub
ile ilk çalıştırmada iner.
"""
import numpy as np
import torch


class VggishEmbedder:
    dim = 256
    sr = 16000

    def __init__(self, device: str):
        # postprocess=False: PCA/quantize adımı yok, ham 128 boyutlu embedding
        self.model = torch.hub.load(
            "harritaylor/torchvggish",
            "vggish",
            postprocess=False,
            device=device,
            trust_repo=True,
        )
        self.model.eval()

    @torch.no_grad()
    def windows(self, windows16k: np.ndarray) -> np.ndarray:
        """(n, örnek) -> (n, 256): her pencerenin karelerinin ortalaması ve std'si."""
        out = []
        for w in windows16k:
            f = self.model.forward(w, fs=16000).cpu().numpy()  # (kare sayısı, 128)
            if f.ndim == 1:
                f = f[None, :]
            out.append(np.concatenate([f.mean(0), f.std(0)]))
        return np.stack(out).astype(np.float32)

    @torch.no_grad()
    def __call__(self, windows16k: np.ndarray) -> np.ndarray:
        """(n, örnek) -> (256,). Her pencere ayrı ayrı modele verilir (genel API, forward(x, fs))."""
        frames = []
        for w in windows16k:
            f = self.model.forward(w, fs=16000).cpu().numpy()  # (kare sayısı, 128)
            if f.ndim == 1:
                f = f[None, :]
            frames.append(f)
        frames = np.concatenate(frames)
        return np.concatenate([frames.mean(0), frames.std(0)]).astype(np.float32)


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t = np.arange(16000 * 5) / 16000
    wav = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    print("VGGish şarkı vektörü boyutu:", VggishEmbedder(dev)(np.stack([wav, wav])).shape)
