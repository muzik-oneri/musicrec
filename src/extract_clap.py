"""LAION müzik CLAP'ı ile 512 boyutlu embedding çıkarımı (HuggingFace transformers).

Giriş: 48 kHz mono pencereler (n, 1440000). Her 30 sn'lik pencere 10 sn'lik parçalara bölünür,
her parça L2 normalize 512 boyutlu ses embedding'i verir, parçaların ortalaması pencere
embedding'idir: (n, 512). Metin sorgusuyla arama için aynı model get_text_features ile kullanılır.
"""
import numpy as np
import torch


class ClapEmbedder:
    dim = 512
    sr = 48000
    chunk_sec = 10

    def __init__(self, device: str, batch: int = 24, name: str = "laion/larger_clap_music"):
        from transformers import ClapFeatureExtractor, ClapModel

        self.device = device
        self.batch = batch
        self.fe = ClapFeatureExtractor.from_pretrained(name)
        self.model = ClapModel.from_pretrained(name).to(device).eval()

    @torch.no_grad()
    def windows(self, w48k: np.ndarray) -> np.ndarray:
        """(n, örnek) -> (n, 512) pencere embedding'leri."""
        n, length = w48k.shape
        c = self.chunk_sec * self.sr
        k = length // c
        chunks = w48k[:, : k * c].reshape(n * k, c)
        outs = []
        for i in range(0, len(chunks), self.batch):
            x = [np.asarray(a, dtype=np.float32) for a in chunks[i:i + self.batch]]
            inp = self.fe(x, sampling_rate=self.sr, return_tensors="pt")
            inp = {key: v.to(self.device) for key, v in inp.items()}
            feats = self.model.get_audio_features(**inp)
            if not torch.is_tensor(feats):  # yeni transformers sürümlerinde çıktı nesnesi olabilir
                p = getattr(feats, "pooler_output", None)
                feats = p if p is not None else feats[0]
            if feats.shape[-1] != self.dim:
                raise RuntimeError(f"CLAP ses embedding boyutu {feats.shape[-1]}, {self.dim} bekleniyordu")
            feats = torch.nn.functional.normalize(feats.float(), dim=-1)
            outs.append(feats.cpu().numpy())
        return np.concatenate(outs).reshape(n, k, self.dim).mean(1).astype(np.float32)

    def __call__(self, w48k: np.ndarray) -> np.ndarray:
        return self.windows(w48k).mean(0)


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x = np.random.randn(2, 48000 * 30).astype(np.float32) * 0.05
    print("CLAP pencere embedding boyutu:", ClapEmbedder(dev).windows(x).shape)
