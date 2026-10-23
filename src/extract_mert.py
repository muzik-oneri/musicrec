"""MERT-v1-95M ile katman bazlı embedding çıkarımı.

Giriş: 24 kHz mono pencereler (n, 720000). Her 30 sn'lik pencere 5 sn'lik parçalara bölünür
(MERT-v1 5 sn bağlamla eğitildi), her parça için tüm gizli katmanların (13 x 768) zaman ortalaması
alınır, parçaların ortalaması pencere embedding'i olur: (n, 13, 768).
Ağırlıklar HuggingFace'ten ilk çalıştırmada iner (Internet açık olmalı). Import hatası alırsan
`pip install nnAudio` dene.
"""
import numpy as np
import torch


class MertEmbedder:
    n_layers = 13
    dim = 768
    sr = 24000
    chunk_sec = 5

    def __init__(self, device: str, batch: int = 12, name: str = "m-a-p/MERT-v1-95M"):
        from transformers import AutoModel, Wav2Vec2FeatureExtractor

        self.device = device
        self.batch = batch
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(name, trust_remote_code=True).to(device).eval()

    @torch.no_grad()
    def windows(self, w24k: np.ndarray) -> np.ndarray:
        """(n, örnek) -> (n, 13, 768) pencere embedding'leri."""
        n, length = w24k.shape
        c = self.chunk_sec * self.sr
        k = length // c
        chunks = w24k[:, : k * c].reshape(n * k, c)
        outs = []
        use_amp = str(self.device).startswith("cuda")
        for i in range(0, len(chunks), self.batch):
            x = [np.asarray(a, dtype=np.float32) for a in chunks[i:i + self.batch]]
            inp = self.processor(x, sampling_rate=self.sr, return_tensors="pt", padding=True)
            inp = {key: v.to(self.device) for key, v in inp.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                hs = self.model(**inp, output_hidden_states=True).hidden_states
            h = torch.stack(hs, 0).float().mean(2)  # (13, B, 768): zaman ortalaması
            outs.append(h.permute(1, 0, 2).cpu().numpy())
        e = np.concatenate(outs).reshape(n, k, self.n_layers, self.dim).mean(1)
        return e.astype(np.float32)

    def __call__(self, w24k: np.ndarray) -> np.ndarray:
        """(n, örnek) -> (13, 768) şarkı vektörü (pencere ortalaması)."""
        return self.windows(w24k).mean(0)


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x = np.random.randn(2, 24000 * 30).astype(np.float32) * 0.05
    print("MERT pencere embedding boyutu:", MertEmbedder(dev).windows(x).shape)
