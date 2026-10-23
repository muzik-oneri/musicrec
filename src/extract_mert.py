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
        # Yeni transformers sürümlerinde uzak MERT kodu output_hidden_states=True olsa da hidden_states
        # döndürmeyebilir (None). Katman girişi/çıkışlarını hook ile yakalayıp bunu aşıyoruz:
        # 1. katmanın girişi + 12 katmanın çıkışı = 13 gizli durum (HubertEncoder'ın hidden_states'i ile aynı).
        self._captured = []
        layers = self.model.encoder.layers
        layers[0].register_forward_pre_hook(self._grab_input, with_kwargs=True)
        for layer in layers:
            layer.register_forward_hook(self._grab_output)

    def _grab_input(self, module, args, kwargs):
        self._captured.append(args[0] if args else kwargs.get("hidden_states"))

    def _grab_output(self, module, args, output):
        self._captured.append(output[0] if isinstance(output, tuple) else output)

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
            self._captured.clear()
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                out = self.model(**inp, output_hidden_states=True)
            hs = getattr(out, "hidden_states", None)
            if hs is None:
                hs = tuple(self._captured)
            if len(hs) != self.n_layers:
                raise RuntimeError(f"MERT {len(hs)} gizli durum verdi, {self.n_layers} bekleniyordu")
            h = torch.stack(list(hs), 0).float().mean(2)  # (13, B, 768): zaman ortalaması
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
