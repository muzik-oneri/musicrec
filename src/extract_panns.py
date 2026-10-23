"""PANNs (Cnn14) ile 2048 boyutlu embedding çıkarımı.

Giriş: 32 kHz mono pencereler, (pencere sayısı, örnek) biçiminde. Şarkı vektörü =
pencere embedding'lerinin ortalaması. Checkpoint (Cnn14_mAP=0.431.pth) ilk çalıştırmada
~/panns_data/ altına iner.
"""
import numpy as np
import torch


class PannsEmbedder:
    dim = 2048
    sr = 32000

    def __init__(self, device: str, batch: int = 16):
        from panns_inference import AudioTagging

        self.tagger = AudioTagging(checkpoint_path=None, device=device)
        self.batch = batch

    @torch.no_grad()
    def windows(self, windows32k: np.ndarray) -> np.ndarray:
        """(n, örnek) -> (n, 2048) pencere embedding'leri. Pencereler batch'ler halinde GPU'ya gider."""
        parts = []
        for i in range(0, len(windows32k), self.batch):
            _, emb = self.tagger.inference(windows32k[i:i + self.batch])
            parts.append(np.asarray(emb, dtype=np.float32))
        return np.concatenate(parts)

    def __call__(self, windows32k: np.ndarray) -> np.ndarray:
        """(n, örnek) -> (2048,) pencere ortalaması (eski tek vektör davranışı)."""
        return self.windows(windows32k).mean(0)


if __name__ == "__main__":
    # Hızlı duman testi: 2 pencere (1 sn sessizlik) -> tek vektör
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    emb = PannsEmbedder(dev)(np.zeros((2, 32000), dtype=np.float32))
    print("PANNs şarkı vektörü boyutu:", emb.shape)
