"""Log-mel spektrogram için sıfırdan eğitilen CRNN (Convolutional Recurrent Neural Network).

Fikir (Choi vd. 2017, müzik etiketleme için CRNN): konvolüsyon katmanları kısa süreli, yerel ses
desenlerini (tını, vuruş, akor dokusu) çıkarır; ardından gelen tekrarlayan katman (çift yönlü GRU)
bu desenlerin ZAMAN İÇİNDEKİ sırasını ve gelişimini modelller (yükselen gerilim, tekrar eden nakarat,
tempo değişimi). Saf CNN zamanı yalnızca ortalama/max havuzla özetler; CRNN önce zaman dizisini okur.

Yapı:
- Giriş (B, 1, mel, kare); mel bantları ayrı ayrı normalize edilir (resnet.py ile aynı).
- 4 konv blok (her biri 2 x [3x3 konv + BN + ReLU] + max havuz). Kanallar w-2w-4w-4w.
  Frekans 64 -> 4 (her blokta /2), zaman /8 (son blok yalnızca frekansı küçültür).
  31.25 kare/sn girişte GRU'ya saniyede ~3.9 adım düşer (15 sn kesit -> 58 adım, 30 sn -> 117 adım).
- Frekans ekseni ortalanır -> (B, adım, 4w) dizisi -> 2 katmanlı çift yönlü GRU (gizli boyut h).
- GRU çıktıları zaman boyunca ortalama + max ile özetlenir -> 4h boyutlu embedding
  (h=128 için 512). Bu vektör öneri sistemi için şarkı vektörüdür.

w=64, h=128, 64 mel: ~2.9 M parametre (küçük CNN 1.2 M ile ResNet-34 21 M arasında).
"""
import torch
import torch.nn as nn


def conv_block(cin, cout, pool):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(pool),
    )


class CRNN(nn.Module):
    def __init__(self, n_tags, n_mels=64, width=64, rnn_hidden=128, rnn_layers=2, dropout=0.5, rnn_dropout=0.3):
        super().__init__()
        w = width
        self.bn_mel = nn.BatchNorm2d(n_mels)
        # havuzlar (frekans, zaman): zaman toplamda /8, frekans /16
        self.convs = nn.Sequential(
            conv_block(1, w, (2, 2)),
            conv_block(w, 2 * w, (2, 2)),
            conv_block(2 * w, 4 * w, (2, 2)),
            conv_block(4 * w, 4 * w, (2, 1)),
        )
        self.rnn = nn.GRU(4 * w, rnn_hidden, num_layers=rnn_layers, batch_first=True, bidirectional=True,
                          dropout=rnn_dropout if rnn_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(4 * rnn_hidden, n_tags)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def embed(self, x):  # x: (B, 1, mel, kare)
        x = self.bn_mel(x.transpose(1, 2)).transpose(1, 2)
        h = self.convs(x).mean(2)  # frekans ortalaması -> (B, C, adım)
        h = h.transpose(1, 2).contiguous()  # (B, adım, C)
        # GRU float32'de çalışır (AMP altında fp16 yinelemeli hesap kararsızlaşabilir)
        with torch.autocast(device_type=h.device.type, enabled=False):
            out, _ = self.rnn(h.float())  # (B, adım, 2h)
        return torch.cat([out.mean(1), out.amax(1)], dim=1)  # (B, 4h)

    def forward(self, x, return_emb=False):
        e = self.embed(x)
        out = self.fc(self.drop(e))
        return (out, e) if return_emb else out
