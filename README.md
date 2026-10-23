# musicrec

MTG-Jamendo mood/theme veri setinde VGGish ve PANNs (Cnn14) embedding'lerini karşılaştırma hattı. Modeller dondurulmuş öznitelik çıkarıcı olarak kullanılır, üzerine aynı MLP eğitilir.

## Yapı

```
src/common.py          TSV okuma, etiket sözlüğü, tam şarkı yükleme + 30 sn pencereleme
src/extract_panns.py   PANNs Cnn14 -> 2048 boyut (pencere ortalaması)
src/extract_vggish.py  VGGish -> 256 boyut (tüm karelerin ortalaması + std)
src/pipeline.py        tar-tar indir, embedding çıkar, .npz yaz, sesi sil (devam edebilir)
src/train.py           MLP eğitimi, çok seed, macro ROC-AUC / PR-AUC / F1, ortalama ± std
kaggle/                Kaggle notebook'ları (embedding çıkarımı, eğitim)
```

## Kaggle'da çalıştırma

1. Repoyu GitHub'a push et (aşağıda), `kaggle/embeddings_notebook.ipynb` dosyasındaki `REPO_URL` satırını kendi repona göre düzenle (varsayılan: zehranozdogan/musicrec).
2. kaggle.com > New Notebook > File > Import Notebook ile `kaggle/embeddings_notebook.ipynb` dosyasını yükle.
3. Sağ panel: Accelerator **GPU T4 x2** (veya P100), Internet **On** (telefon doğrulaması gerekir).
4. Önce `--limit-tars 2` ile kısa bir deneme koş, süreyi ölç. Sonra o satırı kaldırıp **Save Version > Save & Run All (Commit)** ile çalıştır. Commit modu tarayıcı kapalıyken en fazla 12 saat koşar.
5. Bittiğinde Output sekmesinden `embeddings_full` klasörünü bir Kaggle Dataset (örn. `musicrec-emb-full`) olarak kaydet.
6. Süre yetmezse: bu dataset'i yeni commit'e Input olarak ekle, notebook'taki `--prev-dir` satırının yorumunu kaldır. Biten tar'lar atlanır.
7. `kaggle/train_notebook.ipynb` ile eğitimi koş (GPU gerekmez, dakikalar sürer). Input olarak embedding dataset'ini ekle.

## Yerelde

```bash
pip install -r requirements.txt
git clone https://github.com/MTG/mtg-jamendo-dataset /data/mtg
python src/pipeline.py --mtg-dir /data/mtg --out-dir embeddings_full --work-dir /data/work
python src/train.py --emb-dir embeddings_full --model panns vggish --seeds 1 2 3 4 5
```

## Embedding hattı v2 (pencere bazlı, çoklu model)

`src/pipeline.py --models panns vggish mert clap` aynı geçişte seçilen modelleri çalıştırır (indirme ve ses çözme paylaşılır) ve embedding'leri **pencere bazında** `float16` yazar (`<model>/<tar>.npz`: `ids`, `n_win`, `emb`). MERT-v1-95M katman bazlı (13 x 768), CLAP (laion/larger_clap_music) 512 boyutludur. Kaggle notebook'u: `kaggle/embeddings_v2_notebook.ipynb` (önce `--limit-tars 2` ile süre ölç). `extract_mert.py` ve `extract_clap.py` HuggingFace `transformers` ister (`requirements-extra.txt`). Bu çıktı biçimi eski şarkı-vektörü biçimiyle karıştırılmaz (`config.json` içindeki `format`).

`train.py` her iki biçimi okur. Pencere bazlı dosyalarda pencereler şarkı başına ortalanır. `--model` tek model (`mert`, `clap`) veya birleşim (`panns+vggish+mert`, `concat` = `panns+vggish`) alır. `--layers 6 9 12` MERT için katman seçer (verilmezse tüm katmanların ortalaması).

## Notlar

- Şarkının TAMAMI kullanılır: şarkı bir kez 32 kHz'de çözülür, ceil(süre/30 sn) adet eşit aralıklı 30 sn'lik pencereye bölünür (en fazla `--max-windows`, varsayılan 20 = 10 dk; daha uzun şarkılarda pencereler seyreltilir). PANNs pencere embedding'lerinin ortalamasını, VGGish 16 kHz'e indirilmiş pencerelerin tüm karelerinin ortalama+std'sini alır. `--max-windows 1` eski tek kesit davranışıdır.
- `config.json` pencere ayarlarını kaydeder; `--prev-dir` ile farklı ayarlı (veya eski 30 sn kesit) çıktıyla devam etmeye izin verilmez.
- Bir sonraki tar, mevcut tar işlenirken arka planda indirilir. Hata oranı %20'yi aşarsa hat durur.
- Eğitim `--seeds` ile birden fazla seed koşar; `runs/summary.md` ortalama ± std ve modeller arası eşleştirilmiş farkı verir.
- Hazır split-0 dosyaları kullanılır, kendin bölme. Dosyalar yoksa `pipeline.py` `split_filter_subset.py` ile üretir.
- Eşikler validation üzerinde etiket başına F1'e göre seçilir, test'e uygulanır.
- ROC-AUC, PR-AUC ve F1 aynı etiket kümesi (test'te en az bir pozitifi olan etiketler) üzerinden macro ortalanır. `summary.md` rastgele baseline PR-AUC'yi (ortalama prevalence) ve rastgele komşu seçiminin P@k değerini de yazar; 0.13 gibi bir PR-AUC'yi bunlara göre yorumla.
- Komşu getirme (P@5, P@10): test şarkıları arasında kosinüs benzerliğiyle en yakın k komşudan kaçı sorguyla en az bir etiketi paylaşıyor. Tavsiye hedefine en yakın ölçüt budur.
- `--model concat`: PANNs (2048) + VGGish (256) embedding'lerini track_id üzerinden birleştirir (2304 boyut). Örn. `--model panns vggish concat`; fark tablosu ilk iki modeli karşılaştırır.
- `--min-pos N`: test'te N'den az pozitifi olan etiketleri dışarıda bırakan ek (*) metrikleri de raporlar.
- `per_tag.csv` etiket başına prevalence, F1 ve seçilen eşiği de içerir.
- Etiketler dengesiz. Düşük kalan etiketler için `train.py --pos-weight` dene.
- `ImportError` veya numpy sürüm uyuşmazlığı çıkarsa pip kurulumundan sonra çekirdeği (kernel) yeniden başlat.
- Bozuk dosyaların id'leri `embeddings/failed.txt` içine yazılır.
