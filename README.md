# commaOwnTry

[comma video sıkıştırma challenge'ı](https://github.com/commaai/comma_video_compression_challenge) için kendi modellerimizle yaptığımız denemelerin kaydı: Eylül–Ekim 2026, Colab A100.

> **English abstract.** Personal research log for the comma video compression challenge. We train a small HNeRV-style implicit decoder (~243K params, 36-dim latent per frame pair) with quantization-aware training (7-bit), an entropy penalty and the task losses (SegNet/PoseNet). Our best own model reaches **0.240** total score (dist ≈ 0.104, ~205 KB). We also reproduce the public #128 submission exactly (0.18796) and show that the remaining segmentation error of this family is symmetric 1-pixel boundary jitter, so cheap post-hoc fixes cannot pay for their bytes. This is not a challenge submission.

## Skor
```
skor = 100 · seg + √(10 · pose) + 25 · bayt / 37,545,489
```
`seg` uyuşmayan SegNet piksellerinin oranı, `pose` PoseNet MSE'si. 1 KB ≈ 0.00067 skor. Liderlik ölçümü CPU'da.

## Sonuçlar

| Aşama | Model | Toplam skor |
|---|---|---|
| B0 | İlk HNeRV, INT8 | 0.33 (FP32 dist 0.175) |
| B2 | Not ızgarası + ressam (73K) + FiLM kodları | 0.295 |
| B3 kapı 1 | HNeRV 36 latent, QAT | 0.301 |
| B3 kapı 8 | uzun koşu | 0.266 |
| B3 q7 | 7 bit QAT | 0.257 |
| B3 q7hij | canlı final (LR→0, EMA yerine) | 0.250 |
| B3 q7uzun | daha uzun koşu | 0.248 |
| B3 ent5 | entropi cezası λ=5 | 0.244 |
| **B3 ent5b** | **ent5 devamı, en iyi** | **0.240** |
| Referans | #128 (kamu arşivi), birebir yeniden ölçüm | 0.18796 |

En iyi model: [`models/b3_ent5b.pt`](models/). Yükleme ve ölçüm: [`models/README.md`](models/README.md).

## İçerik
```
models/b3_ent5b.pt              en iyi kendi modelimiz (state_dict + bit ayarları)
notebooks/b3_ressam600_v1.ipynb B3: eğitim, QAT, entropi cezası, katman duyarlılığı, budama (çıktılarıyla)
notebooks/b4_damitma_v1.ipynb   B4: #128'i öğretmen alıp kendi modelimize damıtma (henüz koşulmadı)
docs/surec.md                   B0 → B4 kronolojisi: ne denendi, ne çıktı
docs/bulgular.md                ölçülmüş bulgular ve olumsuz sonuçlar
docs/128-cila.md                30 Eki: #128 üzerinde cila denemeleri ve seg hata yapısı
```

## Öne çıkan bulgular
1. **Canlı final EMA'yı geçiyor.** QAT'lı küçük ağda LR→0'da canlı model 0.1066, EMA 0.1149 dist. "Son epoch kötüleşmesi" yalnız EMA ölçümünden geliyordu.
2. **λ skordan türetilebiliyor.** Eğitim kaybı / gerçek dist ≈ 5 olduğundan λ≈5 baytın skordaki fiyatına denk. İlk denemede 0.248 → 0.244.
3. **Budama bu ölçekte ölü.** Entropi kodlu küçük INR'de küçük ağırlıklar zaten ucuz; %10 budama kazandırdığından çok kaybettiriyor.
4. **HNeRV ailesinin seg tabanı yönsüz sınır titremesi.** #128'in hatalı piksellerinin %99.8'i sınıf sınırına ≤ 1 px, %82'si tek piksel; sistematik kısım yalnız %9. Yerel ya da ortak düzeltme baytını ödeyemiyor.
5. **#128 ucuz cila için yerel optimumda.** Kare-0 artığı, kanal bias araması ve seg yamaları sıfır ya da negatif kazanç verdi.

Ayrıntılar [`docs/bulgular.md`](docs/bulgular.md).

## Ortam
Colab A100 40 GB, PyTorch 2.x. Notebook'lar Drive'da `deneme/` klasörünü (değerlendirici ağırlıkları, `0.mkv`, hedef önbelleği) ve challenge reposunu bekler; yollar ilk hücrelerde. 600 pair'de bir HNeRV epoch'u ~6 sn (batch 20).

## İlgili
- Ajan tabanlı yol araştırması (HNeRV tavanı, GPU-verimli yollar): [wHamza/commaArastirma](https://github.com/wHamza/commaArastirma)
- Damıtma ve karşılaştırmada kullanılan kamu gönderimleri: challenge PR #95 ve #128 (MIT). Onların ağırlıkları bu repoya konmadı; B4 notebook'u kurulum sırasında indirir.

## Not
Bu repo bir yüksek lisans tez çalışmasının deney kaydıdır, challenge gönderimi değildir. Notebook'lar ve dokümanlar Claude (Anthropic) yardımıyla hazırlandı.
