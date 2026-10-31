# Süreç: B0 → B4

Bütün koşular Colab A100 40 GB'ta, 600 pair (1200 kare) üzerinde. Karar her zaman toplam skorla, gerçek arşivdeki bit sayısında ölçülen dist ile verildi.

## B0 (Eylül): ilk HNeRV
- Pair başına latent + ortak decoder. FP32 dist 0.175.
- INT8'e düz nicemleyince toplam 0.33. Ders: nicemleme açığı, dist'ten büyük. QAT şart.

## B2 (Ekim ortası): not ızgarası + ressam
- Pair başına çok ölçekli "not" ızgaraları (1/64, 1/32, 1/16), entropi kodlu; ortak ressam ağı (~73K parametre, 8 bit QAT).
- Oran–bozulma eğitimi (Laplace bit tahmini, λ=30), C3 tarzı yumuşak yuvarlama denemeleri, bağlamlı entropi modeli.
- FiLM kodları (pair başına 8 boyut) ile en iyi: **0.295** (dist 0.215, 119.7 KB).
- Ders: ızgaralar KB'ın büyük kısmını yiyor, kaba seviyeler %97–98 sıfır taşıyor. 0.2 için bu mimari yetersiz kaldı.

## B3 (17–26 Ekim): HNeRV'e dönüş, kendi eğitim zincirimiz
Model: sinüs aktivasyonlu HNeRV decoder, kanallar [36,36,36,27,20,18,18], latent 36. Kayıp: L7 (marj tabanlı seg) + √(10·pose).

| Koşu | Değişen | Toplam |
|---|---|---|
| kapı 1 | ilk QAT koşusu | 0.301 |
| kapı 8 | uzun eğitim | 0.266 |
| A1 | katman duyarlılık haritası (eğitimsiz) | — |
| A2 | bit ızgarası taraması → 7 bit seçimi | — |
| A3 | budama | olumsuz |
| q6 / q6k | 6 bit | doygun, 7 bitin gerisinde |
| q7 | 7 bit QAT | 0.257 |
| q7hij | canlı final (LR→0, EMA yerine canlı model) | 0.250 |
| q7uzun | daha uzun | 0.248 |
| ent5 | entropi cezası λ=5 (yumuşak histogram) | 0.244 |
| ent10 | λ=10 | 0.251 (kötü) |
| **ent5b** | ent5'ten devam | **0.240** |
| ent5c | bir tur daha | doygun |
| hlr | yüksek LR'de entropi cezası | kararsız |

Altyapı: QAT (STE; Conv kanal başına, Linear tensör başına ölçek), ideal entropi KB ölçümü, `os.replace` ile atomik kayıt ve kaldığı yerden devam.

## Ajan araştırmaları (26–30 Ekim)
Ayrı repo: [commaArastirma](https://github.com/wHamza/commaArastirma).
- **HNeRV tavanı:** bizim zincirle 0.2 altı ancak #95 tarifinin sıfırdan yeniden üretimiyle (~53–70 A100 saati, ~yazı tura).
- **GPU-verimli yollar** (hedef ilk 10, HPAC ailesi yasak): en ucuz yol #128'i alıp cilalamaktı.

## 30 Ekim: #128 + cila (kapandı)
#128 resmi CPU değerlendirmesiyle birebir yeniden üretildi (0.187961). Cila denemelerinin hepsi sıfır ya da negatif çıktı. Ayrıntı: [128-cila.md](128-cila.md).

## B4 (31 Ekim–): damıtma
Notebook: [`notebooks/b4_damitma_v1.ipynb`](../notebooks/b4_damitma_v1.ipynb). Henüz koşulmadı.
- **Hipotez:** bizim dist (0.104) ile #128'in dist'i (0.070) arasındaki fark kapasite değil, havza: QAT öncesi yüksek LR'li FP32 fazımız yok.
- **Öğrenci:** ent5b (aynı kanallar, latent 36 ≥ #128'in 28'i), yani kapasite sorunu yok.
- **Faz 1:** FP32, değerlendiricisiz, öğretmen çıktısına L1, 2000 epoch. Karar: FP32 dist ≤ ent5b − 0.01.
- **Faz 2:** 7 bit QAT + görev kaybı (λ=0, 1500 ep), sonra λ=5 ile 600 ep ve canlı final.
- Toplam ~4–4.5 A100 saati. Liderlik için kazanç beklenmiyor (öğrenci ≈ öğretmen); amaç kendi modelimizle 0.2 altı ve havza hipotezini sınamak.
