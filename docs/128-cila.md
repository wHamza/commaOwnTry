# 30 Ekim: #128 üzerinde cila denemeleri

Amacım kamu gönderimi #128'i (HNeRV ailesinin en iyisi, 0.187) yeniden eğitmeden, ucuz eklerle ilk 10 eşiğine (0.166) yaklaştırmaktı. Sonuç: kapattım.

Kaynak: PR #128 `submissions/rhnerv_latent_polish` (MIT), arşiv 176,531 B.

## 1. Yeniden üretim
Resmi `evaluate.sh --device cpu`:

| | pose | seg | bayt | skor |
|---|---|---|---|---|
| Benim ölçümüm | 0.00002937 | 0.00053279 | 176,531 | **0.187961** |

Bot ile birebir. Değerlendirme hedeflerini resmi yoldan (PyAV + `yuv420_to_rgb`) yeniden ürettim; bu aynı zamanda eski hedeflerimin GT kontrolü yerine geçti.

## 2. K1: kare-0 artığı
Pair başına 6 sürekli parametre (geometrik kayma + parlaklık), Adam 150 adım, 3–6 bit nicemleme. Dist kazancı parametrelerin baytını karşılamadı: **net zarar**.

## 3. Kanal bias araması
#98'den gelen 6 kanal bias sabiti üzerinde ±1 açgözlü arama (GPU, eşik 2e-5). Hiçbir değişiklik kabul edilmedi; CPU raporu #128 ile birebir. Sabitler **yerel optimumda**.

## 4. K3: seg yamaları (ön ölçüm)
Hata yapısını ölçtüm (bkz. [bulgular.md](bulgular.md#hnerv-ailesinin-seg-tabanı)):
- Medyan küme 1 px, en büyük 8 px.
- ≥ 4 hatalı 8×8 hücrelerde hataların %9'u, ≥ 8 hatalıda %0.5 (32 hücre), ≥ 16'da hiç yok.
- En kötü %10 pair hataların yalnız %14'ü.

Hatalar dağınık 1 piksellik sınır titremesi; yamalar kendini ödemez. **K3 ölü.**

## Hüküm
#128, HPAC yasağı içinde ve yeniden eğitimsiz yollarla yerel optimumda. Yeniden eğitimli seçenekler: #95 tarifini sıfırdan üretmek (~60 A100 saati) ya da kendi modelime dönmek (B4 damıtma). B4'ü seçtim.
