# Süreç: 17 Eylül – 31 Ekim

Bütün sayılar 600 pair üzerinde, aksi yazılmadıkça. "toplam" = dist + bayt. Kararı her zaman koşunun **SON** satırıyla (en iyi checkpoint) verdim.

---

## Dönem 1 — B0: sinüslü doğrudan decoder (17 Eyl – 4 Eki)
Notebook'lar: `gecmis/01`, `02`, `03`, `04`, `05`.

### Distortion gelişimi (FP32, bayt hariç)
| Tarih | Ne yaptım | dist |
|---|---|---|
| 18 Eyl | 96×128 codec | 0.8725 |
| 19–22 Eyl | small2 → embedding CE → margin → feature branch | 0.5476 → 0.5058 |
| 23–25 Eyl | 384×512 doğrudan decoder (28-d latent, sinüs, 229K parametre), margin + düşük LR | 0.3305 |
| 27–29 Eyl | boundary kaybı + ~20 EMA'lı warm restart | FP32 0.1750; INT8 0.1876, 216 KB, **toplam 0.3315** |
| 30 Eyl | eski kayıp, düşük LR | 0.17246 |
| 1 Eki | L7 kaybı | 0.16898 |
| 29 Eyl | latent 28 → 36 | B3'ün başlangıcı sonradan bu model oldu |

Gerçek karelerin 384×512 gidiş-dönüş tabanı ≈ 0.065.

### Maskeler (1–4 Eki): kapattım
- Şerit sınıfı (1) tam maske + "boyacı" ağı E849'da 0.1076; sınıf 3 maskesi eklenince **FP32 0.0999**. Ama tam maske kayıpsız ~105–113 KB, toplamda kendini ödemiyor.
- Kayıpsız maske kodlaması (sayım tabanlı bağlam, 11–16 bit bağlam, zamansal, hareket telafili): sınıf 1 tek başına ~105–111 KB'ın altına inmedi.
- 1/8 kaba maske ve öğrenilen şerit ızgarası: mekanizma çalıştı ama sabit maske maliyeti yüzünden kâra geçmedi.

### bf16 hatası (4 Eki)
Eğitimde SegNet/PoseNet'i bf16 autocast ile çalıştırıp ölçümü FP32 yapıyordum; decoder bf16'ya göre optimize olmuş. O dönemin QAT / maskesiz / ilk 1/8 maske sonuçları **geçersiz**. O günden beri kuralım: eğitim ölçümle aynı hassasiyette (FP32; conv'da TF32 kabul, ölçümde TF32 kapalı).

### İlk QAT denemeleri (4 Eki): yanlış kapattım
- FP32 kaynak 0.15981; INT8 PTQ 0.16715 / 261.5 KB → toplam 0.341.
- Oran farkında QAT (kanal başına LSQ adımı, Laplace bit modeli): r3_v2 ~207 KB / 6.3 bit, dist +0.028, **toplam 0.326**. r10_v2: 175 KB, toplam 0.340.
- Sinüs → GELU kademeli geçiş: alfa arttıkça dist 0.33 → 0.67, durdurdum.
- O günkü hükmüm "sinüslü decoder düşük bite dayanmıyor" idi. **B3'te yanlış olduğunu gördüm:** 50–80 epoch'luk koşulara dayanıyordu (#95'in QAT bütçesinin ~%0.5'i).

### Kâhin deneyi (6 Eki)
Ressam ve eğitim olmadan, gerçek segmentasyon haritasını basit görüntülere çevirip SegNet'in ne kadar doğru okuduğunu ölçtüm. Yerleşimi mükemmel bilen bir temsil için bu bir tavan ölçümüydü.

### Bu dönemde işe yaramayanlar
96×128'de bikübik/nearest büyütme, sınıf ağırlıklandırma, detail head, boundary ring, son katmanı açmak; sinüs→GELU; latent 28→36 (o kurulumda); HNeRV makale mimarisi (PSNR ~34 dB ama dist 9.05); smooth disagreement kaybı (o kurulumda); kaba şerit maskeleri; elle ipuçları.

---

## Dönem 2 — B2: görev farkında not ızgarası + küçük ressam (5–16 Eki)
Notebook'lar: `gecmis/06`, `07`, `08`, `09`.

Mimari: pair başına 1/64, 1/32, 1/16 çözünürlükte 2'şer kanallı tamsayı not ızgaraları (2016 değer/pair, entropi kodlu) + paylaşılan küçük GELU "ressam" (45K, sonra 73K parametre). Tarif: FP32 ön eğitim → bit cezalı aşama (Laplace bit modeli, λ) → notlar donuk ressam ince ayarı → devam turları.

| Ölçek | En iyi | Not |
|---|---|---|
| 60 pair pilot | 0.261 | 0.200 dist / 46 KB not + 44 KB ressam |
| 200 pair | 0.341 | FP32 uzun eğitim 0.0898 → model yeterli, eksik eğitilmiş |
| 600 pair FP32 | dist **0.0835** | kapasite sorun değil |
| 600, devam turları | 0.368 → 0.344 → 0.333 → 0.328 | her turun kazancı bir öncekinin ~%46'sı |
| 600, geniş ressam + QAT8 | **0.312** | |
| 600, FiLM devamı | **0.295** | B2'nin en iyisi; duvar ~0.295–0.31 |

B3'e taşıdığım dersler:
- Kalite LR → 0'da geliyor. Ara satırlar yanıltıcı; yalnız SON satırını okuyorum.
- Dönüşümlü optimizasyon / düşük tepeli devam turları işe yarıyor ama sönüyor.
- QAT8 bedava (8 bit kaybını tamamen siliyor).
- 120 pair tezgâhı iyimser (aynı tarif 600'de +0.09–0.10).
- Notlar 600'de %97–98 sıfır; bilgi ressama kayıyor → decoder ağırlıklı yapıya geçiş.
- `laplace_bits` işaret hatası ve seviye çöküşü (b_min ile çözdüm) B2'ye özgüydü.
- Gerçek arşivi (constriction range coder) bir kez doğruladım (200 pair, 77 KB).

---

## Dönem 3 — B3: decoder ağırlıklı sinüslü HNeRV (16–27 Eki)
Notebook: `b3_ressam600_v1.ipynb`. Kaynak: 4 Ekim'deki 36 latentli sinüslü decoder (FP32 dist 0.1598, seg 0.1255, pose 0.0343).

### Faz A — 8 bit QAT turları (17–20 Eki)
Tarif: QAT8 (Conv kanal başına, Linear tensör başına), latent FP32 eğitilip ölçümde 8 bit, lr 1e-4 / 1e-3, batch 20, 10 epoch ısınma + kosinüs, EMA ufku 25 epoch, tur başına 1500 epoch, her tur bir öncekinin en iyisinden taze kosinüs. Ölçüm yalnızca EMA ile (sonradan önemli çıktı).

| Tur | 8 bit dist | seg / pose | KB | toplam |
|---|---|---|---|---|
| duman (E1) | 0.1659 | | 224.4 + 20.4 | 0.329 |
| kapı1 | 0.1384 | 0.1046 / 0.0338 | 224.8 + 20.0 | 0.301 |
| kapı7 | 0.1046 | 0.0738 / 0.0308 | 225.3 + 20.0 | 0.268 |
| **kapı8** | **0.1028** | 0.0724 / 0.0305 | 225.3 + 20.0 | **0.266** |

- QAT'lı ağ FP32 kaynağın altına indi (0.138 < 0.160).
- Her tur E100'de ~+0.007 yeniden başlama vergisi; kapı7→8 −0.0018 → 8 bitte doygun. 8 tur ≈ 12,000 epoch.

### Faz B — bit genişliği (20–25 Eki)
| Koşu | Tarif | dist | KB | toplam | Not |
|---|---|---|---|---|---|
| q6 | kapı8'den 6 bit (latent yanlışlıkla 6 bit) | 0.1595 | 164.8 + 14.5 | 0.279 | tek koşuda iki değişken (hatam) |
| q6 (latent 8) | 6 bit, latent 8 | 0.1453 | 164.8 + 20.0 | 0.268 | |
| q6k | + küçük katmanlar 8 bit | 0.1432 | 166.2 + 20.0 | 0.267 | 8 bitle berabere |
| lr3e5 | kapı8'den, lr 3e-5/3e-4, 8 bit | 0.1016 | 225.4 + 20.0 | 0.265 | düşük LR vergiyi +0.007 → +0.002'ye indirdi |
| **q7** | lr3e5'ten 7 bit | 0.1139 | 195.3 + 20.0 | **0.257** | A2 tahmini (~0.258) tuttu |

Eğitimsiz teşhisler:
- **A1 katman duyarlılığı** (kapı8, tek katman 8→6): kayıp bayta orantılı; derin bloklar bayt başına daha kırılgan; `rgb_1` aşırı kırılgan (0.1 KB için +0.0083).
- **A2 bit taraması** (lr3e5, eğitimsiz): 8 bit 0.1016 / 225 KB · 7: 0.1305 / 195 · 6.5: 0.1555 / 180 · 6: 0.1901 / 165 · 5.5: 0.2558 / 150 · 5: 0.3681 / 133. QAT kalan payı 6 bitte 0.46, 7 bitte 0.43.
- **A3 budama** (q7, eğitimsiz): %10 → +0.039 dist / −5.8 KB. Ölü.

### Faz C — hijyen ve canlı final (25 Eki)
Her ölçüme canlı + EMA ölçümü ve epoch ortalama eğitim kaybı ekledim.

| Koşu | Tarif | Sonuç |
|---|---|---|
| **q7hij** | q7'den, 7 bit, lr 3e-5/3e-4, 600 epoch | koşu boyunca canlı EMA'dan 0.02–0.04 kötü; **E600'de (lr 0) canlı 0.1066 → toplam 0.250**; EMA 0.1149 |
| q7hij2 | aynı tarif tekrar | yine 0.250 → kısa tur doygun |
| q7uzun | 4000 epoch tek kosinüs, lr 3e-5 (devam altyapısıyla) | **0.248** (6.5 saatte −0.002) |

### Faz D — ağırlık entropi cezası (26 Eki)
`yumusak_kb`: katman başına doğrusal çekirdekli yumuşak histogram entropisi. Doğrulama: yuvarlanmış ağda yumuşak KB = gerçek − 4.7 KB (sabit fp16 ölçek + bias) → birebir. Kayıp += λ · rate_of_kb(yumuşak KB). Eğitim kaybı gerçek dist'in ~5 katı olduğundan λ≈5 baytın gerçek fiyatı.

| Koşu | Başlangıç | λ | dist | decoder KB | toplam |
|---|---|---|---|---|---|
| ent5 | q7uzun | 5 | 0.1065 | 186.8 | **0.244** |
| ent10 | q7uzun | 10 | | | 0.251 (fazla sert) |
| **ent5b** | ent5 | 5 | ~0.104 | ~185 | **0.240** ← en iyi |
| ent5c | ent5b | 5 | | | 0.242 → doygun |

Bayt başına dist maliyeti ≈ 0.00024/KB, yani baytın fiyatının (0.00067) altında: ceza kârlı.

### Faz E — yüksek LR denemesi (26–27 Eki)
hlr: ent5b'den, lr 1e-4/1e-3, λ=5, 1500 epoch. Başarısız: KB 183 → 167 düştü ama eğitim kaybı LR düşerken yükseldi, E1500 canlı toplam 0.252. Yumuşak histogram yüksek LR'de kararsız.

### B3 bütçesi
ent5b'nin soy hattı ≈ 20,800 QAT epoch'u (kapı1–8 12,000 + lr3e5 1,500 + q7 1,500 + q7hij 600 + q7uzun 4,000 + ent5 600 + ent5b 600). Karşılaştırma: #95 ~29,650 epoch, ~19,500'ü QAT.

---

## Ajan araştırmaları (26–30 Eki)
[commaArastirma](https://github.com/wHamza/commaArastirma) reposunda Claude Code üzerinde bir orkestratör + alt ajan sistemi kurdum (yol kâşifi, challenge arkeoloğu, literatür tarayıcı, verim analisti, eleştirmen, deney tasarımcısı).
- **HNeRV tavanı:** benim zincirimle 0.2 altı ancak #95 tarifinin sıfırdan yeniden üretimiyle (~53–70 A100 saati, yaklaşık yazı tura).
- **GPU-verimli yollar** (hedef ilk 10, HPAC ailesi hariç): en ucuz yol #128'i alıp cilalamak.

## #128 cilası (30 Eki)
Ayrıntı: [128-cila.md](128-cila.md). #128'i birebir yeniden ürettim (0.187961), cilaların hepsi sıfır ya da negatif çıktı. Kapattım.

## B4 — damıtma (31 Eki –)
Notebook: `b4_damitma_v1.ipynb`, henüz koşmadım.
- **Hipotez:** benim dist'im (0.104) ile #128'in dist'i (0.070) arasındaki fark kapasite değil, havza: QAT öncesi yüksek LR'li FP32 fazım yok.
- **Öğrenci:** ent5b (aynı kanallar, latent 36 ≥ 28); kapasite sorunu yok.
- **Faz 1:** FP32, değerlendiricisiz, öğretmen çıktısına L1, 2000 epoch. Karar: FP32 dist ≤ ent5b − 0.01.
- **Faz 2:** 7 bit QAT + görev kaybı (λ=0, 1500 epoch), sonra λ=5 ile 600 epoch ve canlı final.
- Toplam ~4–4.5 A100 saati. Liderlik için kazanç beklemiyorum (öğrenci ≈ öğretmen); amaç kendi modelimle 0.2 altı ve havza hipotezini sınamak.
