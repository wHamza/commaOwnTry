# models

## `b3_ent5b.pt` (toplam 0.240)

| | |
|---|---|
| Mimari | HNeRV sinüs decoder, kanallar [36,36,36,27,20,18,18], ~242.8K parametre |
| Latent | pair başına 36 boyut, 600 pair |
| Nicemleme | decoder 7 bit QAT (Conv kanal başına, Linear tensör başına ölçek) |
| Eğitim | L7 seg kaybı + √(10·pose), entropi cezası λ=5, canlı final |
| dist | ≈ 0.104 |
| Boyut | ~205 KB (ideal entropi tahmini) |

Dosya biçimi:
```python
{
    "model": state_dict,   # "codes" [600, 36] + "dec.*"
    "bits": ...,           # decoder ağırlık biti (7)
    "kod_bits": ...,       # latent kod biti
    "kucuk": ...,          # küçük tensörlerin (bias vb.) saklanma ayarı
}
```

## Yükleme
Sınıflar `notebooks/b4_damitma_v1.ipynb` 4. bölümde (`HNeRVDecoder`, `B3`, `kaynak_yukle`):
```python
ck = torch.load("models/b3_ent5b.pt", map_location="cpu", weights_only=False)
m = B3(600, ck["model"]["codes"].shape[1])
m.load_state_dict(ck["model"])
frames = m(torch.arange(4))   # [4, 2, 3, 384, 512], 0–255
```

## Ölçüm
Aynı notebook'taki `olc(m, bits=ck["bits"], kod_bits=ck["kod_bits"])` fonksiyonu decoder'ı ve kodları checkpoint'teki bitlere yuvarlar, 600 pair'de FP32 (TF32 kapalı) dist'i ve entropi KB'ını hesaplar. 4. bölümün sonundaki sağlama hücresi bu modelde 0.240 civarını vermeli; vermezse GT çözücüsü (PyAV + `yuv420_to_rgb`) resmi değerlendiriciden farklı demektir.
