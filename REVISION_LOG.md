# GreenNet-RL Revizyon Günlüğü (REVISION_LOG.md)

## 1. Yapılan Değişiklikler
`src/agents/baselines.py` dosyasındaki kural ve eşik tabanlı ajanların (`ThresholdLoadAgent` ve `StaticHeuristicAgent`) gözlem vektöründen okuduğu float32 değerleri üzerinde 5 ondalık basamak yuvarlama (`round(float(...), 5)`) uygulandı:

- **`ThresholdLoadAgent.predict`**:
  ```python
  # Önceki:
  norm_demand = observation[-2]

  # Güncellenen:
  norm_demand = round(float(observation[-2]), 5)
  ```
- **`StaticHeuristicAgent.predict`**:
  ```python
  # Önceki:
  diurnal_factor = observation[-3]

  # Güncellenen:
  diurnal_factor = round(float(observation[-3]), 5)
  ```

## 2. Değişikliğin Gerekçesi
- **Problem**: `GreenNetworkEnv` gözlem vektörünü `np.float32` veri tipiyle üretmektedir. IEEE-754 32-bit kayan nokta temsilinde `0.35` sayısı tam olarak temsil edilemeyip ikili tabanda `0.3499999940395355...` olarak saklanır.
- Python float64 eşiği olan `0.35` ile karşılaştırıldığında `0.3499999940395355 < 0.35` ifadesi `True` vermekte; bu durum `ThresholdLoadAgent`'ın `demand=0.35` sınır değerinde 3 çekirdek (`actions[2]`) yerine bir alt basamak olan 2 çekirdeği (`actions[1]`) seçmesine yol açarak `test_threshold_core_counts_k4[0.35-3]` testinin başarısız olmasına neden olmaktaydı.
- **Çözüm**: Gözlem vektöründen alınan float32 değerlerinin `round(float(...), 5)` ile 5 ondalık basamağa yuvarlanması, float32 hassasiyetinden kaynaklanan yuvarlama gürültüsünü (`~1e-8`) tamamen temizler ve `0.15`, `0.35`, `0.55`, `0.6`, `1.0` gibi tüm eşik sınırlarında tutarlı ve kararlı davranış sağlar. Karşılaştırma mantığı veya test beklentileri değiştirilmeden minimum müdahale ile sorun giderilmiştir.

## 3. Test ve Doğrulama Sonuçları
- **Birim Testler (`python -m pytest tests -v`)**:
  - 73 testin tamamı başarıyla geçti (`73 passed`).
  - `test_threshold_core_counts_k4[0.35-3]` testi başarıyla doğrulandı.
- **Benchmark Doğrulaması (`python scripts/benchmark.py`)**:
  - Benchmark çalıştırıldı ve `results/benchmark_summary.json` dosyasında herhangi bir sapma/değişiklik olmadığı (sonuçların korunduğu) teyit edildi.
