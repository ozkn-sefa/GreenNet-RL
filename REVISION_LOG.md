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

---

## 4. SNDlib Abilene Gerçek Dünya Trafik Verisi Entegrasyonu (2026-10-19)

### Değişen ve Eklenen Dosyalar
- `src/data/__init__.py`: Veri modülü dışa aktarımları (`AbileneDataLoader`, `RealTrafficSource`, özel istisnalar).
- `src/data/abilene_loader.py`: SNDlib Native 1.0 güvenli ayrıştırıcı, kronolojik bölme (%70/%15/%15), lazy loading & LRU önbellek, veri bütünlüğü ve kesinti doğrulayıcıları.
- `src/network/topology.py`: 12 düğümlü, 15 omurga bağlantılı `AbileneTopology` sınıfı ve `TopologyFactory` kaydı.
- `src/env/green_network_env.py`: İsteğe bağlı `traffic_source="real"` desteği, telemetri etiketleme, güvenli pod ayrıştırma, sentetik uyumluluk koruması.
- `tests/test_abilene_loader.py`: 15 yeni kapsamlı birim ve entegrasyon testi.
- `README.md`: Veri işleme, kronolojik bölme ve kullanım talimatları eklendi.
- `.gitignore`: `data/raw/` ve `data/processed/` dizinlerinin Git tarafından izlenmemesi sağlandı.

### Değişiklik Gerekçesi
- **Gerçekçi Doğrulama:** Sentetik simülasyona ek olarak operatör seviyesinde gerçek WAN trafiği (Yin Zhang Abilene 6 aylık veri seti) üzerinde enerji ve QoS davranışının değerlendirilebilmesi.
- **Güvenli Ayrıştırma ve Doğrulama:** META (`unit=MBITPERSEC`), NODES ve DEMANDS bölümlerinin doğrulanması; bozuk dosyalar, geçersiz düğümler, negatif trafik ve ölçüm kesintilerinin (`>5min`) tespit edilmesi.
- **Veri Sızıntısını Önleme:** 48.096 matrisin rastgele değil, katı kronolojik sırayla %70 Train (33.667), %15 Val (7.214) ve %15 Test (7.215) olarak ayrılması.
- **Hafıza ve Performans Optimizasyonu:** 48.096 dosyanın RAM'i tüketmesini önlemek için `abilene_manifest.json` dizinlemesi ve 1024 boyutlu LRU bellek önbelleği ile lazy loading uygulanması.
- **Sessiz Düşüşün (Silent Fallback) Engellenmesi:** Gerçek veri talep edildiğinde veri biterse veya eksikse sentetik veriye sessizce geçilmemesi (`AbileneMissingDataError` fırlatılması).

### Test ve Doğrulama Sonuçları
- **Test Süiti (`pytest -v`)**:
  - Mevcut 73 testin tamamı eksiksiz korundu.
  - 15 yeni test eklendi ve tamamı geçti (`88 passed in 21.37s`).
  - Model ağırlıkları veya PPO hiperparametreleri değiştirilmedi.

