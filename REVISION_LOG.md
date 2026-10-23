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

---

## 5. Gerçek Abilene Trafik Verisi Üzerinde CPU ve GPU (RTX 4060) PPO Performans Karşılaştırması ve Cihaz Desteği (2026-10-23)

### Yapılan Değişiklikler
- **`src/agents/rl_agent.py`**:
  - `GreenRLAgent.__init__` metoduna `device: str = "auto"` parametresi eklendi ve `self.device = device` olarak saklandı.
  - `_init_model` içinde `PPO` ve `DQN` başlatılırken `device=self.device` argümanı geçildi.
  - `load` metodunda `PPO.load` ve `DQN.load` çağrılarına `device=self.device` eklendi.
- **`scripts/train.py`**:
  - `--device` argümanı (`auto`, `cuda`, `cpu`) eklendi.
  - `--topology`, `--traffic-source` (`synthetic`, `real`), `--real-split` (`train`, `val`, `test`), `--seed` ve `--max-steps-per-episode` parametreleri CLI'ya eklendi.
  - `traffic_source="real"` seçildiğinde gerçek Abilene veri setinin `train` split'i ile güvenli başlatılması sağlandı.
- **`scripts/benchmark_ppo_device.py`**:
  - Sıfır harici paket bağımlılığıyla (Windows standart `ctypes` Win32 API ve `torch.cuda` kullanarak) CPU vs GPU karşılaştırması yapan, RAM/VRAM tepe noktalarını, adım sürelerini ve PPO rollout kayıp metriklerini kaydeden benchmark runner eklendi.
  - Çıktılar `results/cpu_gpu_benchmark_results.json` dosyasına yazıldı.

### Donanım ve Ortam Bilgileri
- **İşletim Sistemi**: Windows 11
- **CPU**: 16 Mantıksal Çekirdek
- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM)
- **CUDA & PyTorch**: PyTorch 2.5.1+cu121, CUDA Aktif
- **SNDlib Abilene Veri Seti**: 48.096 matris; kronolojik %70 Train (33.667 matris, indeks: 0 - 33.666).

### Gerçek Veri Kaynağı ve İzolasyon Doğrulaması
- **Gerçek Trafik Kaynağı Doğrulandı**: `GreenNetworkEnv` `traffic_source="real"` ve `real_traffic_split="train"` ile başlatıldı; `RealTrafficSource` nesnesinin doğrudan SNDlib Abilene train split verilerini beslediği, sentetik fallback'e kesinlikle düşülmediği teyit edildi.
- **Bölme İzolasyonu**: Yalnızca `train` split (0 - 33.666) kullanıldı; `val` ve `test` split'leri tamamen izole tutuldu.
- **Mevcut Modeller Korundu**: Ana modeller (`models/ppo_green_agent*.zip`) değiştirilmedi; benchmark ağırlıkları izole olarak `models/benchmark/` dizinine kaydedildi.

### Gerçek Ölçüm Sonuçları (10.000 Timestep, Seed=42, PPO MlpPolicy [128, 128])

| Metrik | CPU | CUDA (RTX 4060 Laptop) | Karşılaştırma / Kazanan |
| :--- | :---: | :---: | :---: |
| **Toplam Süre** | **26.93 s** | 38.35 s | **CPU %29.78 daha kısa sürede tamamladı** |
| **Eğitim Hızı (FPS)** | **371.27 steps/s** | 260.75 steps/s | **CPU %42.39 daha hızlı (1.42x)** |
| **Süreç RAM (RSS Sonrası)** | **601.99 MB** | 1355.77 MB | CPU 753.78 MB daha az RAM kullandı |
| **Süreç Tepe RAM (Peak)** | **602.92 MB** | 1355.77 MB | CPU %55.5 daha az tepe bellek tüketti |
| **Tepe GPU VRAM (Peak)** | **0.0 MB** | 18.06 MB | GPU üzerinde küçük model tahsisi |
| **Son Eğitim Kaybı (Loss)** | 0.146247 | 0.154606 | Dengeli yakınsama |
| **Politika Kaybı (Policy Loss)** | -0.002241 | -0.000144 | Benzer gradyan stabilitesi |
| **Değer Kaybı (Value Loss)** | 0.434209 | 0.431207 | Eşdeğer değer kestirimi |
| **Açıklanan Varyans (Exp. Var.)** | 0.479526 | 0.562418 | CUDA'da hafif daha yüksek açıklanan varyans |
| **Model Kayıt Yolu** | `models/benchmark/ppo_abilene_cpu_10000.zip` | `models/benchmark/ppo_abilene_cuda_10000.zip` | İzole dizin korundu |

### Sonuç ve Cihaz Seçim Analizi (Neden CPU Daha Hızlı?)
1. **Model Boyutu ve Ağ Mimarisi**: Politika ağı `[128, 128]` 2 gizli katmandan oluşan küçük bir MLP'dir (parametre sayısı ~35K).
2. **Transfer ve Kernel Başlatma Yükü (Overhead)**: Çevre simülasyonu (ağ grafiği, ECMP routing ve QoS hesaplamaları) CPU üzerinde çalışmaktadır. Küçük tensörlerin her rollout ve adımda CPU RAM'inden GPU VRAM'ine PCIe veri yolu üzerinden kopyalanması ve CUDA kernel fırlatma gecikmeleri, GPU'nun paralel işlem gücü avantajından daha baskın çıkmaktadır.
3. **Öneri**: Abilene topolojisi üzerindeki PPO eğitimleri için **CPU seçeneği** hem %42.39 daha yüksek FPS sunmakta hem de sistem bellek ayak izini yarıdan fazla düşürmektedir.

### Test Doğrulaması
- `python -m pytest tests -q` çalıştırıldı:
  - **88 passed in 11.29s** (Tüm testler eksiksiz ve yeşil).

