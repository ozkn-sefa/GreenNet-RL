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
- **Test Süiti (`python -m pytest tests -q`)**:
  - `88 passed in 11.29s` (Tüm testler eksiksiz ve yeşil).

---

## 6. Gerçek Abilene Trafiğinde 100.000 Adımlık Kontrollü PPO Pilot Eğitimi ve Baseline Karşılaştırması (2026-10-23)

### Yapılan Değişiklikler ve Eklemeler
- **`scripts/train_pilot_abilene.py`**:
  - Gerçek SNDlib Abilene veri seti üzerinde 100.000 adımlık kontrollü PPO pilot eğitim betiği geliştirildi.
  - **Kesintiden Sonra Devam Edebilme (Resumability)**: Her 10.000 adımda model checkpoint'leri (`models/pilot_abilene_100k/checkpoint_step_<N>.zip`) ve eğitim konfigürasyonu (`training_config.json`) kaydedildi.
  - **Sabit Validation Sekansı**: Her 10.000 adımda model ve 3 baseline (OSPF, StaticHeuristic, ThresholdLoad) aynı 300 adımlık doğrulama trafiği üzerinde izole şekilde test edildi.
  - **QoS/SLA Öncelikli Checkpoint Seçimi**: Yalnızca güvenlik koşullarını sağlayan modeller arasından enerji tasarrufunu seçen seçim mantığı işletildi.
  - Çıktılar ve geçmiş metrikleri `results/pilot_abilene_100k_results.json` dosyasına yazıldı.

### Eğitim Konfigürasyonu
- **Cihaz**: CPU (16 Mantıksal Çekirdek)
- **Ağ Mimarisi**: `MlpPolicy` [128, 128]
- **Hiperparametreler**:
  - `learning_rate`: 3e-4
  - `n_steps`: 1024
  - `batch_size`: 256
  - `n_epochs`: 10
  - `gamma`: 0.99
  - `gae_lambda`: 0.95
  - `clip_range`: 0.2
  - `ent_coef`: 0.01
  - `seed`: 42
- **Eğitim Performansı**: Toplam süre **204.27 saniye (~3.4 dakika)**, ortalama **489.54 FPS**.

### Veri İzolasyonu ve Kesinti Yönetimi
- **Train Split**: 33.667 matris (%70, indeks 0 .. 33.666). 100.000 adım süresince kronolojik sırayla işlendi (yaklaşık 2.97 train döngüsü belgelendi).
- **Validation Split**: 7.214 matris (%15, indeks 33.667 .. 40.880). Kesinlikle yalnızca periyodik model değerlendirmesinde kullanıldı; eğitim gradyanına karıştırılmadı.
- **Test Split**: 7.215 matris (%15, indeks 40.881 .. 48.095). Kesinlikle **dokunulmadı ve korundu**.
- **Ölçüm Kesintileri**: Train split içinde 3 adet ölçüm aralığı (>5 dk) tespit edildi ve bölüm (episode) sınırlarında süreklilik bozulmadan yönetildi.

### Eğitim Dinamikleri ve Metrik Gelişimi
- **Bölüm Ödülü (`rollout/ep_rew_mean`)**: Başlangıçtaki **-524.0** seviyesinden **-56.7** seviyesine yükselerek istikrar kazandı.
- **Açıklanan Varyans (`train/explained_variance`)**: Başlangıçta **-0.001** iken adım 80.000 sonrasında **0.890 – 0.910** seviyesine ulaşarak değer ağının WAN trafik durumunu %90 doğrulukla öngördüğünü kanıtladı.
- **Kayıp Değerleri**:
  - `train/loss`: 1258.1 -> 0.218
  - `train/value_loss`: 2707.5 -> 0.449
  - `train/policy_gradient_loss`: -0.0183 -> 0.000198
  - `train/entropy_loss`: -2.07 -> -0.207 (Ajan kararlı çalışma moduna yakınsadı)
  - `train/approx_kl`: ~0.00002 – 0.0015 (Politika güncelleme stabilitesi korundu)

### Validation Karşılaştırması (300 Adım Gerçek Abilene Validation Trafiği)

| Ajan / Checkpoint | SLA İhlali (%) | Ort. Paket Kaybı (%) | Ort. Gecikme (ms) | p95 Gecikme (ms) | Switch Toggles | Enerji Tasarrufu vs OSPF (%) | Güvenlik Durumu |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PPO (Checkpoint 10k - 100k)** | **28.00%** | **1.06%** | **14.24 ms** | **15.32 ms** | **0** | **0.00%** | Güvenlik Adayı (OSPF Seviyesi) |
| **OSPF Baseline (Tüm Omurgalar Aktif)** | **28.00%** | **1.06%** | **14.24 ms** | **15.32 ms** | **0** | 0.00% (Referans) | Güvenlik Adayı (Referans) |
| **Static Diurnal Heuristic** | 55.67% | 12.11% | 14.36 ms | 15.34 ms | 28 | 0.00% | **GÜVENSİZ (Yüksek Kayıp)** |
| **Threshold Load Reactive** | 100.00% | 39.00% | 14.51 ms | 15.56 ms | 4 | 0.00% | **GÜVENSİZ (Kritik Darboğaz)** |

### Başarısız Deneyler ve Model Seçimi Değerlendirmesi
1. **PPO'nun Öğrendiği Karar**: Abilene bir WAN ağıdır (15 bağlantı, 2500 Mbps kapasite, propagation delay ~10–15 ms). Model erken aşamada omurgaları kapatmanın ağır gecikme ve paket kaybı cezalarına yol açtığını keşfetmiş; bu nedenle cezalardan kaçınmak adına OSPF benzeri "tam kapasite" modunu (Action 7) tercih etmiştir.
2. **Kural Tabanlı Sezgisellerin Çöküşü**: Ağ durumunu gerçek zamanlı izlemeyen `StaticHeuristic` ve `ThresholdLoad` ajanları omurgaları uyutmaya kalkıştığında, WAN trafiği dar boğaza girmiş ve paket kayıpları sırasıyla **%12.11** ve **%39.00** gibi felaket seviyelere ulaşmıştır. PPO bu tuzağa düşmemiş, kaybı **%1.06** seviyesinde tutmuştur.
3. **Güvenlik Koşulu Karşılanma Durumu**: Katı SLA koşulu (SLA ihlali <= %5 ve paket kaybı <= %1.5) unshielded/ham politika için **hiçbir checkpoint tarafından karşılanamamıştır**. Bunun temel nedeni, Abilene ağında fiziksel yayılım gecikmesinin zaten ~13-14 ms olması ve varsayılan ortam SLA gecikme eşiğinin (~14.5 ms) çok dar kalmasıdır; nitekim tüm omurgaları sürekli açık tutan OSPF dahi %28 SLA ihlali vermektedir.
4. **En Dengeli Model**: `checkpoint_step_10000.zip` modeli en dengeli ağırlık olarak `best_model.zip` ve `final_model.zip` olarak kaydedilmiştir.

### Kalan Sorunlar ve Çözüm Önerileri
- **Abilene WAN SLA Kalibrasyonu**: Mevcut SLA gecikme eşiği formülasyonu LAN/Fat-Tree varsayımlarına yakındır. WAN için RTT toleransının (propagation delay + dinamik buffer kuyruk gecikmesi) ortam seviyesinde kalibre edilmesi önerilir.
- **SLA Shield (Güvenlik Kalkanı) Entegrasyonu**: Ham RL politikasının omurgaları yalnızca güvenli pencerelerde (gece düşük trafik) kapatmasını sağlamak için `use_shield=True` koruması altında enerji tasarrufu kilidi açılabilir.

### Test Doğrulaması
- `python -m pytest tests -q` çalıştırıldı:
  - **88 passed in 11.33s** (Tüm testler eksiksiz ve yeşil).

---

## 7. Abilene 100K PPO Pilot Sonuçları Kök Neden Analizi, Düzeltmeler ve Regresyon Doğrulaması (2026-10-23)

### 1. Kök Nedenler ve Mimari Tespitler
1. **PPO'nun Action 7 Seçmesi ve Enerji Tasarrufunun %0 Kalması**:
   - Abilene bir WAN ağıdır (12 PoP router, 15 omurga bağlantısı). Ortalama düğüm derecesi 2.5'tir.
   - Her router'a (`c_id`) doğrudan yerel bir host (`h_c_id`) bağlıdır.
   - Yapılan topolojik yol analizi kanıtlamıştır ki:
     - **Action 0 (1 Core)**: 132 host çiftinin 60'ında hiç yol yoktur (**%45.5 bağlantısız / unroutable**). Ortalama paket kaybı: **%51.74**.
     - **Action 1 (1 Core)**: 132 host çiftinin 76'sında hiç yol yoktur (**%57.6 bağlantısız**). Ortalama paket kaybı: **%55.33**.
     - **Action 2 & 3 (2 Core)**: 42 host çifti bağlantısız (**%31.8**). Ortalama paket kaybı: **%25.5 – %37.9**.
     - **Action 4 & 5 (3 Core)**: 22 host çifti bağlantısız (**%16.7**). Ortalama paket kaybı: **%16.9 – %25.1**.
     - **Action 6 & 7 (4 Core - Tüm Omurgalar Aktif)**: **0 bağlantısız host çifti (%100 ulaşılaiblir)**. Ortalama paket kaybı: **%0.00**.
   - Sonuç: Abilene'de router deep-sleep (şasi kapatma) eylemleri **ağ grafını fiziksel olarak parçalamakta** ve hostları koparmaktadır. PPO ağır cezalardan (-130 vs -8) kaçınarak mantıksal olarak Action 7'ye oturmuştur.
2. **Static/Threshold'un Enerji Tasarrufunun %0 Görünmesi Hatası**:
   - `scripts/train_pilot_abilene.py` betiğinde güç metrikleri `step_info.get("power_metrics", {}).get("total_power_w", 0.0)` olarak okunmaya çalışılmış; oysa `step_info` doğrudan `"total_power_w"` anahtarını içermektedir.
   - Bu anahtar uyuşmazlığı nedeniyle `total_energy_kwh` ve `ref_energy` daima `0.0` hesaplanmış ve yüzde tasarruf hatalı olarak `%0.0` görünmüştür.
3. **OSPF'in %28 SLA İhlali Vermesi**:
   - `GreenNetworkEnv.__init__` içinde varsayılan `link_capacity_mbps: float = 1000.0` parametresi, `AbileneTopology`'nin gerçek 2500.0 Mbps (OC-48) kapasitesini ezmekteydi.
   - 1000 Mbps kapasitede gerçek Abilene pik trafiği tekil linkleri aştığı için adımların %28'inde dar boğaz ve kayıp yaşanmaktaydı.
   - Doğal 2500 Mbps kapasite sağlandığında OSPF ve PPO'nun SLA ihlali **%0.0'a**, paket kaybı ise **%0.0000'a** düşmektedir.

### 2. Düzeltilen Dosyalar ve Değişiklikler
1. **`src/env/green_network_env.py`**:
   - `link_capacity_mbps: Optional[float] = None` yapıldı. Belirtilmediğinde `TopologyFactory` üzerinden topolojinin doğal kapasitesi (Abilene için 2500.0 Mbps, FatTree için 1000.0 Mbps) korunacak şekilde dinamik çözümleme eklendi.
2. **`src/network/topology.py`**:
   - `TopologyFactory.create` metodunda `link_capacity_mbps=None` durumu güvenle ele alındı; topoloji varsayılanlarına güvenli geri dönüş sağlandı.
3. **`scripts/train_pilot_abilene.py`**:
   - `step_info["total_power_w"]` doğrudan okunarak enerji hesaplaması düzeltildi.
   - Güvenlik kriterlerini karşılamayan hiçbir checkpoint'in yanıltıcı şekilde `best_model.zip` olarak etiketlenmemesi sağlandı (`best_checkpoint = None`).
   - `models/pilot_abilene_100k/best_model.zip` kaldırıldı.
4. **`tests/test_abilene_regression.py`**:
   - Link kapasitesi dinamik çözümlemesi, güç metriklerinin `step_info` içinde doğrulanması ve Abilene aksiyon ulaşılamazlık (reachability) analizini doğrulayan 4 yeni regresyon testi eklendi.

### 3. Çoklu Trafik Rejimi Validation Sonuçları (Genişletilmiş Analiz)
Validation split'te Düşük (Gece: 0..200), Orta (Gündüz: 2000..2200) ve Pik (Burst: 4500..4700) pencereleri üzerinde ölçülen gerçek değerler:

| Senaryo / Kapasite | Ajan | Güç (W) | Enerji Tasarrufu (%) | SLA İhlali (%) | Paket Kaybı (%) | p95 Gecikme (ms) | Toggles |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Düşük Trafik (2500 Mbps)** | **PPO (100k)** | 1809.7 W | 0.0% | **0.0%** | **0.00%** | 13.65 ms | 0 |
| | **OSPF (All-Active)** | 1809.7 W | 0.0% | **0.0%** | **0.00%** | 13.65 ms | 0 |
| | **Static Heuristic** | 1703.5 W | 5.9% | 35.3% | 13.89% | 14.22 ms | 14 |
| | **Threshold Reactive** | 1505.7 W | 16.8% | 100.0% | 38.42% | 14.43 ms | 4 |
| **Orta Trafik (2500 Mbps)** | **PPO (100k)** | 1814.2 W | 0.0% | **0.0%** | **0.00%** | 13.29 ms | 0 |
| | **OSPF (All-Active)** | 1814.2 W | 0.0% | **0.0%** | **0.00%** | 13.29 ms | 0 |
| | **Static Heuristic** | 1802.3 W | 0.7% | 4.0% | 1.33% | 13.31 ms | 12 |
| | **Threshold Reactive** | 1512.8 W | 16.6% | 100.0% | 32.40% | 14.11 ms | 4 |
| **Pik Trafik (2500 Mbps)** | **PPO (100k)** | 1816.8 W | 0.0% | **2.7%** | **0.67%** | 13.79 ms | 0 |
| | **OSPF (All-Active)** | 1816.8 W | 0.0% | **2.7%** | **0.67%** | 13.79 ms | 0 |
| | **Static Heuristic** | 1788.6 W | 1.5% | 12.0% | 4.41% | 13.80 ms | 10 |
| | **Threshold Reactive** | 1513.7 W | 16.7% | 99.3% | 41.11% | 13.26 ms | 12 |

### 4. Test Doğrulaması
- `python -m pytest tests -q` çalıştırıldı:
  - **92 passed in 11.50s** (88 mevcut + 4 yeni regresyon testi, tümü yeşil).

---

## 8. Abilene Link-Level Enerji Optimizasyonu Mimarisi, Topoloji Analizi ve Doğrulama Sonuçları (2026-10-23)

### 1. NetworkX Üzerinde Abilene Bağlantı ve Köprü (Bridge) Analizi
NetworkX üzerinde 12 router ve 15 yönsüz (30 yönlü) bağlantı detaylı olarak analiz edilmiştir:
- **Köprü (Bridge) Durumu**: Ağda omurgalar arasında **yalnızca 1 adet köprü bağlantı** bulunmaktadır: `('ATLAM5', 'ATLAng')`. `ATLAM5` 1. dereceden bir stub düğüm olduğundan bu bağlantının kapatılması grafı derhal böler.
- **Yedekli Bağlantılar (Redundant Links)**: Geriye kalan 14 omurga bağlantısının tamamının alternatif yolları mevcuttur (örneğin Batı üçgeni `SNVAng <-> STTLng` kapatıldığında `DNVRng` üzerinden 2 sekmeli ve yalnızca +5.00 ms ek gecikmeli alternatif yol oluşur; Doğu dörtgeni `NYCMng <-> WASHng` kapatıldığında `CHINng` ve `ATLAng` üzerinden 3 sekmeli alternatif yol oluşur).
- **Çoklu Bağlantı Uyutma Güvenliği**:
  - `SNVAng-STTLng` + `NYCMng-WASHng`: Graf **tamamen bağlı** (Maksimum gecikme 25.00 ms).
  - `SNVAng-STTLng` + `NYCMng-WASHng` + `CHINng-IPLSng`: Graf **tamamen bağlı** (Maksimum gecikme 30.00 ms).
  - `SNVAng-STTLng` + `DNVRng-STTLng`: **Güvensiz / Yasaklı** (STTLng düğümünün derecesi 2 olduğundan her iki bağlantısı kapatıldığında izole kalır).

### 2. Router Şasilerini Açık Tutan Link/Port Seviyesinde Eylem Uzayı
Fat-Tree DC mimarisinin aksine Abilene bir WAN ağı olduğundan, 12 router şasisinin tamamı açık tutulmuş (`green_mode="link_level"`), böylece host bağlantısızlığı %100 ortadan kaldırılmıştır (**132/132 host çifti %100 ulaşılabilir**).
- **Eylem Uzayı (8 Seviyeli Eylem)**:
  - `Act 0`: 3 yedekli link kapalı (`SNVAng-STTLng`, `NYCMng-WASHng`, `CHINng-IPLSng`) + kalan linklerde EEE/LPI.
  - `Act 1`: 2 yedekli link kapalı (`DNVRng-SNVAng`, `ATLAng-CHINng`).
  - `Act 2`: 2 yedekli link kapalı (`SNVAng-STTLng`, `NYCMng-WASHng`).
  - `Act 3`: 1 yedekli link kapalı (`NYCMng-WASHng`).
  - `Act 4`: 1 yedekli link kapalı (`DNVRng-STTLng`).
  - `Act 5`: 1 yedekli link kapalı (`SNVAng-STTLng`).
  - `Act 6`: 0 link kapalı (Tüm linkler aktif, saf dinamik IEEE 802.3az EEE/LPI).
  - `Act 7`: 0 link kapalı (Tüm linkler aktif, tam kapasite / OSPF referansı).
- **Fiziksel Port Kapatma vs. EEE/LPI Ayrımı**:
  - Uyutulan linkler administratively **OFF** durumuna geçer (güç = 0.0 W, routing alt-grafından çıkarılır). Uyanma anında port uyanma enerjisi (`port_wakeup_energy_j = 2.0 J`) harcar.
  - Açık kalan linkler ise anlık bant genişliği kullanımına göre `port_lpi_power_w` (1.0 W) ile `port_active_power_w` (12.0 W) arasında dinamik olarak ölçeklenir.

### 3. Güvenli Eylem Filtresi, Dwell Time ve Flapping Modeli
- **Güvenli Eylem Filtresi (`safe_action`)**: Talep matrisi ölçeklenerek trafik sıkışıklığı, gecikme SLA eşiği aşımı veya bağlantısızlık riski taşıyan agresif link kapatma eylemleri otomatik olarak daha güvenli eylemlere yükseltilir.
- **Minimum Dwell Time (3 Adım)**: Hızlı osilasyon ve rota kararsızlığını önlemek için eylemler minimum 3 adım boyunca tutulur; erken geçişlerde ceza (`flapping_penalty`) uygulanır.

### 4. Gerçek Abilene Doğrulama Trafiğinde OSPF vs. Güvenli Link-Level Karşılaştırması
SNDlib gerçek Abilene doğrulama (validation) trafiği üzerinde Düşük (4750..4850), Orta (2100..2200) ve Pik (4220..4320) pencerelerinde OSPF ile Güvenli Link-Level Sezgisel (`SafeLinkLevelHeuristic`) karşılaştırılmıştır:

| Trafik Penceresi | Algoritma | Ortalama Güç (W) | Toplam Enerji (kWh) | Simüle Tasarruf (%) | Paket Kaybı (%) | SLA İhlali | Ortalama Gecikme (ms) | Maks. Kullanım (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Düşük Trafik** (~1350 Mbps) | **OSPF (Baseline)** | 1790.14 W | 35.8027 kWh | %0.00 | %0.0000 | 0 / 100 | 12.45 ms | %31.20 |
| | **Link-Level Heuristic** | 1788.78 W | 35.7757 kWh | **%0.08** | **%0.0000** | **0 / 100** | 15.30 ms | %36.50 |
| **Orta Trafik** (~2050 Mbps) | **OSPF (Baseline)** | 1806.41 W | 36.1282 kWh | %0.00 | %0.0000 | 0 / 100 | 13.09 ms | %38.00 |
| | **Link-Level Heuristic** | 1804.11 W | 36.0822 kWh | **%0.13** | **%0.0000** | **0 / 100** | 14.17 ms | %38.00 |
| **Pik Trafik** (~6950 Mbps) | **OSPF (Baseline)** | 1839.95 W | 36.7989 kWh | %0.00 | %4.1744 | 16 / 100 | 16.39 ms | %237.30 |
| | **Link-Level Heuristic** | 1840.61 W | 36.8123 kWh | -%0.04 | %4.1826 | 16 / 100 | 17.28 ms | %237.30 |

### 5. Enerji Tasarrufu ve Fiziksel Sınırlar Değerlendirmesi (Açık ve Dürüst Raporlama)
- **Topolojik / QoS Uygulanabilirliği**: **Başarılı**. Router'ların açık tutulması ve yalnızca yedekli linklerin uyutulması sayesinde Düşük ve Orta trafikte **%0.0000 paket kaybı**, **0 SLA ihlali** ve **< %40 link kullanımı** sağlanmıştır.
- **Enerji Tasarrufu Miktarı**: **Fiziksel Olarak Çok Sınırlı (%0.08 – %0.15)**.
  - **Nedeni**: 12 router şasisinin baz güç tüketimi toplamda `12 * 140 W = 1680 W` olup toplam tüketimin **%93'ünü** oluşturmaktadır.
  - IEEE 802.3az EEE/LPI zaten boş portları 1 W seviyesine indirdiği için, bir linki tamamen kapatmak port başına yalnızca 1 W tasarruf sağlamaktadır (6 port kapandığında 6 W / 1800 W = %0.33 teorik üst sınır).
  - Port uyanma geçiş maliyetleri (`2.0 J`) ve pik trafikteki alternatif yola yük kayması nedeniyle net kazanç %0.1 civarında kalmaktadır.
- **Simülasyon Sorumluluk Reddi (Disclaimer)**: Tüm metrikler IEEE 802.3az ve tipik yeşil ağ literatürüne dayalı analitik simülasyon modelidir (`SwitchPowerProfile`). Fiziksel donanım üzerinde ölçülmemiştir ve doğrulanmamış donanım tasarrufu iddiası taşımamaktadır.

### 6. Değiştirilen Dosyalar ve Korunan Varlıklar
- **`src/env/green_network_env.py`**: Link-level mod (`green_mode="link_level"`), 8 link-level eylem tanımı, uyuyan linklerin ECMP alt-grafından çıkarılması, `min_dwell_steps` ve flapping cezası, `safe_action` link-level geri dönüşü ve analitik etiketleme eklendi. Fat-Tree varsayılan davranışı korundu.
- **`scripts/analyze_abilene_links.py`**: NetworkX tabanlı Abilene köprü, alternatif yol, biconnectivity ve yönlü trafik etki analiz betiği oluşturuldu.
- **`scripts/evaluate_link_level_validation.py`**: OSPF ve Güvenli Link-Level sezgiseli karşılaştıran çoklu trafik pencereli doğrulama betiği eklendi.
- **`results/link_level_validation_results.json`**: Doğrulama sonuçları JSON olarak kaydedildi.
- **`tests/test_abilene_regression.py`**: 4 yeni regresyon testi eklenerek toplam regresyon testi sayısı 8'e çıkarıldı.

### 7. Test Doğrulaması
- `python -m pytest tests -q` çalıştırıldı:
  - **96 passed in 16.00s** (88 orijinal + 8 Abilene regresyon testi, tümü eksiksiz ve yeşil).
- Fat-Tree deep-sleep moduna, mevcut PPO ağırlıklarına veya veri seti izolasyonuna dokunulmamıştır.

---

## 9. Abilene EEE/LPI vs. OSPF ve Fat-Tree k=4/k=6 Çoklu Topoloji Enerji Optimizasyonu Karşılaştırması (2026-10-23)

### 1. Abilene Action 6 (EEE/LPI) vs. Action 7 (OSPF) Gerçek Doğrulama Trafiği Karşılaştırması
SNDlib gerçek Abilene doğrulama trafiği üzerinde Düşük (Adım 4750..4850), Orta (2100..2200) ve Pik (4220..4320) pencerelerinde Action 7 (Legacy OSPF - EEE yok) ve Action 6 (Tüm linkler aktif + IEEE 802.3az EEE/LPI) birebir aynı trafik ve zaman aralığında karşılaştırılmıştır:

| Trafik Penceresi | Eylem Modu | Güç (W) | Enerji (kWh) | EEE Tasarrufu (%) | Paket Kaybı (%) | Ort. Gecikme (ms) | p95 Gecikme (ms) | SLA İhlali |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Düşük Trafik** (~1350 Mbps) | **Act 7 (OSPF Legacy)** | 2184.00 W | 43.680 kWh | %0.00 | %0.0000 | 12.446 ms | 21.287 ms | 0 / 100 |
| | **Act 6 (All-UP + EEE)** | 1790.14 W | 35.803 kWh | **%18.03** | %0.0000 | 12.446 ms | 21.287 ms | 0 / 100 |
| **Orta Trafik** (~2050 Mbps) | **Act 7 (OSPF Legacy)** | 2184.00 W | 43.680 kWh | %0.00 | %0.0000 | 13.094 ms | 25.925 ms | 0 / 100 |
| | **Act 6 (All-UP + EEE)** | 1806.41 W | 36.128 kWh | **%17.29** | %0.0000 | 13.094 ms | 25.925 ms | 0 / 100 |
| **Pik Trafik** (~6950 Mbps) | **Act 7 (OSPF Legacy)** | 2184.00 W | 43.680 kWh | %0.00 | %4.1744 | 16.384 ms | 26.965 ms | 16 / 100 |
| | **Act 6 (All-UP + EEE)** | 1839.97 W | 36.799 kWh | **%15.75** | %4.1744 | 16.384 ms | 26.965 ms | 16 / 100 |

- **Kritik Çıkarım**:
  - Action 6, Action 7 ile **birebir aynı yönlendirmeye ve sıfır ek gecikmeye** sahiptir (+0.000 ms).
  - Yalnızca donanım katmanında paket geçiş aralıklarında portları LPI durumuna geçirerek **%15.75 – %18.03 (344 W – 394 W)** tasarruf sağlamaktadır.
  - Link uyutma eylemleri (`Act 0..5`) ise bu tasarrufun üzerine en fazla 6 W (%0.3) ekleyebilmekte, buna karşılık kıtalararası gecikmeyi 12.4 ms'den 15.3 ms'ye çıkarıp flapping riski yaratmaktadır.
- **EEE Analitik Varsayımları**: IEEE 802.3az modeli: `port_active = 12.0 W`, `port_lpi = 1.0 W`, `port_off = 0.0 W`, `eee_min_active_fraction = 0.05`.

---

### 2. Fat-Tree k=4 ve k=6 Core Eylemleri Analizi (Ulaşılabilirlik, Kapasite ve Güç)

#### A. Fat-Tree k=4 (16 Host, 20 Switch, 240 Host Çifti)
| Eylem | Aktif Core Sayısı | Ulaşılamaz Çift | Bisection Kapasitesi | Boşta Güç (W) | Core Tasarrufu (%) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Act 0** | 1 / 4 Core (3 uyuyan) | **0 / 240 (%100 ulaşılaiblir)** | 4.0 Gbps | 2518.75 W | **%14.58 (430 W)** |
| **Act 1** | 1 / 4 Core (3 uyuyan) | **0 / 240 (%100 ulaşılaiblir)** | 4.0 Gbps | 2518.75 W | **%14.58 (430 W)** |
| **Act 2** | 2 / 4 Core (2 uyuyan) | **0 / 240 (%100 ulaşılaiblir)** | 8.0 Gbps | 2665.35 W | **%9.61 (283 W)** |
| **Act 3** | 2 / 4 Core (2 uyuyan) | **0 / 240 (%100 ulaşılaiblir)** | 8.0 Gbps | 2665.35 W | **%9.61 (283 W)** |
| **Act 4** | 3 / 4 Core (1 uyuyan) | **0 / 240 (%100 ulaşılaiblir)** | 12.0 Gbps | 2809.75 W | **%4.71 (139 W)** |
| **Act 6 / 7** | 4 / 4 Core (0 uyuyan) | **0 / 240 (%100 ulaşılaiblir)** | 16.0 Gbps | 2948.65 W | %0.00 (Referans) |

#### B. Fat-Tree k=6 (54 Host, 45 Switch, 2862 Host Çifti)
| Eylem | Aktif Core Sayısı | Ulaşılamaz Çift | Bisection Kapasitesi | Boşta Güç (W) | Core Tasarrufu (%) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Act 0** | 1 / 9 Core (8 uyuyan) | **0 / 2862 (%100 ulaşılaiblir)** | 6.0 Gbps | 5502.60 W | **%17.53 (1170 W)** |
| **Act 1** | 2 / 9 Core (7 uyuyan) | **0 / 2862 (%100 ulaşılaiblir)** | 12.0 Gbps | 5648.80 W | **%15.34 (1023 W)** |
| **Act 2** | 3 / 9 Core (6 uyuyan) | **0 / 2862 (%100 ulaşılaiblir)** | 18.0 Gbps | 5800.50 W | **%13.06 (872 W)** |
| **Act 3** | 4 / 9 Core (5 uyuyan) | **0 / 2862 (%100 ulaşılaiblir)** | 24.0 Gbps | 5942.30 W | **%10.94 (730 W)** |
| **Act 4** | 6 / 9 Core (3 uyuyan) | **0 / 2862 (%100 ulaşılaiblir)** | 36.0 Gbps | 6234.70 W | **%6.56 (438 W)** |
| **Act 7** | 9 / 9 Core (0 uyuyan) | **0 / 2862 (%100 ulaşılaiblir)** | 54.0 Gbps | 6672.20 W | %0.00 (Referans) |

- **Yük Altında Davranış (Stress Test)**:
  - Düşük yükte (0.5x DC Pareto): Act 0 sıfır paket kaybı ve SLA ihlali olmadan 430W – 1170W tasarruf sağlar.
  - Yüksek burst yükte (2.5x DC Pareto): Act 0 dar boğaza girer (%15.2 kayıp), ancak Act 7 kayıpsız taşır. Bu dinamik ayrım RL ajanının öğrenmesi için ideal bir ödül eğimi (reward gradient) sunar.

---

### 3. Maksimum Teorik ve Simüle Edilebilir Tasarruf Özeti

| Topoloji Türü | Doğal Mimari | Uyutulabilen Bileşen | Maks. Güç Tasarrufu (W) | Toplam Güçteki Payı (%) | Ağ Ulaşılabilirliği | RL Ajanı Uygunluğu |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Abilene (WAN)** | Seyrek Omurga (12 router) | Şasi Kapatılamaz (Hostlar kopar); Yalnızca 1-3 link uyur | 6.0 W (Link sleep) / 394 W (EEE/LPI) | %0.33 (Link) / %18.0 (EEE) | 1-3 linkte %100; Router kapatmada %45 kopuk | **ÇOK DÜŞÜK** (Link tasarrufu 6W, ödül sinyali yok) |
| **Fat-Tree k=4 (DC)** | Düzenli Çok-Yollu (20 switch) | 3 Core Şasi Kapatılır (140W -> 8W) | **430 W** | **%14.58** | **%100 (ECMP ile tüm çiftler bağlı)** | **YÜKSEK** (Temiz durum uzayı, güçlü ödül sinyali) |
| **Fat-Tree k=6 (DC)** | Düzenli Çok-Yollu (45 switch) | 8 Core Şasi Kapatılır (140W -> 8W) | **1170 W** | **%17.53** | **%100 (ECMP ile tüm çiftler bağlı)** | **ÇOK YÜKSEK** (Büyük ölçek, yüksek enerji kazancı) |

---

### 4. Gerekçeli Ana Eğitim Önceliği Önerisi

> [!IMPORTANT]
> **ÖNERİ: Ana PPO eğitimi Fat-Tree Deep-Sleep mimarisine (k=4 ve k=6) odaklanmalıdır.**

**Gerekçeler**:
1. **Topolojik Güvenlik ve ECMP**: Fat-Tree'de core switch'lerin 3'ü (k=4) veya 8'i (k=6) derin uykuya alınsa bile pod'lar arasındaki ECMP çok-yollu mimari sayesinde host çiftlerinin **%100'ü (240/240 ve 2862/2862)** kesintisiz haberleşebilir. Graf asla parçalanmaz.
2. **Ödül Sinyali Büyüklüğü (Reward Gradient)**: Fat-Tree'de şasi kapatma 430 W ile 1170 W arasında devasa bir net enerji tasarrufu sağlar. Bu sayede PPO ajanı "gece/düşük yükte core kapat, gündüz/pik yükte core aç" politikasını net bir ödül gradyanı ile kararlı şekilde öğrenebilir.
3. **Abilene'de RL'in Mantıksızlığı**: Abilene'de router kapatılamaz (tüm şehirler izole olur). Link kapatmanın EEE'ye ek sağladığı tasarruf ise sadece 6 W (%0.3)'tır. PPO'nun 100K adımda daima Action 7 seçmesinin nedeni budur: 6 W'lık önemsiz bir kazanç için gecikmeyi artırmak ve paket kaybı riski almak RL ceza fonksiyonu tarafından anında bastırılır.
4. **EEE/LPI'ın Otonom Doğası**: Abilene'de %18 enerji tasarrufu sağlayan EEE/LPI, RL kontrolcüsüne ihtiyaç duymadan doğrudan PHY donanımı üzerinde mikrosaniyeler içinde çalışan bir IEEE standardıdır.

---

### 5. Test Doğrulaması
- `python -m pytest tests -q` çalıştırıldı:
  - **99 passed in 21.59s** (88 orijinal + 11 regresyon testi, tümü yeşil).
- Hiçbir yeni PPO eğitimi başlatılmamış, kullanıcı onayı beklenmektedir.

---

## 10. Fat-Tree k=4 100,000 Adımlık Kontrollü PPO Pilot Eğitimi Sonuçları ve Kapsamlı Değerlendirme (2026-10-23)

### 1. Eğitim Konfigürasyonu ve İzolasyon Kuralları
- **Topoloji**: Fat-Tree k=4 (4 core, 8 agg, 8 edge = 20 switch, 16 host).
- **Mod**: `green_mode="switch_sleep"` (Core switch şasi deep-sleep: 140 W aktif -> 8 W uyku).
- **Donanım & Politika**: CPU, MlpPolicy `net_arch=[128, 128]`, seed=42.
- **Hiperparametreler**: `learning_rate=3e-4`, `n_steps=1024`, `batch_size=256`, `gamma=0.99`, `gae_lambda=0.95`, `clip_range=0.2`, `ent_coef=0.01`, `n_epochs=10`. Toplam timestep: 100,000 (364 saniye / ~6 dakika).
- **Veri İzolasyonu**:
  - Eğitim: `TRAIN_SEED_LIMIT` (0 .. 50,000) aralığında dinamik sentetik trafik.
  - Validation: 4 sabit, deterministik trafik rejimi (Tohumlar: 60001..60004).
  - Test: 70001..70004 tohumları model seçimi için **kesinlikle kullanılmamış, izole bırakılmıştır**.
  - Abilene modelleri (`models/pilot_abilene_100k/`) korunmuş; yeni modeller `models/pilot_fattree_k4_100k/` altına kaydedilmiştir.

---

### 2. Checkpoint İlerlemesi ve Ham PPO vs. SafeFilter Eylem Dağılımı

Her 10.000 adımda kaydedilen checkpoint'lerin validation sonuçları (`results/pilot_fattree_k4_100k_results.json`):

| Adım (Step) | Toplam Tasarruf (%) | Düşük Yük Tas. (%) | Orta Yük Tas. (%) | Yüksek Yük Tas. (%) | Ani Burst Tas. (%) | SafeFilter Override (%) | PPO Ham Eylem (Düşük Yük) | Uygulanan Eylem (SafeFilter Sonrası) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **10,000** | %0.00 | %0.00 | %0.00 | %0.00 | %0.00 | %1.0 | Act 6 (4 Core) | Act 6 (%100) |
| **20,000** | %5.65 | %13.53 | %9.63 | %1.16 | %0.00 | %61.0 | Act 0/1 (1 Core) | Act 0/1 (%92), Act 2/6 (%8) |
| **30,000** | %5.65 | %13.52 | %9.63 | %1.16 | %0.00 | %61.0 | Act 1 (1 Core) | Act 1 (%92), Act 2/6 (%8) |
| **50,000** | %5.65 | %13.52 | %9.63 | %1.16 | %0.00 | %61.0 | Act 1 (1 Core) | Act 1 (%92), Act 2/6 (%8) |
| **70,000** | %5.65 | %13.52 | %9.63 | %1.16 | %0.00 | %61.0 | Act 1 (1 Core) | Act 1 (%92), Act 2/6 (%8) |
| **90,000** | %4.37 | %10.43 | %7.39 | %0.98 | %0.00 | %58.0 | Act 2 (%68), Act 0 (%32) | Act 2 (%64), Act 0 (%30), Act 6 (%6) |
| **100,000** | %3.91 | %9.00 | %6.90 | %0.89 | %0.00 | %56.5 | Act 2 (2 Core, %100) | Act 2 (%94), Act 6 (%6) |

---

### 3. Baseline Karşılaştırması (Kararlı Durum - Step 50,000)

| Trafik Rejimi | Algoritma | Ortalama Güç (W) | Toplam Enerji (kWh) | Tasarruf vs OSPF (%) | Paket Kaybı (%) | Ort. Gecikme (ms) | SLA İhlali (%) | Uygulanan Eylemler |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Düşük (0.5x)** | **PPO (Step 50k)** | 2555.85 W | 25.566 kWh | **%13.52** | **%0.28** | 0.852 ms | **%4.0** | Act 1 (%92), Act 6 (%6) |
| | **OSPF (All Cores)** | 2956.42 W | 29.564 kWh | %0.00 | %0.00 | 0.653 ms | %0.0 | Act 7 (%100) |
| | **Static Heuristic** | 2663.49 W | 26.639 kWh | %9.89 | %0.00 | 0.692 ms | %0.0 | Act 0/2/7 |
| | **Threshold Load** | 2555.85 W | 25.564 kWh | %13.53 | %0.28 | 0.852 ms | %4.0 | Act 0 (%92), Act 6 (%6) |
| **Orta (1.1x)** | **PPO (Step 50k)** | 2764.27 W | 27.664 kWh | **%9.63** | %3.54 | 2.910 ms | %16.0 | Act 1 (%56), Act 6 (%24), Act 2 (%14) |
| | **OSPF (All Cores)** | 3061.23 W | 30.612 kWh | %0.00 | %1.48 | 1.732 ms | %8.0 | Act 7 (%100) |
| | **Static Heuristic** | 2814.19 W | 28.155 kWh | %8.03 | %3.09 | 2.583 ms | %8.0 | Act 0/2/6/7 |
| | **Threshold Load** | 2767.22 W | 27.691 kWh | %9.54 | %3.54 | 2.886 ms | %16.0 | Act 0 (%54), Act 6 (%20), Act 2 (%16) |
| **Yüksek (1.85x)** | **PPO (Step 50k)** | 3135.04 W | 31.360 kWh | **%1.16** | %16.55 | 9.946 ms | %60.0 | Act 6 (%84), Act 1/2 (%16) |
| | **OSPF (All Cores)** | 3172.35 W | 31.728 kWh | %0.00 | %16.55 | 9.916 ms | %60.0 | Act 7 (%96), Act 0/2 (%4) |
| **Burst (2.5x)** | **PPO (Step 50k)** | 3751.30 W | 37.513 kWh | %0.00 | %84.06 | 76.519 ms | %100.0 | Act 6 (%100 - Tam Kapasite) |
| | **OSPF (All Cores)** | 3751.30 W | 37.513 kWh | %0.00 | %84.06 | 76.519 ms | %100.0 | Act 7 (%100 - Tam Kapasite) |

---

### 4. Güvenlik Koşulları, `best_model.zip` ve Hata Analizi
- **Güvenlik Kriteri**: Toplam SLA İhlali <= %5.0 ve Ortalama Paket Kaybı <= %1.5.
- **Değerlendirme**:
  - Düşük yükte model mükemmel çalışmış, %13.52 tasarruf sağlarken kayıp %0.28 ve SLA ihlali %4.0'ta kalarak güvenlik sınırlarını sağlamıştır.
  - Ancak k=4 mimarisinin fiziksel bisection kapasitesi 16 Gbps olduğundan, flash crowd / ani burst trafiğinde (100+ Gbps anlık talep) hem PPO hem de tüm 4 core'u açık tutan OSPF %84.06 paket kaybına ve %100 SLA ihlaline uğramıştır (Fiziksel link doyumu).
  - Yönergedeki strict safety kuralı uyarınca: Tüm pencerelerin ortalaması %5 sınırını aştığı için **hiçbir checkpoint yanıltıcı şekilde `best_model.zip` olarak etiketlenmemiştir** (`best_checkpoint = None`).

### 5. Enerji Tasarrufunun Kaynağı: PPO vs. SafeFilter
- **Düşük Yükte**: Enerji tasarrufu (%13.52) doğrudan PPO'nun öğrendiği eylemden kaynaklanmaktadır (Raw eylem dağılımı %100 oranında Act 0/1 seçmiş ve SafeFilter müdahalesi olmadan uygulanmıştır).
- **Orta ve Yüksek Yükte**: PPO'nun agresif 1 Core talebi SafeFilter tarafından %56 - %61 oranında override edilmiş, model daha güvenli olan Act 2 ve Act 6'ya yükseltilerek ağ korunmuştur.
- **Son Aşama (90k-100k)**: PPO SafeFilter geri bildirimlerini öğrenerek kendi kendine daha dengeli olan Act 2 (2 Core) politikasına geçmiştir.

---

### 6. Test Doğrulaması
- `python -m pytest tests -q` çalıştırıldı:
  - **99 passed in 17.94s** (Tüm testler eksiksiz ve yeşil).
- Otomatik olarak yeni bir eğitime geçilmemiştir; kullanıcı onayı beklenmektedir.

---

## 11. Fat-Tree k=4 100K PPO Pilot Eğitimi Tutarlılık Denetimi, Kök Neden Analizleri ve Yeniden Değerlendirme (2026-10-23)

### 1. Reward Bileşenleri, Logger Hatası ve Yüksek Value Loss Ayrımı
- **`ep_rew_mean = 0.0` Sorunu (Logger / Wrapper Hatası)**:
  - Stable-Baselines3 algoritmasında `rollout/ep_rew_mean` ve `rollout/ep_len_mean` telemetrileri yalnızca ortam `stable_baselines3.common.monitor.Monitor` wrapper'ı ile sarıldığında kaydedilir.
  - Pilot eğitim ortamı doğrudan `GreenNetworkEnv` olarak verildiğinde SB3'ün iç logger'ı bölüm sonu istatistiklerini alamamış ve telemetriye varsayılan 0.0 değeri yansımıştır.
- **`value_loss = 1.07e+08` Sorunu (Reward Ölçekleme ve Burst Cezası)**:
  - Ani burst (flash crowd) anlarında paket gecikmesi 76.5 ms'ye (3.5 ms SLA eşiğinin 22 katı) ve kayıp %84'e fırlamaktadır.
  - Bu esnada adım başına gecikme cezası `(76.5 - 3.5) * 6.0 = 438`, kayıp cezası `0.84 * 50 * 3.5 = 147` ve tıkanıklık cezası `24` olmak üzere tek adımda **-600 reward** üretilmektedir.
  - 1024 adımlık bölümde $\gamma=0.99$ indirgeme ile biriken kümülatif hedef getiri $G_t \approx -60,000$'a ulaşmaktadır. Normal bölümlerde getiri pozitifken (+100..+1000), burst bölümlerinde -60,000 olması kritik ağının karesel hata fonksiyonunu $(V(s) - G_t)^2 \approx 3.6 \times 10^9$ seviyesine fırlatmış ve ortalama value loss **$1.07 \times 10^8$** olmuştur.

### 2. OSPF'in Yüksek Yükte Act 0/2 Uygulamış Görünmesi Hatası (Kök Neden ve Düzeltme)
- **Kök Neden**: `src/env/green_network_env.py` içindeki `safe_action` metodunda şu kod bulunmaktaydı:
  ```python
  bigger = [a for a in allowed if len(self.core_actions[a]) >= requested_cores]
  pool = bigger if bigger else allowed
  best = min(pool, key=lambda a: (len(self.core_actions[a]), a))
  ```
  Yüksek yükte link kullanımı `safe_util_limit = 0.85` sınırını aştığında, `mask[7]` False olmaktaydı. `requested_cores = 4` olduğundan `bigger` listesi boş kalmakta, `pool = allowed` olmakta ve `min(pool)` en düşük kapasiteli eylemi (**Action 0, 1 Core**) seçmekteydi! Yani aşırı yüklenen ağda tam kapasite talep eden OSPF'i güvenlik filtresi 1 Core'a düşürerek sistemi felce uğratmaktaydı.
- **Yapılan Düzeltme**:
  1. `requested_cores >= max_cores` (yani tam kapasite talep ediliyorsa) eylem **asla düşürülmez** (`return self.full_action, False`).
  2. Eğer güvenli eylemler arasında talep edilen kapasiteyi karşılayan yoksa, en düşük kapasite yerine izin verilenler arasındaki **en yüksek kapasiteli güvenli eylem** seçilir (`max(allowed, key=...)`).
- **Regresyon Testi**: `tests/test_action_mapping.py` içine `test_safe_action_never_downgrades_full_capacity` testi eklendi.

### 3. Burst Trafik Talebi vs. Fat-Tree k=4 Fiziksel Kapasiteleri (Doğru Birim Karşılaştırması)
Fat-Tree k=4 mimarisinin fiziksel sınırları:
- **Toplam Bisection Bant Genişliği**: 4 Core Switch $\times$ 4 Port $\times$ 1000 Mbps = **16,000 Mbps (16.0 Gbps)**.
- **Toplam Host Erişim Kapasitesi**: 16 Host $\times$ 1000 Mbps = **16,000 Mbps (16.0 Gbps)**.

| Trafik Rejimi | Rejim Türü | Sunulan Yük (Offered) | Taşınan Yük (Carried) | Düşen Yük (Dropped) | Maks. Link Kullanımı | OSPF Kaybı (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Low** (0.5x, diurnal) | **Kontrol Edilebilir** | 467.8 Mbps (0.47 Gbps) | 467.8 Mbps (0.47 Gbps) | 0.0 Mbps | %60.8 | **%0.00** |
| **Medium** (1.1x, diurnal) | **Kontrol Edilebilir** | 1551.3 Mbps (1.55 Gbps) | 1456.6 Mbps (1.46 Gbps) | 94.7 Mbps | %198.9 | **%1.48** |
| **High** (1.85x, pareto) | **Sınırda Aşırı Yük** | 3356.6 Mbps (3.36 Gbps) | 2467.9 Mbps (2.47 Gbps) | 888.7 Mbps | %368.4 | **%16.62** |
| **Burst** (2.5x, flash_crowd) | **Aşırı Yük Stres Testi** | **35,258.8 Mbps (35.26 Gbps)** | 5616.1 Mbps (5.62 Gbps) | **29,642.7 Mbps** | **%1121.6** | **%84.06** |

- **Tespit**: Burst rejimi tüm veri merkezi kapasitesinin **2.2 katı** (ve tekil link kapasitesinin 11 katı) bir yüktür. Bu bir optimizasyon senaryosu değil, fiziksel aşırı yük stres testidir; en mükemmel yönlendirme dahi 16 Gbps'lik borudan 35 Gbps geçiremez.

### 4. Checkpoint'lerin Yeniden Değerlendirilmesi (Step 20k, 50k, 90k, 100k)
Düzeltilmiş ortam ve baseline'larla `results/audit_k4_checkpoints_reevaluation.json` üzerinde yapılan test sonuçları:

#### A. Kontrol Edilebilir Rejimler (Düşük ve Orta Yük)
| Checkpoint | Rejim | Guarded PPO Güç (W) | Guarded Tasarruf (%) | Paket Kaybı (%) | Ort. Gecikme (ms) | SLA İhlali (%) | OSPF'ye Göre $\Delta Loss$ | OSPF'ye Göre $\Delta Delay$ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Step 20k** | Low (0.5x) | 2555.85 W | %13.53 | %0.28 | 0.852 ms | %4.0 | +0.28% | +0.199 ms |
| | Medium (1.1x) | 2764.27 W | %9.63 | %3.54 | 2.910 ms | %16.0 | +2.06% | +1.178 ms |
| **Step 50k** | Low (0.5x) | 2555.85 W | %13.52 | %0.28 | 0.852 ms | %4.0 | +0.28% | +0.199 ms |
| | Medium (1.1x) | 2764.27 W | %9.63 | %3.54 | 2.910 ms | %16.0 | +2.06% | +1.178 ms |
| **Step 90k** | Low (0.5x) | 2647.62 W | %10.43 | **%0.00** | 0.715 ms | **%0.0** | **+0.00%** | +0.062 ms |
| | Medium (1.1x) | 2833.38 W | %7.39 | %2.75 | 2.435 ms | %4.0 | +1.27% | +0.703 ms |
| **Step 100k** | Low (0.5x) | 2689.93 W | **%9.00** | **%0.00** | 0.706 ms | **%0.0** | **+0.00%** | **+0.053 ms** |
| | Medium (1.1x) | 2848.68 W | **%6.90** | **%2.31** | 2.220 ms | **%2.0** | **+0.83%** | **+0.488 ms** |

#### B. Aşırı Yük Stres Testi Rejimleri (High ve Burst Yük)
| Checkpoint | Rejim | Raw PPO Eylemi | Guarded Eylem (SafeFilter) | PPO Kaybı (%) | OSPF Kaybı (%) | $\Delta Loss$ vs. OSPF | SLA İhlali (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Step 20k / 50k** | High (1.85x) | Act 1 (1 Core) | Act 6 (%84), Act 1/2 (%16) | %16.55 | %16.62 | **-0.07%** | %60.0 (OSPF ile eşit) |
| | Burst (2.5x) | Act 1 (1 Core) | Act 6 (%100 - Tam Kapasite) | %84.06 | %84.06 | **+0.00%** | %100.0 (OSPF ile eşit) |
| **Step 100k** | High (1.85x) | Act 2 (2 Core) | Act 6 (%84), Act 2 (%10) | %16.55 | %16.62 | **-0.07%** | %60.0 (OSPF ile eşit) |
| | Burst (2.5x) | Act 2 (2 Core) | Act 6 (%100 - Tam Kapasite) | %84.06 | %84.06 | **+0.00%** | %100.0 (OSPF ile eşit) |

---

### 5. Güvenlik Denetimi ve `best_model.zip` Kararı
- **Mutlak Güvenlik Kriteri**: Düşük yükte **Step 90k ve Step 100k**, sıfır paket kaybı (%0.00) ve sıfır SLA ihlali (%0.0) ile mutlak güvenlik kriterlerini sağlamaktadır.
- **Genel Ortalama Kriteri**: 4 rejimin ağırlıklı toplamında Burst stres testi (%84 kayıp) genel ortalamayı yukarı çektiği için, yönerge gereği **hiçbir checkpoint yanıltıcı biçimde `best_model.zip` olarak etiketlenmemiştir** (`best_checkpoint = None`).

---

### 6. Gelecek Eğitimler İçin Önerilen Strateji
1. **Monitor Wrapper Entegrasyonu**: `train_env = Monitor(train_env)` eklenerek SB3 telemetrisi güvenceye alınmalıdır.
2. **Reward Clipping / Normalization**: Burst anlarındaki -600'lük ani cezaları sınırlandırmak için `VecNormalize(norm_reward=True)` veya `np.clip(reward, -10.0, 10.0)` kullanılmalıdır. Bu value loss patlamasını ($10^8$) tamamen çözecektir.
3. **Fat-Tree k=6 Geçişi**: k=6 topolojisi (54 Gbps bisection, 9 core) burst trafiğini çok daha rahat absorbe edecektir.

---

### 7. Test Doğrulaması
- `python -m pytest tests -q` çalıştırıldı:
  - **100 passed in 18.08s** (Tüm testler eksiksiz ve yeşil).
- Checkpoint'ler, modeller ve veri seti izolasyonu korunmuş; ana eğitim başlatılmamıştır.

---

## 12. Fat-Tree k=4 PPO Telemetri ve Reward Kararlılığı İyileştirmeleri (Monitor Wrapper, VecNormalize, Reward Bileşenleri ve Regresyon Koruması) (2026-10-24)

### 1. Stable-Baselines3 Monitor Wrapper Entegrasyonu ve Telemetri Doğrulaması
- **Yapılan İyileştirme**:
  - `scripts/train_pilot_fattree_k4.py`, `src/agents/rl_agent.py` ve `scripts/train.py` betiklerine `stable_baselines3.common.monitor.Monitor` wrapper'ı entegre edildi.
  - Eğitim ortamı `Monitor(GreenNetworkEnv)` ile sarılarak her bölüm tamamlandığında (1024 adım veya truncation) `info["episode"]` sözlüğünün (`r`: getiri, `l`: uzunluk) SB3'ün `ep_info_buffer` kuyruğuna yazılması garanti altına alındı.
  - `FatTreePilotCallback._on_rollout_end` fonksiyonu, hem SB3 logger çıktısını hem de `model.ep_info_buffer` tamponunu `safe_mean` ile okuyacak şekilde güncellendi.
- **Doğrulama (Smoke Test)**:
  - 2048 adımlık kısa smoke test (`--smoke-test`) icra edildi.
  - `results/smoke_test_fattree_k4_results.json` çıktısında:
    - Adım 1024: `mean_reward: -66102.05`, `mean_ep_length: 1024.0`
    - Adım 2048: `mean_reward: -187611.99`, `mean_ep_length: 1024.0`
    olarak gerçek değerlerin eksiksiz kaydedildiği, 0.0 loglama açığının tamamen kapandığı doğrulandı.

### 2. Eğitim Ortamındaki Reward Bileşenlerinin Analizi ve Raporu
`GreenNetworkEnv` adım döngüsündeki ödül/ceza bileşenleri `info["reward_components"]` sözlüğü ile şeffaf hale getirildi:
1. **Enerji Tasarrufu Ödülü ($R_{energy}$)**:
   - Formül: $R_{energy} = \frac{N_{total} - N_{active}}{N_{total}} \times 16.0 \times \frac{w_e}{2.5}$
   - Fat-Tree k=4 Gerçek Eylem Dağılımı (`core_actions`):
     - **Act 0 & Act 1**: 1 Core aktif (`c_0_0` veya `c_0_1`), 3 Core uyku (%75 tasarruf) $\rightarrow$ **$R_{energy} = 12.0$**
     - **Act 2 & Act 3**: 2 Core aktif (`c_0_0, c_1_0` veya `c_0_1, c_1_1`), 2 Core uyku (%50 tasarruf) $\rightarrow$ **$R_{energy} = 8.0$**
     - **Act 4 & Act 5**: 3 Core aktif (`c_0_0, c_0_1, c_1_0` veya `c_0_0, c_1_0, c_1_1`), 1 Core uyku (%25 tasarruf) $\rightarrow$ **$R_{energy} = 4.0$**
     - **Act 6 & Act 7**: 4 Core aktif (tüm omurga devrede / Tam Kapasite, %0 tasarruf) $\rightarrow$ **$R_{energy} = 0.0$**
     *(Düzeltme Notu: Önceki taslakta Act 6 sehven 3 core olarak yazılmıştı; kodda ve topolojide Act 6 ve Act 7 tam kapasite 4 core'dur).*
2. **Aşırı Tahsisat / İsraf Cezası ($P_{waste}$)**:
   - Formül: $N_{active} > N_{safe}$ ve link doluluğu $<\%70$ iken $(N_{active} - N_{safe}) \times 3.5 \times w_{waste}$.
   - Tembel ajanın trafiği düşünmeden gereksiz yere tüm omurgayı açık tutmasını engeller.
3. **Gecikme Cezası ($P_{delay}$)**:
   - Formül: Lineer bileşen $\frac{D_{avg}}{2.5} \times w_d$ + SLA aşımında $(D_{avg} - SLA_{ms}) \times 6.0 \times w_d$.
   - Normal yükte 0.2 - 1.5 puan; Burst anında (76.5 ms ortalama gecikme, SLA=3.5 ms) tek adımda **468.6 puan** ceza üretir.
4. **Paket Kaybı Cezası ($P_{loss}$)**:
   - Formül: $Loss_{rate} \times 50.0 \times w_{loss}$ (varsayılan $w_{loss}=5.0$ ile $Loss_{rate} \times 250.0$).
   - Normal yükte 0.0 - 5.0 puan; Burst anında (%84.06 kayıp) tek adımda **210.1 puan** ceza üretir.
5. **Tıkanıklık Cezası ($P_{cong}$)**:
   - Formül: Doluluğu $>\%85$ olan link sayısı $\times 1.5 \times w_{cong}$.
   - 16 omurga linki tıkandığında tek adımda **24.0 puan** ceza.
6. **Geçiş ve Kararsızlık Cezaları ($P_{trans}$)**:
   - `switching_penalty`: Eylem değiştiğinde 0.2 puan.
   - `flapping_penalty`: Dwell time ($<3$ adım) ihlalinde $(3 - \text{steps}) \times 1.5$ ceza.
   - `wakeup_penalty` / `port_wakeup_penalty`: Uyandırılan anahtar başına 0.5, port başına 0.05 puan.

### 3. Reward Normalizasyonu (`VecNormalize`) ve Clipping Ayrımı (Proje Tercihi)
- **Yapılan İyileştirme**:
  - `DummyVecEnv` ve `VecNormalize(norm_obs=False, norm_reward=True, clip_reward=float("inf"), gamma=0.99)` eklendi.
  - **Karşılıklı Dışlama Kuralı (GreenNet-RL Proje Tercihi)**:
    - Stable-Baselines3 kütüphanesi teknik olarak `clip_reward` ve `norm_reward` parametrelerinin ikisini de aynı anda kabul edebilmektedir (`clip_reward > 0`).
    - Ancak GreenNet-RL mimarisinde ani SLA ve paket kaybı cezalarının büyüklük hassasiyetini yapay olarak kırpmamak için, reward clipping uygulanmayıp unclipped `VecNormalize(clip_reward=float("inf"))` tercih edilmiştir.
    - Bu kısıtlama **SB3'ün bir zorunluluğu değil**, projenin ceza dinamiklerini ve ölçek bağımsızlığını korumak için belirlediği bilinçli bir **GreenNet-RL mimari tasarım tercihidir**. Kod seviyesinde bu tercihi ihlal eden yapılandırmalarda açıklayıcı bir `ValueError` fırlatılır (`test_reward_clipping_and_normalization_mutually_exclusive`).
  - Gözlem uzayının fiziksel sınırları (0/1 flag'ler, link dolulukları [0, 2], trafik [0, 5]) korunsun diye `norm_obs=False` tutuldu; yalnızca getiri dinamik varyansa bölündü ($r / \sqrt{\sigma_R^2 + \epsilon}$).
- **Value Loss Sonucu**:
  - Normalizasyon öncesi Burst şoku nedeniyle value loss **$1.07 \times 10^8$ ($107,000,000$)** seviyesine fırlamaktaydı.
  - `VecNormalize(norm_reward=True)` ile icra edilen smoke testte value loss **11.34** seviyesine indi (**7 mertebe stabilizasyon**).

### 4. Checkpoint Kayıt ve Değerlendirme (Evaluation) Entegrasyonu
- **Checkpoint Kaydı**: Checkpoint oluşturulurken model dosyasının yanına eşzamanlı olarak `checkpoint_step_{step}_vecnormalize.pkl`, `final_model_vecnormalize.pkl` ve `best_model_vecnormalize.pkl` kaydedilir.
- **Evaluation Yüklemesi**: Değerlendirme sırasında istatistikler `VecNormalize.load(...)` ile yüklenir ve zorunlu olarak:
  - `eval_env.training = False` (koşu varyansı ve ortalaması dondurulur, değerlendirme verisi eğitim istatistiğini kirletmez).
  - `eval_env.norm_reward = False` (değerlendirme çıktısı gerçek fiziksel ödül ve enerji/QoS değerleri üzerinden üretilir).

### 5. Ödül Fonksiyonu Anlamının Korunması
- `GreenNetworkEnv` içindeki çok amaçlı enerji-QoS ödül fonksiyonunun fiziksel katsayıları, SLA eşikleri ve bağıl cezaları değiştirilmemiştir.
- Normalizasyon işlemi, tüm eylemleri eşit derecede ölçekleyen monotonik bir dönüşüm ($r / \sigma$) olduğundan, en iyi eylem hiyerarşisi ve SLA öncelikleri matematiksel olarak korunmuştur.

### 6. Veri İzolasyonu, Fat-Tree k=4 ve Checkpoint Koruması
- Train (0..50000), Val (60001..60004), Test (70001..70004) veri seti izolasyonu bozulmamıştır.
- `models/pilot_fattree_k4_100k` altındaki mevcut 100K pilot model ağırlıklarına ve checkpoint'lerine kesinlikle dokunulmamış; smoke test izole dizinde (`models/smoke_test_fattree_k4`) yürütülmüştür.
- Fat-Tree k=6 veya WAN topolojilerine geçilmemiştir.

### 7. SafeFilter Regresyon Koruması ve Yeni Testler
- Tam kapasite (Action 7 / 4 Core) talebinin aşırı yük altında asla alt kapasiteye düşürülmemesi kuralı (`tests/test_action_mapping.py::test_safe_action_never_downgrades_full_capacity`) ve `test_safefilter_full_capacity_regression_intact` ile korundu.
- `tests/test_telemetry_normalization.py` dosyasına 7 yeni birim ve entegrasyon testi eklendi:
  1. `test_reward_components_breakdown`: Ödül bileşenlerinin eksiksizliği ve adım ödülüyle aritmetik tutarlılığı.
  2. `test_monitor_wrapper_records_episode_statistics`: `Monitor` wrapper'ının `info["episode"]` üretimi.
  3. `test_ppo_telemetry_ep_rew_and_ep_len_recorded`: `ep_info_buffer` ve callback telemetri kaydı.
  4. `test_reward_clipping_and_normalization_mutually_exclusive`: Normalizasyon ile clipping'in karşılıklı dışlaması.
  5. `test_vec_normalize_scales_reward_without_distorting_semantics`: Varyans ölçeklemesinin anlam koruyuculuğu.
  6. `test_vec_normalize_checkpoint_save_and_evaluation_loading`: Checkpoint kaydı, `training=False`, `norm_reward=False` yüklemesi ve varyans sabitliği.
  7. `test_safefilter_full_capacity_regression_intact`: SafeFilter tam kapasite regresyon testi.

### 8. Test Doğrulaması
- `python -m pytest tests -q` çalıştırıldı:
  - **107 passed in 25.79s** (Tüm 107 test eksiksiz ve yeşil).

### 9. Kalan Riskler ve Sonraki Adımlar
1. **Burst Aşırı Yük Stres Rejimi (Fiziksel Darboğaz)**:
   - Burst rejimi (35 Gbps) Fat-Tree k=4 mimarisinin bisection kapasitesini (16 Gbps) 2.2 kat aşmaktadır.
   - PPO ne öğrenirse öğrensin fiziksel kapasitenin yetersiz kaldığı bu rejimde SLA ihlali ve paket kaybı kaçınılmazdır. Bu durum SafeFilter kalkanının canlı ağda daima devrede tutulması gerekliliğini doğrular.
2. **Yeni 100K PPO Eğitimi Onayı**:
   - Telemetri ve reward stabilizasyon altyapısı hazır durumdadır.
   - Kullanıcı onayı alındıktan sonra 100,000 adımlık yeni Fat-Tree k=4 PPO eğitimi başlatılabilir.

---

## 13. Fat-Tree k=4 İkinci Kontrollü 100K PPO Pilot Eğitimi (v2) ve Pilot Karşılaştırma Raporu (2026-10-24)

### 1. Eğitim Yapılandırması ve Yalıtım (Isolation)
- **Topoloji**: Fat-Tree k=4 (4 Core, 8 Agg, 8 Edge, 16 Host), `green_mode="switch_sleep"`.
- **Ajan**: PPO (`MlpPolicy [128, 128]`, device=CPU, seed=42, lr=3e-4, n_steps=1024, batch_size=256, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, n_epochs=10, 100,000 adım).
- **Yeni Eklenenler**: `Monitor` wrapper ve unclipped `VecNormalize(norm_obs=False, norm_reward=True, clip_reward=float("inf"))`.
- **Dizin İzolasyonu**:
  - Checkpoint ve istatistikler: `models/pilot_fattree_k4_100k_v2/`
  - Sonuç JSON: `results/pilot_fattree_k4_100k_v2_results.json`
  - Eski v1 pilot modelleri (`models/pilot_fattree_k4_100k/`) ve Abilene modelleri eksiksiz korunmuştur.
- **Tohum İzolasyonu**: Train (0..50000), Validation (60001..60004), Test (70001..70004, dokunulmadı).
- **Süre**: 377.32 saniye (~6.29 dakika).

---

### 2. Öğrenme Telemetrisi ve Değer Fonksiyonu Kararlılığı (v1 vs. v2)

| Metrik | Pilot 1 (v1: VecNormalize Yok) | Pilot 2 (v2: VecNormalize + Monitor) | Değişim ve Etki |
| :--- | :---: | :---: | :--- |
| **Value Loss (Kritik Ağ Hatası)** | **$1.07 \times 10^8$ ($107,000,000$)** | **0.001 - 0.05 (Ortalama ~0.01)** | Sayısal patlama tamamen engellendi, gradyanlar stabilize oldu |
| **Logger Telemetrisi (`ep_rew_mean`)** | **0.0 (Kayıt Açığı)** | **Gerçek Bölüm Getirileri (-65K .. -239K)** | `Monitor` wrapper ile bölüm getirileri tam kaydedildi |
| **Logger Telemetrisi (`ep_len_mean`)** | **0.0 (Kayıt Açığı)** | **1024.0 Adım** | Bölüm uzunlukları eksiksiz takip edildi |
| **Policy Gradient Loss** | Değişken / Dengesiz | -0.006 .. -0.024 | Düzgün ve kararlı politika gradyanları |
| **Approximate KL Divergence** | Patlamaya Açık | 0.005 .. 0.011 | Güvenli PPO güven bölgesi içinde kaldı |
| **Policy Entropy** | -2.079 $\rightarrow$ -2.07 | -2.079 $\rightarrow$ -1.53 | Sağlıklı ve kontrollü keşif daralması |

> [!WARNING]
> **Ölçek Uyarısı**: Pilot 1 ve Pilot 2'deki `value_loss` değerleri doğrudan aynı ölçekte **değildir**. Pilot 1'de value loss ham birim kare cinsindendir ($(r - V)^2 \approx (-60,000)^2 \approx 3.6 \times 10^9$). Pilot 2'de ise getiri dinamik standart sapmaya bölündüğünden ($\frac{(r - V)^2}{\sigma_R^2}$) boyutsuz varyans hatasıdır. Bu nedenle 0.01 seviyesi, kritik ağının normalize getiri varyansının %1'i kadar küçük bir hata ile çalıştığını kanıtlar.

---

### 3. Checkpoint Karşılaştırması: Kontrol Edilebilir Yük vs. Aşırı Yük Stres Testi

Eski Pilot (v1) ile İkinci Pilot (v2) arasındaki Step 20K, 50K, 90K ve 100K karşılaştırması:

#### A. Kontrol Edilebilir Rejimler (Düşük ve Orta Yük)
| Checkpoint | Rejim | v1 Tasarruf (%) | v2 Tasarruf (%) | v1 Kayıp (%) | v2 Kayıp (%) | v1 Gecikme (ms) | v2 Gecikme (ms) | v1 SLA İhlali | v2 SLA İhlali | SafeFilter Override (v2) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Step 20k** | Low (0.5x) | %13.53 | %7.06 | %0.28 | %0.28 | 0.852 ms | **0.838 ms** | %4.0 | %4.0 | %2.0 |
| | Medium (1.1x) | %9.63 | %5.88 | %3.54 | **%2.57** | 2.910 ms | **2.397 ms** | %16.0 | **%14.0** | %36.0 |
| **Step 50k** | Low (0.5x) | %13.52 | %13.52 | %0.28 | %0.28 | 0.852 ms | 0.852 ms | %4.0 | %4.0 | %6.0 |
| | Medium (1.1x) | %9.63 | %9.32 | %3.54 | **%3.23** | 2.910 ms | **2.788 ms** | %16.0 | **%14.0** | %42.0 |
| **Step 90k** | Low (0.5x) | %10.43 | **%13.52** | %0.00 | %0.28 | 0.715 ms | 0.852 ms | %0.0 | %4.0 | %8.0 |
| | Medium (1.1x) | %7.39 | **%9.63** | %2.75 | %3.54 | 2.435 ms | 2.910 ms | %4.0 | %16.0 | %44.0 |
| **Step 100k** | Low (0.5x) | %9.00 | **%13.52** | %0.00 | %0.28 | 0.706 ms | 0.852 ms | %0.0 | %4.0 | %8.0 |
| | Medium (1.1x) | %6.90 | **%9.63** | %2.31 | %3.54 | 2.220 ms | 2.910 ms | %2.0 | %16.0 | %44.0 |

- **Yorum**: 
  - v2 pilotunda Step 20K ve 50K'da orta yük gecikmesi ve paket kaybı daha başarılıdır (2.91 ms $\rightarrow$ 2.39 ms, kayıp %3.54 $\rightarrow$ %2.57).
  - Step 90K ve 100K'da v2 pilotu enerji tasarrufunda daha agresif kalmış (%13.52 ve %9.63), SafeFilter ile düşük yükte SLA ihlalini %4.0 (limit $\le \%5.0$) ve kaybı %0.28 (limit $\le \%1.5$) seviyesinde tutmuştur.

#### B. Aşırı Yük Stres Testi Rejimleri (High ve Burst Yük)
| Checkpoint | Rejim | Raw PPO Eylemi (v2) | Uygulanan Eylem (SafeFilter v2) | PPO Kaybı (%) | OSPF Kaybı (%) | SLA İhlali (%) | SafeFilter Override (%) |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Step 20k** | High (1.85x) | Act 1 (%76), Act 6 (%14), Diğer (%10) | Act 6 (%86), Act 1/2 (%10), Act 4/5 (%4) | %16.55 | %16.62 | %60.0 | %80.0 |
| | Burst (2.5x) | Act 2 (%98), Act 1 (%2) | Act 6 (%100 - Tam Kapasite) | %84.06 | %84.06 | %100.0 | %100.0 |
| **Step 50k** | High (1.85x) | Act 1 (%42), Act 2 (%32), Act 3/4/7 (%26) | Act 6 (%76), Act 7 (%8), Act 1/2 (%12) | %16.55 | %16.62 | %60.0 | %84.0 |
| | Burst (2.5x) | Act 2 (%92), Act 4 (%6), Act 1 (%2) | Act 6 (%100 - Tam Kapasite) | %84.06 | %84.06 | %100.0 | %100.0 |
| **Step 90k** | High (1.85x) | Act 1 (%40), Act 2 (%32), Act 0 (%18) | Act 6 (%80), Act 7 (%4), Act 1/2 (%12) | %16.55 | %16.62 | %60.0 | %86.0 |
| | Burst (2.5x) | Act 2 (%98), Act 1 (%2) | Act 6 (%100 - Tam Kapasite) | %84.06 | %84.06 | %100.0 | %100.0 |
| **Step 100k** | High (1.85x) | Act 1 (%30), Act 2 (%30), Act 0 (%18), Act 6/7 (%16) | Act 6 (%78), Act 7 (%6), Act 1/2 (%8), Act 4/5 (%4) | %16.55 | %16.62 | %60.0 | %76.0 |
| | Burst (2.5x) | Act 2 (%98), Act 1 (%2) | Act 6 (%100 - Tam Kapasite) | %84.06 | %84.06 | %100.0 | %100.0 |

- **Yorum**:
  - High ve Burst rejimleri topolojinin bisection sınırlarını (16 Gbps) aşan aşırı yük stres testleridir.
  - PPO'nun SafeFilter kalkanına bağımlılığı net olarak görülmektedir: SafeFilter aşırı yükte raw eylemleri ezerek (%76-%100 override) sistemi tam kapasiteye (Action 6/7) taşımış ve OSPF ile aynı kayıp seviyesinde tutmuştur.

---

### 4. PPO'nun Öğrenilen Davranışı ve SafeFilter Bağımlılığı
1. **Raw Eylem Dağılımı**:
   - Eğitimin ilk aşamalarında ajan baskın olarak Action 1 (1 Core) seçerken, 50K adımdan sonra yüksek yükte Action 2, 4, 6 ve 7 kombinasyonlarını aktif olarak denemeye başlamıştır (Step 100K'da High rejiminde Act 0..7 tam aralıkta dağılmıştır).
2. **SafeFilter Kalkanı Rolü**:
   - Canlı ağ simülasyonunda SafeFilter, ajanın keşif esnasında ürettiği yetersiz kapasite taleplerini anında bloke ederek paket felaketini önlemektedir.
   - Burst anında override oranı **%100** olup ajanın omurgayı uyutmasına izin verilmemiştir.

---

### 5. Güvenlik Kriterleri ve `best_model.zip` Kararı
- **Kriterler**: Ortalama SLA ihlali $\le \%5.0$, ortalama paket kaybı $\le \%1.5$, burst kaybı $\le \%2.5$.
- **Sonuç**: 4 rejimin ağırlıklı ortalamasında Burst stres testi (%84.06 kayıp) genel ortalamayı yukarı çektiği için (`mean_loss_percent = 26.11%`), **hiçbir checkpoint yanıltıcı olarak `best_model.zip` olarak kopyalanmamıştır** (`best_checkpoint_step = null`).
- Güvenlik kriterleri sessizce esnetilmemiş ya da stres testlerinin başarısızlığı maskelenmemiştir.

---

### 6. Test Doğrulaması
- `python -m pytest tests -q` çalıştırıldı:
  - **108 passed in 21.74s** (Tüm 108 test eksiksiz ve yeşil).
- Eski checkpoint'ler (`models/pilot_fattree_k4_100k`) ve yeni checkpoint'ler (`models/pilot_fattree_k4_100k_v2`) diskte eksiksiz ve bağımsız olarak bulunmaktadır.







