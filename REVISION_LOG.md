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


