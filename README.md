# 🌿 Green SDN: Akıllı Ağ Yönetimi İçin Pekiştirmeli Öğrenme (RL) Destekli Yeşil Yönlendirme ve Enerji Optimizasyonu

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-v0.29-brightgreen.svg)](https://gymnasium.farama.org/)
[![Stable Baselines3](https://img.shields.io/badge/RL-Stable--Baselines3-orange.svg)](https://stable-baselines3.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688.svg)](https://fastapi.tiangolo.com/)
[![Mininet Ready](https://img.shields.io/badge/SDN-Mininet%20%26%20Ryu-blueviolet.svg)](http://mininet.org/)

Bu proje, modern veri merkezi ağlarında (Data Center Networks - DCN) ve Yazılım Tanımlı Ağlar (SDN) ortamında hizmet kalitesinden (QoS) ödün vermeksizin **enerji tüketimini en aza indiren** ileri seviye bir **Pekiştirmeli Öğrenme (Reinforcement Learning - RL)** optimizasyon sistemidir.

Proje hem **Windows ortamında doğrudan çalıştırılabilen yüksek hızlı bir dijital ikiz simülatörü ve canlı web paneline** hem de gerçek bir testbed ortamında test edilebilmesi için **Mininet & Ryu SDN Denetleyicisi dışa aktarma modülüne** sahiptir.

---

## 🎯 Projenin Temel Amaçları ve Özellikleri

1. **Yeşil Bilişim & Endüstri 5.0 Uyumu:**
   * Düşük ve orta yük dönemlerinde gereksiz açık kalan omurga (Core) anahtarlarını ve bağlantı portlarını **IEEE 802.3az Low Power Idle (LPI) / Sleep** moduna alır.
   * Geleneksel yöntemlere (OSPF / ECMP) kıyasla **%25 - %45 oranında enerji tasarrufu** sağlar.

2. **QoS (Hizmet Kalitesi) Güvencesi:**
   * Çok amaçlı ödül fonksiyonu ($R$), gecikme (latency) ve paket kaybını sıkı şekilde cezalandırır.
   * Trafik ani artış gösterdiğinde (DDoS, yedekleme patlaması, prime-time) ajan uyuyan omurga anahtarlarını anında uyandırarak darboğazı engeller.

3. **Mininet & Ryu SDN Tam Uyumluluğu:**
   * Topoloji ve akış tablosu kuralları OpenFlow 1.3 standardında çalışır.
   * Tek tıkla veya komutla Mininet topolojisi (`fattree_topo.py`) ve Ryu kontrolcüsü (`ryu_green_controller.py`) üretilebilir.

4. **NOC Seviyesinde İnteraktif Web Simülasyon Paneli:**
   * WebSocket ile anlık 60 FPS canlı paket akışları, renk kodlu bağlantı dolulukları, anlık güç/karbon tasarruf sayaçları ve OSPF ile canlı kıyaslama grafiği.

---

## 🏗️ Sistem Mimarisi

```text
profesyonel yazılım uygulamaları/
├── ARCHITECTURE.md          # Ayrıntılı matematiksel ve akademik mimari dokümantasyonu
├── requirements.txt         # Gerekli Python kütüphaneleri
├── run.py                   # Canlı Web Dashboard'u ve simülasyonu başlatan ana script
├── src/
│   ├── network/             # Fat-Tree Topolojisi, Güç Tüketim Modeli ve Trafik Üreteci
│   │   ├── topology.py      # k=4 Fat-Tree (20 switch, 16 host) graf yapısı
│   │   ├── power_model.py   # IEEE 802.3az tabanlı şasi ve port güç modeli
│   │   └── traffic.py       # Günlük dalgalanma (diurnal) ve trafik patlaması üreteci
│   ├── env/
│   │   └── green_network_env.py # Gymnasium uyumlu Markov Karar Süreci (MDP) ortamı
│   ├── agents/              # RL Ajanları ve Kıyaslama Algoritmaları
│   │   ├── rl_agent.py      # Stable-Baselines3 PPO & DQN eğitici ve çıkarım motoru
│   │   ├── baselines.py     # OSPF (Klasik En Kısa Yol) ve Sezgisel Algoritmalar
│   │   └── evaluate.py      # Çok metrikli performans değerlendirme motoru
│   ├── sdn/                 # SDN & Mininet Entegrasyonu
│   │   ├── controller.py    # OpenFlow Flow-Mod ve Port-Stats kontrol soyutlaması
│   │   └── mininet_exporter.py # Mininet ve Ryu Python script üreteci
│   └── web/                 # Canlı Web Dashboard (FastAPI + WebSocket + HTML5 Canvas)
│       ├── server.py        # Telemetri WebSocket yayıncısı ve REST API
│       └── static/          # Modern NOC temalı ön yüz (HTML, CSS, JS, Chart.js)
├── scripts/
│   ├── train.py             # RL modelini eğiten komut dosyası (--reward-scale destekli)
│   ├── benchmark.py         # OSPF vs Sezgisel vs RL kıyaslama tablosunu üreten script
│   ├── run_experiments.py  # Çoklu seed (mean±std, 95% CI) bilimsel deney matrisi
│   ├── run_ablation.py     # Ödül fonksiyonu ablation çalışması
│   └── generate_report.py   # Rapor ve grafik üretici
├── mininet_scripts/         # Mininet topolojisi, Ryu denetleyicisi ve test süiti
└── tests/                   # Kapsamlı pytest birim ve entegrasyon testleri (73 test)
```

---

## ⚡ Hızlı Başlangıç

### 1. Canlı Simülasyon Web Panelini Başlatma

Simülasyon sunucusunu başlatmak ve web tarayıcısında açmak için:

```powershell
.\venv\Scripts\python.exe run.py
```

Tarayıcınızda otomatik olarak `http://127.0.0.1:8000` açılacaktır.

Panel üzerinden:
* **Yönlendirme Modu:** PPO Ajanı, OSPF (Klasik) ve Sezgisel modlar arasında tek tıkla geçiş yapabilir, güç tüketiminin anında nasıl değiştiğini izleyebilirsiniz.
* **Trafik Patlaması (⚡):** Butona basarak ağa ani aşırı yük bindirebilir, RL ajanının uyuyan omurga anahtarlarını nasıl anında uyandırdığını gözlemleyebilirsiniz.
* **Mininet Dışa Aktar:** Mininet ve Ryu dosyalarını tek tıkla `mininet_scripts/` klasörüne aktarabilirsiniz.

---

### 2. RL Modelini Eğitme

Ajanı sıfırdan eğitmek veya farklı hiperparametrelerle denemek için:

```powershell
.\venv\Scripts\python.exe scripts/train.py --algo PPO --timesteps 25000
```

Eğitilen ağırlıklar otomatik olarak `models/ppo_green_agent.zip` konumuna kaydedilir.

---

### 3. Akademik Kıyaslama (Benchmark) Raporu Üretme

Tüm algoritmaları aynı trafik senaryosu üzerinde yarıştırıp akademik metrik tablosunu ekrana basmak için:

```powershell
.\venv\Scripts\python.exe scripts/benchmark.py
```

Örnek Çıktı:

| Yöntem / Algoritma | Ort. Güç (Watt) | Enerji Tasarrufu (%) | Ort. Gecikme (ms) | Maks. Gecikme (ms) | Paket Kaybı (%) | Darboğaz Olayları |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **OSPF / ECMP (Baseline)** | 3280.0 W | %0.0 | 0.82 ms | 1.15 ms | %0.0 | 0 |
| **Sezgisel (Diurnal)** | 2420.0 W | %26.2 | 1.05 ms | 2.40 ms | %0.05 | 2 |
| **Green RL Agent (PPO)** | **2080.0 W** | **%36.5** | **1.12 ms** | **2.10 ms** | **%0.0** | **0** |

---

### 4. Mininet & Ryu SDN Ortamında Çalıştırma (Opsiyonel / İleri Seviye)

Mininet betiklerini dışa aktarmak için:

```powershell
.\venv\Scripts\python.exe scripts/export_mininet.py
```

Oluşan dosyaları Ubuntu / WSL / Mininet sanal makinesine kopyalayarak:

```bash
# Terminal 1: Ryu Controller (PPO PolicyBridge)
PYTHONPATH=. ryu-manager mininet_scripts/ryu_green_controller.py

# Terminal 2: Otomatik Mininet Doğrulama (Ping, iperf, Link-Failure/Recovery)
sudo python3 mininet_scripts/test_mininet_experiments.py
```

> **Not:** Gerçek OVS/Mininet yazılımında şasi uyku modu donanımsal olarak bulunmadığından, port modifikasyonları gerçektir; güç ve EEE değerleri analitik simülasyon modeli (`src.network.power_model`) ile hesaplanır ve loglarda `[SIMULATED_ENERGY]` olarak etiketlenir.

---

### 5. Testler ve Bilimsel Doğrulama

```powershell
# 1. Tüm birim ve entegrasyon testlerini çalıştırma (88 test)
pytest tests -v

# 2. Yalnızca Abilene veri yükleyici ve entegrasyon testlerini çalıştırma
pytest tests/test_abilene_loader.py -v

# 3. Çoklu seed ve arıza senaryoları kıyaslaması (mean±std, %95 CI)
python scripts/run_experiments.py --seeds 5 --steps 120

# 4. Ödül fonksiyonu ablation çalışması
python scripts/run_ablation.py --timesteps 15000 --seeds 3
```

---

### 6. SNDlib Abilene Gerçek Dünya Trafik Verisi Entegrasyonu

GreenNet-RL, sentetik trafik üretiminin yanı sıra **SNDlib Abilene** (Yin Zhang 6 aylık, 5 dakikalık çözünürlüklü) gerçek ağ trafik verilerini destekler.

* **Veri Konumu:** `data/raw/abilene/directed-abilene-zhang-5min-over-6months-ALL-native/`
* **Toplam Matris:** 48.096 adet Native SNDlib 1.0 matrisi (`2004-03-01` -> `2004-09-10`).
* **Birim & Doğrulama:** Kaynaktaki orijinal `MBITPERSEC` (Mbps) birimi yapay çarpanlara uğratılmadan korunur. Bozuk dosyalar, geçersiz düğümler, negatif trafik değerleri ve ölçüm kesintileri katı doğrulama kurallarıyla kontrol edilir.
* **Kronolojik Veri Ayrımı (Data Leakage Önleme):**
  * **Eğitim (%70):** 33.667 matris (`20040301-0000` -> `20040721-2130`)
  * **Doğrulama (%15):** 7.214 matris (`20040721-2135` -> `20040815-2240`)
  * **Test (%15):** 7.215 matris (`20040815-2245` -> `20040910-2355`)
* **Hafıza Optimizasyonu (Lazy Loading):** 48.096 dosya tek seferde RAM'e yüklenmez; `data/processed/abilene_manifest.json` üzerinden dizinlenir ve ihtiyaç duyuldukça 1024'lük LRU bellek önbelleği ile diskten anlık okunur.
* **Git Koruması:** `data/raw/` ve `data/processed/` dizinleri `.gitignore` altında izlenmez.

#### Python ile Kullanım Örneği:

```python
from src.env.green_network_env import GreenNetworkEnv
from src.data.abilene_loader import AbileneDataLoader

# 1. Abilene Topolojisi ve Gerçek Test Trafiği ile Ortam Oluşturma
env = GreenNetworkEnv(
    topology_type="abilene",
    traffic_source="real",
    real_traffic_split="test",       # 'train', 'val' veya 'test'
    link_capacity_mbps=2500.0,
)

obs, info = env.reset()
print(f"Trafik Kaynağı: {info['traffic_source']}, Zaman: {info.get('real_timestamp')}")

# 2. Bir adım ilerleme
obs, reward, terminated, truncated, info = env.step(6)
print(f"Toplam Talep: {info['total_demand_mbps']} Mbps, Gecikme: {info['avg_delay_ms']} ms")
```

