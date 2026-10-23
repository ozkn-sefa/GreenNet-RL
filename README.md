# GreenNet-RL 🌿

**Pekiştirmeli öğrenmeyle ağ enerjisi ve hizmet kalitesi (QoS) optimizasyonu için Python tabanlı araştırma prototipi.**

GreenNet-RL; NetworkX topolojileri, Gymnasium ortamı ve Stable-Baselines3 ajanları kullanarak ağdaki enerji–performans ödünleşimini inceler. FastAPI tabanlı bir simülasyon paneli, baseline karşılaştırmaları ve isteğe bağlı SDN/Mininet deney betikleri içerir.

> **Araştırma durumu (23 Ekim 2026):** Abilene üzerinde 100.000 adımlık PPO pilotu tamamlanmıştır. Mevcut *router deep-sleep* eylemleriyle PPO, OSPF'ye karşı **%0 enerji tasarrufu** sağlamıştır; çünkü router kapatma bazı kaynak–hedef çiftlerini bağlantısız bırakmaktadır. Link/port seviyesinde enerji kontrolü bir sonraki araştırma aşamasıdır; henüz doğrulanmış sonuç olarak sunulmamaktadır.

## Özellikler ve kapsam

- **Topolojiler:** Fat-Tree, NSFNET ve SNDlib Abilene için NetworkX tabanlı ağ modelleri.
- **RL:** Gymnasium özel ortamında Stable-Baselines3 PPO; ayrıca DQN ve OSPF/ECMP, Static Heuristic, Threshold baseline altyapısı.
- **Enerji modeli:** Switch şasisi, port durumu, LPI ve geçiş maliyetlerini ayıran **simüle edilmiş** enerji hesapları; gerçek donanım ölçümü değildir.
- **QoS:** Yönlendirme, uçtan uca gecikme, paket kaybı, SLA ihlalleri, enerji tüketimi ve switching ölçümleri.
- **Veri:** Gerçek SNDlib Abilene 5 dakikalık talep matrisleri; sentetik trafik senaryoları ayrıca desteklenir.
- **Arayüz:** `run.py` üzerinden yerel web paneli.
- **SDN:** PolicyBridge ve Ryu/Mininet için deneysel entegrasyon; canlı Linux/OVS üzerinde uçtan uca doğrulama henüz tamamlanmamıştır.

## Hızlı başlangıç — Windows PowerShell

**Gereksinimler:** Python **3.12** önerilir. Projedeki `numpy<2` bağımlılığı nedeniyle Python 3.13 ortamında kurulum başarısız olabilir.

```powershell
# Proje kök dizininde
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pytest
python -m pytest tests -q
python run.py
```

Sunucu açıldıktan sonra `http://127.0.0.1:8000` adresine gidin. Panelin başlaması, tüm model/SDN entegrasyonlarının uçtan uca doğrulandığı anlamına gelmez.

## Gerçek Abilene trafik verisi

Trafik kaynağı: **SNDlib / Yin Zhang Abilene Traffic Matrices**, Native 1.0 biçimi (`MBITPERSEC`), 5 dakikalık ölçümler. Toplam **48.096** zaman damgalı talep matrisi, her matriste 12 düğüm arasında 132 yönlü kaynak–hedef talebi vardır.

- Resmî veri sayfası: https://sndlib.put.poznan.pl/dynamicmatrices.overview.action
- Ham veriyi arşiv olarak ve/veya çıkarılmış dosyalar halinde `data/raw/abilene/` altında tutun.
- `data/raw/` ve `data/processed/` Git'e dahil edilmez. Veri seti depoyu klonlamakla otomatik gelmez.
- Verilerin birimi zaten **Mbps**'tir; tekrar çarpan uygulamayın.

Örnek dizin:

```text
data/
├── raw/abilene/
│   ├── directed-abilene-zhang-5min-over-6months-ALL-native.tgz
│   └── directed-abilene-zhang-5min-over-6months-ALL-native/
│       └── demandMatrix-abilene-zhang-5min-*.txt
└── processed/                  # Oluşturulan indeks/önbellek
```

**Kronolojik veri ayrımı (karışık rastgele bölme yok):**

| Bölüm | Matris sayısı | Zaman aralığı | Kullanım |
|---|---:|---|---|
| Train | 33.667 (%70) | 2004-03-01 00:00 – 2004-07-21 21:30 | Ağırlık güncellemeleri |
| Validation | 7.214 (%15) | 2004-07-21 21:35 – 2004-08-15 22:40 | Checkpoint / hiperparametre seçimi |
| Test | 7.215 (%15) | 2004-08-15 22:45 – 2004-09-10 23:55 | Nihai bağımsız değerlendirme |

Ölçüm aralıklarında dört tarihsel kesinti bulunmuştur; eğitim ve değerlendirmede süreklilik varsayılmamalıdır. Gerçek veri seçiliyken eksik veri nedeniyle sentetik trafiğe sessiz geçiş yapılmaz.

## Eğitim ve benchmark

Cihaz karşılaştırmasında gerçek Abilene `train` bölümünde, 10.000 adım ve seed 42 ile **CPU 371,27 adım/sn**, **RTX 4060 Laptop CUDA 260,75 adım/sn** ölçülmüştür. Bu küçük MLP + CPU tabanlı simülasyon için **CPU** tercih edilir; değerler kullanılan makineye özgüdür.

Güncel eğitim seçeneklerini görüntüleyin:

```powershell
python scripts/train.py --help
python scripts/benchmark_ppo_device.py --help
```

100.000 adımlık pilot için `scripts/train_pilot_abilene.py` geliştirilmiştir. Çalıştırmadan önce betiğin CLI seçeneklerini ve çıktı dizinlerini inceleyin. Eski eğitilmiş modelleri koruyun; yeni deneyleri ayrı klasörlerde yürütün.

**Pilot yapılandırması:** PPO, `MlpPolicy [128,128]`, CPU, seed 42, `learning_rate=3e-4`, `n_steps=1024`, `batch_size=256`, `n_epochs=10`, `gamma=0.99`, `gae_lambda=0.95`, `clip_range=0.2`, `ent_coef=0.01`. Eğitim 100.000 adımda 204,27 saniye sürmüştür. Validation model seçimi için kullanılmış, test bölümü kullanılmamıştır.

Önemli dosyalar:

```text
src/data/abilene_loader.py             # Native veri yükleyici / kronolojik ayrım
src/network/topology.py                # Ağ topolojileri
src/network/power_model.py             # Simüle edilmiş güç modeli
src/env/green_network_env.py          # Gymnasium ortamı
src/agents/rl_agent.py                 # PPO / DQN
src/agents/baselines.py                # Referans ajanlar
scripts/train.py                       # Genel eğitim CLI
scripts/train_pilot_abilene.py         # Abilene pilot eğitimi
scripts/benchmark_ppo_device.py        # CPU–GPU hız karşılaştırması
results/pilot_abilene_100k_results.json
results/cpu_gpu_benchmark_results.json
REVISION_LOG.md                        # Teknik revizyon geçmişi
```

## Bulgular — enerji ve QoS

Kök neden incelemesinde iki ölçüm hatası giderildi:

1. `total_power_w` alanı yanlış anahtardan okunuyordu; enerji tasarrufu gerçekte sıfır olmayan baseline'lar için yanlışlıkla `%0` raporlanıyordu.
2. Ortamın 1000 Mbps sabit kapasitesi Abilene'nin 2500 Mbps bağlantı kapasitesini eziyordu; bu durum yapay darboğaz ve SLA ihlali oluşturuyordu. Topolojiye özgü varsayılan kapasite kullanımı düzeltildi.

Düzeltme sonrasında gerçek Abilene **validation** trafiğindeki seçili pencere sonuçları:

| Trafik penceresi | Politika | Güç (W) | OSPF'ye göre tasarruf | SLA ihlali | Paket kaybı |
|---|---|---:|---:|---:|---:|
| Düşük | PPO / OSPF | 1809,7 | %0 | %0 | %0 |
| Düşük | Static | 1703,5 | %5,9 | %35,3 | %13,89 |
| Düşük | Threshold | 1505,7 | %16,8 | %100 | %38,42 |
| Orta | PPO / OSPF | 1814,2 | %0 | %0 | %0 |
| Pik | PPO / OSPF | 1816,8 | %0 | %2,7 | %0,67 |

**Yorum:** Static/Threshold'un görünen enerji kazancı, bazı trafik taleplerinin bağlantısız veya kayıplı kalması pahasına elde edilmiştir; **QoS korunarak enerji kazancı** olarak sunulamaz. PPO, Abilene'de tüm çekirdekleri aktif tutan Action 7'ye yakınsamıştır. Bunlar pilot/validation sonuçlarıdır, bağımsız final test sonucu değildir.

Abilene'de bir router'ı tamamen kapatmak yerel PoP/host erişilebilirliğini bozabilir. Bu nedenle WAN üzerinde bir sonraki çalışma, router'ları açık tutup link/port seviyesi enerji durumlarını ve güvenli yeniden yönlendirmeyi değerlendirmektir. **EEE/LPI güç düşüşleri şu an simülasyon varsayımıdır; ölçülmüş donanım tasarrufu değildir.**

## Testler ve doğrulama

```powershell
python -m pytest tests -q
```

23 Ekim 2026 tarihli revizyon raporunda **92 testin geçtiği** bildirildi (88 mevcut + 4 Abilene regresyon testi). Bu, gerçek ağ üzerindeki davranışı veya enerji tasarrufunu kanıtlamaz.

## Bilinen sınırlamalar

- Abilene için mevcut router-level deep-sleep eylem uzayı güvenli enerji optimizasyonuna uygun değildir; link/port seviyesinde kontrol henüz geliştirilmemiştir.
- Pilot PPO modeli OSPF'ye karşı enerji tasarrufu üretmemiştir; başarısız deney sonuçları saklanmıştır.
- Pilotun ilk sürümünde yanlış etiketlenen `best_model.zip` kaldırılmıştır; hiçbir checkpoint'in katı güvenlik kriterlerini karşıladığı iddia edilmemelidir.
- EEE/LPI, uyku ve güç tüketimi rakamları simülasyon düzeyindedir; gerçek ölçüm değildir.
- Canlı Mininet/Open vSwitch/Ryu entegrasyonu Linux test ortamında henüz uçtan uca doğrulanmamıştır.
- Nihai `test` bölümüne dayalı çoklu-seed karşılaştırması henüz yapılmamıştır.

Ayrıntılı teknik kronoloji için [`REVISION_LOG.md`](REVISION_LOG.md) dosyasına bakın.

## Lisans ve kaynak kullanımı

Bu depodaki kodun yeniden kullanım koşulları için varsa depo lisans dosyasını kontrol edin. Abilene trafik verileri projeye dahil edilmemiştir; verilerin kullanım ve atıf koşulları için **SNDlib** ve orijinal Yin Zhang veri kaynağını esas alın.
