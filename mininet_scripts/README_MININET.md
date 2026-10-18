# Mininet & Ryu SDN Enerji Tasarruflu Ağ Ortamı

Bu klasör, projenin **Mininet** ve **Ryu SDN Denetleyicisi** ile gerçek bir Linux/WSL2 ağ ortamında çalıştırılabilmesi için otomatik olarak üretilen betikleri içerir.

## Dosyalar:
1. `fattree_topo.py`: Mininet k=4 Fat-Tree Veri Merkezi Topolojisi.
2. `ryu_green_controller.py`: OpenFlow 1.3 destekli, dinamik port uyku ve yeşil akış yönlendirme mantığına sahip Ryu uygulaması.

## Nasıl Çalıştırılır (Ubuntu / WSL / Mininet VM):

### 1. Terminal 1: Ryu Controller'ı Başlatın
```bash
ryu-manager ryu_green_controller.py
```

### 2. Terminal 2: Mininet Topolojisini Başlatın
```bash
sudo mn --custom fattree_topo.py --topo fattree --controller=remote,ip=127.0.0.1,port=6653 --switch ovsk --link tc
```

### 3. Mininet CLI Üzerinde Test:
```bash
mininet> pingall
mininet> iperf h_0_0_0 h_3_1_1
```
