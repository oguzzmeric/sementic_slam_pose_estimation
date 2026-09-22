# CLAUDE.md — Monoküler Görsel-Anlamsal SLAM

Kullanıcıyla **Türkçe** konuş. Kod yorumları ve commit mesajları İngilizce.

## Proje

GNSS bağımsız İHA seyrüseferi. Nadir açılı (70–90°) tek kameradan metrik
konum kestirimi. Hareketli nesneler YOLO ile maskelenir, boyutu bilinen
araçlar metrik referans olarak kullanılır.

Hedef problem, TEKNOFEST Havacılıkta Yapay Zeka yarışmasının **2. Görevi**
ile birebir aynı (şartname: `docs/` altında). Veri seti de o yarışmanın.

**Önemli bağlam:** Bu proje aynı zamanda kullanıcının mezuniyet sonrası
(yaklaşık 1 ay içinde) endüstriyel şirketlere göstereceği bir portfolyo
projesi. Bu, araç/kütüphane seçimlerini etkiliyor — "aynı sayıyı ver"
yetmiyor, mümkün olduğunda endüstri standardı araçlar (örn. GTSAM) tercih
ediliyor, çünkü mülakatta bunun bir sinyal değeri var.

## Çalışma kuralları — ÖNEMLİ

1. **Ölçmeden değiştirme, ölçmeden açıklama.** Her değişiklik: hipotez →
   aynı config ile önce/sonra ölçüm → karar. "Muhtemelen şundandır" deme,
   ölç.
2. **Tek seferde tek değişiklik.** İki şeyi aynı anda değiştirirsen hangisinin
   etkilediği bilinmez. Bu projede bu hata iki kez yapıldı.
3. **İyileşme iddiası için önce/sonra tablosu ver.** Aynı `frame_step`, aynı
   koordinat bayrakları, aynı kare sayısı. Config farklıysa karşılaştırma
   geçersiz.
4. Değişiklikler kullanıcı onayı olmadan büyük refaktör içermesin. Kullanıcı
   kodu öğreniyor — her değişikliğin **neden** yapıldığını açıkla.
5. Yalnızca kanonik dosyaları düzenle. `core/` altında `*_clean.py`,
   `*_final.py`, `*_v2.py` gibi eski kopyalar var — dokunma, import edilmiyorlar.
6. Sabit kodlanmış parametre yok. Her ayar `config.yaml`'dan gelir.
7. **Kaydetmeden önce dikkat: editördeki eski bir sekmeyi ctrl+S ile
   "overwrite" etmek, diskteki güncel/düzeltilmiş kodun üzerine eski hali
   yazabilir** (22 Eylül'de tam olarak bu oldu, bkz. aşağıdaki oturum notu).
   Şüpheli bir durumda önce `git diff` ile dosyanın gerçekten neyi
   içerdiğini kontrol et.

## Ortam

- Windows + PowerShell, venv. Proje kökünden çalıştır.
- Modülleri `-m` ile çalıştır: `python -m core.scale_recovery` (dosya içinden
  `python dosya.py` çalıştırınca `utils` import edilemez).
- PowerShell here-string içinde `\"` kaçışı bozuluyor — tek tırnak kullan.
- Cache temizliği:
  ```powershell
  Get-ChildItem -Recurse -Filter "*.pyc" | Remove-Item -Force
  Get-ChildItem -Recurse -Filter "__pycache__" | Remove-Item -Recurse -Force
  ```

## Komutlar

```powershell
python main.py --max-frames 900          # tam koşu (frame_step=5 ile 450 kare)
python -m utils.evaluate_evo             # SE(3)/Sim(3) ATE, ölçek faktörü
python errdig.py                         # kare bazlı hata profili
python -m core.scale_recovery            # ölçek modülü tek başına
```

## Mimari

```
data_loader → camera_calibration → feature_extractor → matcher
  → motion_estimator → scale_recovery → pose_graph → visualizer / evaluate_evo
```

| Dosya                         | Sorumluluk                                                   | Çıktı                       |
| ----------------------------- | ------------------------------------------------------------ | --------------------------- |
| `utils/data_loader.py`        | Tek dosya erişim noktası, lazy load, generator, `frame_step` | kare, GT, tespit            |
| `utils/camera_calibration.py` | K, K⁻¹, undistort (remap), izdüşüm                           | temiz kare                  |
| `core/feature_extractor.py`   | ORB + semantik maske (`detectAndCompute(gray, mask)`)        | `FrameFeatures`             |
| `core/matcher.py`             | BF Hamming, kNN k=2, Lowe 0.75                               | `MatchResult`               |
| `core/motion_estimator.py`    | RANSAC H+E, hibrit seçim (R_H>0.45), ayrıştırma              | `PoseEstimate` (R, birim t) |
| `core/scale_recovery.py`      | Isınma + `s = k·Z` (+ opsiyonel akış tabanlı ölçek)          | `ScaleResult`               |
| `core/pose_graph.py`          | `T_world = T_world @ T_local`, Umeyama Sim(3), ATE           | `TrajectoryPoint`           |
| `yolo_preprocessing.py`       | Offline YOLO → `data/detections.json`                        | —                           |

Modüller arası veri sözleşmeleri (dataclass) değişmemeli — Faz 2'de ORB yerine
SuperPoint, BF yerine LightGlue takılacak ve aşağı akış etkilenmemeli.

## Veri seti gerçekleri

- 2250 kare, 3840×2160, **7.5 fps**, `.webp`. GT: `translation_x/y/z`.
- GT **bağıl** ve **NED** (Z aşağı pozitif). Yerden yükseklik:
  `Z_agl = 41.6 − Z_gt`. 41.6 m ofseti bbox kestirimiyle ölçülerek bulundu
  (korelasyon −0.67 → toplam sabit).
- Kalibrasyon 4000×3000'de yapıldı, video 3840×2160. Türetilmiş değerler:
  `fx=2680.51 fy=2683.39 cx=1908.48 cy=1139.71` (dikey 375 px kırpma + ×0.96).
  **Doğrulanmadı.**
- `frame_step: 5` bir **geliştirme** ayarı. Ardışık karelerde baseline çok
  küçük (H ve E dejenere oluyordu). Yarışmada her kare gönderilir — ileride
  ara kare stratejisi gerekecek.
- Isınma sayımı işlenen kare üzerinden: `warmup_frames: 150` × 5 = 750 orijinal
  kare. Şartname yalnızca ilk 450 kareyi garanti ediyor. Şimdilik bilinçli.

## Yarışma metriği

Şartname Denklem 2 — **hizalamasız ortalama 3B Öklid hatası**:

```
E = (1/N) Σ sqrt((x̂−x)² + (ŷ−y)² + (ẑ−z)²)
```

Sim(3) değil, RMSE değil. `main.py` şu an RMSE veriyor; yarışma için
ortalama da raporlanmalı.

## Mevcut durum (22 Eylül, 450 kare, DLT fix + retval gate + Umeyama Sim(3) + flow-scale)

```
geçerli poz            : 444 / 450
HIZALANMAMIS mean      : 23.12 m     <- flow-scale kalibrasyonuyla, once 25.37 m
Sim(3) ATE (evo)       : 24.03 m     (flow-scale oncesi: 26.15 m)
SE(3) rigid ATE (evo)  : 26.59 m     (flow-scale oncesi: 27.55 m)
olcek faktoru          : 0.8744      (flow-scale oncesi: 1.1242)
hız                    : ~3.6-4.0 FPS
```

**Önemli:** eski "ham/hizalanmamış" kötü sayılar (135-244 m) VO'nun kendisinin
kötü olmasından değil, **yanlış koordinat çerçevesinde raporlamaktan**
kaynaklanıyormuş (`swap_xy`/`flip_y` kaba tahmini + skaler `k`). Bkz.
aşağıdaki "Umeyama Sim(3) entegrasyonu" bölümü. Alttaki yön hatası (~20-27°)
**hâlâ açık** — ama artık bunun üzerindeki her iyileştirme doğrudan nihai
metriğe yansıyacak.

## Son düzeltme (21 Eylül) — DLT triangulation hatası

`core/motion_estimator.py::_decompose_homography` içinde nokta normalize
koordinattaydı (`K_inv` uygulanmış) ama projeksiyon matrisleri `K` içeriyordu:

```python
p1 = K_inv @ [u1, v1, 1]                 # normalize
P1 = K @ [I | 0]                         # piksel uzayı  ← uyumsuz girdi
```

Düzeltme (satır ~303, ~322): `P1 = [I | 0]`, `P2 = [R | t]` (K olmadan),
`p1/p2` normalize kaldı. Geri izdüşüm kontrolü (piksel uzayı, `fx·X/Z+cx`)
aynen kaldı — sadece DLT'ye giren `P1`/`P2`'den `K` çıkarıldı.

Doğrulama: `python main.py --max-frames 900` → `python -m utils.evaluate_evo`
ve `python errdig.py`, aynı config (`frame_step=5`, `altitude_source=gt`,
`warmup_frames=150`). Sonuç yukarıdaki tabloda. `H selected` 0'dan 54'e
çıktı — homography artık gerçekten oy alıyor. **Yön hatası bu düzeltmeyle
değişmedi** (beklenen buydu — DLT hatası triangulation/vote mekanizmasını
bozuyordu, R seçiminin kendisini değil).

Ayrı not: `utils/camera_calibration.py` içinde `undistort()`,
`_compute_undistort_maps()`, `project_point()`, `backproject_point()`
metotlarında yanlışlıkla bırakılmış (2298dae ile commit'lenmiş) 4 adet
`print()` vardı — her karede tüm undistort map'ini (3840×2160 float array)
stdout'a basıyordu. Hesaplamayı etkilemiyordu, temizlendi (bu ölçümü almak
için gerekliydi — çıktı okunamıyordu).

Kök dizindeki `gate2.py`, `gate3.py`, `tridiag.py` teşhis betikleri
`core/motion_estimator.py`'yi import etmiyor, DLT'yi kendi içlerinde
bağımsız tekrar yazıyorlar — hâlâ eski (K'lı) hatayı taşıyorlar, bu
düzeltmeyi yansıtmazlar. Güncellenmedi.

## Umeyama Sim(3) entegrasyonu — relocalization Faz A (21 Eylül)

**Neden:** Yarışmanın soru-cevap dokümanı (`Soru_Cevap_Gruplanmis_TR.pdf`,
GitHub'daki resmi repo — `TEKNOFEST-YARISMALAR/havacilikta-yapay-zeka-yarismasi`)
netleştirdi: *"Sim3 kullanılmıyor. Doğrudan RMSE karşılaştırması yapılır;
**ölçekleme ve hizalamayı yarışmacı yapar**."* Yani sunucu hizalama yapmıyor
— takımın kendi VO çıktısını GT çerçevesine (ölçek + yönelim + öteleme ile)
hizalayıp öyle göndermesi bekleniyor. Diyagram: *"1. Kalibrasyon (ilk 450
kare): ölçek, yönelim ve öteleme hesaplanır... 3. GT Geri Geldiğinde:
Konum anında GT'ye eşitlenir, birikmiş drift sıfırlanır, ölçek ve yönelim
güncellenir."*

Bizim eski mimarimiz bunu **iki ayrı, kaba yamayla** yapıyordu:
`scale_recovery.py`'de sadece **skaler** bir `k` (yönelimi hiç düzeltmiyor),
`pose_graph.py`'de `sweep.py` ile kaba kuvvetle bulunmuş **sabit**
`swap_xy`/`flip_y` (bu datasete özel, genellemez — tıpkı `R_H` eşiği
tartışmasındaki overfit endişesiyle aynı kategoride).

**Değişiklik (`core/pose_graph.py`):** `swap_xy`/`flip_y` kaldırıldı.
Bunun yerine: `scale_result.mode == "warmup"` olduğu her karede
(lokal_pozisyon, GT_pozisyon) çifti biriktiriliyor; GT kullanılabilir
olduğu sürece pozisyon doğrudan GT olarak raporlanıyor (tahmin etmeye
gerek yok, zaten biliniyoruz). Warmup bitip GT kesildiği anda, biriken
çiftlerden **tek seferlik Umeyama Sim(3)** (`s, R, t`) uyduruluyor ve
sonraki her karede `pozisyon = s·R·lokal_pozisyon + t` ile uygulanıyor.
`config.yaml`'a `min_sim3_samples: 10` eklendi (fit için gereken asgari
örnek — sabit kodlanmadı).

**Ölçüm (aynı config, `frame_step=5`, 450 kare):**

```
                          onceki (DLT+gate, k*Z+swap/flip)   sonraki (Umeyama Sim3)
HIZALANMAMIS mean          ~205-225 m                          25.37 m
Sim(3) ATE (evo)           ~26-52 m                            26.15 m
```

Eski ham metrik ile yeni Sim3-hizalı teşhis metriği (25.37 vs 26.15) artık
**neredeyse eşit** — üretimdeki hizalama diagnostic'lerimizin bulduğu en
iyi sonuca çok yakın, yani hizalama artık neredeyse optimal.

**Kalıntı sorun:** `errdig.py` üretim çıktısını kendi Umeyama'sıyla tekrar
hizalayınca ölçek **1.1242** buluyor (1.0 değil) — warmup'ta öğrenilen
Sim3 sabit, autonomous fazdaki `k·Z` kaymasını (Görev 2) telafi edemiyor.
Bu, hem kalibreli-akış ölçeğinin (bugüne kadar test edilen, henüz entegre
edilmemiş) hem de Faz B'nin (periyodik relocalization — GT geri
döndüğünde yeniden fit) neden hâlâ gerekli olduğunu gösteriyor.

**Not:** Görev 1 (taban yön hatası ~20-27°) ve Görev 2 (ölçek kararsızlığı)
bu değişiklikle **çözülmedi** — onlar hâlâ açık, aşağıda duruyor. Ama artık
üzerlerindeki her iyileştirme doğrudan nihai (hizalanmamış) metriğe
yansıyacak; önceden yanlış çerçeve hatası bunları gölgeliyordu.

## Açık görevler — öncelik sırasıyla

### 1. Yön (heading) hatası — DLT düzeltmesinden sonra da duruyor (EN ÖNCE)

Std 71.5°→72.9°, kare oranı %37.9→%37.6 — pratikte değişmedi. Sim(3)
hizalaması sonrası yön hatası çok yüksek kalmaya devam ediyor; RMSE'nin asıl
kaynağı bu, ölçek değil (ölçek faktörü zaten 0.976 — 1'e yakın).

**Hipotez A — GT dönüş hızıyla korelasyon (`turncheck.py`): REDDEDİLDİ.**
Korelasyon -0.008. Düşük turn-rate karelerinde bile medyan hata zaten
20-30° — hata olaya (dönüş) bağlı değil, sürekli/sistematik.

**Hipotez B — E yolunda disambiguation güveni yok: DOĞRULANDI (`hecheck.py`).**

`_decompose_essential` (satır 433-453), `cv2.recoverPose`'un 4 adaydan
seçtiği R,t'nin kaç nokta tarafından cheirality ile doğrulandığını
(`retval`, ilk dönüş değeri) okuyup **atıyor** (`_, R, t, _ = ...`). H
yolunda olan "yetersiz destek varsa pozu reddet" güvencesi E yolunda yok.
`min_inlier_count` eşiği de (config, 40) sadece log basıyor, `pose.is_valid`
etkilemiyor (satır 566-571) — fiilen işlevsiz. **Bu ayrı bug hâlâ
düzeltilmedi, açık.**

`retval / recoverPose'a giren nokta sayısı` oranını gerçek pipeline'dan
(monkeypatch ile, üretim kodunu değiştirmeden) örnekleyip yön hatasıyla
koreleyince monoton bir ilişki çıktı:

```
oran=1.0 (tam konsensus)  n=322 (%82)  |yon_hata| medyan 27.5°
oran<0.9                  n=35         |yon_hata| medyan 42.0°
oran<0.5                  n=18         |yon_hata| medyan 76.2°
oran<0.1                  n=7          |yon_hata| medyan 119.9°
retval=0 (hic destek yok) n=4          |yon_hata| medyan 122.9°
```

4 kare (`frame_000035`, `frame_001510/1515/1520`) `retval=0` — OpenCV
"hiçbir aday güvenilir değil" diyor ama pose yine de kabul ediliyor, ve
tek başına 120-180° hata üretiyor.

**Önemli:** tam konsensuslu (oran=1.0) karelerde bile medyan hata hâlâ
27.5° — bu hipotez sıçramaların/en kötü kuyruğun büyük kısmını açıklıyor,
ama tabandaki (baseline) hatayı açıklamıyor. Yön probleminin tamamı bu
değil.

**Uygulanan düzeltme — `retval<=0` gate (21 Eylül): SONUÇ KARIŞIK, geri
alınmadı ama şüpheli.**

`_decompose_essential`'a H'deki `best_votes<=0` gate'inin eşi eklendi:
`retval<=0` ise pozu reddet (satır ~447-454). 4 kare artık invalid
(444/448 geçerli poz). Aynı config, aynı komutlar:

```
                    once (DLT fix sonrasi)   sonra (retval gate)
Sim(3) ATE (evo)    51.80 m                  25.99 m   <- cok iyilesti
Sim(3) olcek        0.9758                   1.1436    <- 17% sicradi
ham RMSE            135.7 m                  219-244 m <- kotulesti (2 kosuda)
Mean error (ham)    108.6 m                  204.9 m   <- 2x kotulesti
```

**Çelişki:** Sim(3)-hizalı metrik (trajectory'nin *şekli*) çok iyileşti,
ama **hizalanmamış** metrik (yarışmanın gerçekte kullandığı metrik,
"Yarışma metriği" bölümüne bak) iki katına çıktı. Ölçek faktörünün
0.976'dan 1.144'e sıçraması, 4 kareyi düşürmenin toplam kat edilen mesafe/
warmup kalibrasyonunu (`_k_history`, `_scale_history`) beklenenden fazla
etkilediğini düşündürüyor — kök neden doğrulanmadı.

RANSAC stokastik olduğu için (`findHomography`/`findEssentialMat`)
kareler arası küçük varyans normal (aynı "sonra" durumunu iki kere
çalıştırınca RMSE 219 ve 244 çıktı) ama 108→205 m'lik fark bu varyansla
açıklanamayacak kadar büyük.

**Karar (kullanıcı, 21 Eylül): gate TUTULUYOR.** Gerekçe: Sim(3)-hizalı
sonuç trajectory'nin *şeklinin* artık GT'ye çok daha yakın olduğunu
gösteriyor (25.99 m) — bu yapısal olarak doğru davranış. Ham metrikteki
kötüleşme muhtemelen tek bir global ölçek/offset kalibrasyon sorunu
(ölçek 0.976→1.144 sıçraması) ve ayrıca çözülebilir; şekli bozuk bir
trajectory'i hiçbir kalibrasyon kurtaramaz, o yüzden önce şekil
düzeltiliyor. Ölçek sıçramasının kök nedeni **Görev 2**'de bulundu
(`k` sabit varsayımıyla aynı problem çıktı).

**Not (22 Eylül):** Umeyama Sim(3) entegrasyonu sonrası bu tablo artık
tarihsel — güncel sayılar yukarıdaki "Mevcut durum" bölümünde. Gate hâlâ
kodda ve tutuluyor.

**Hipotez C — taban hata rastgele mi, biriken zincirleme mi: KISMEN
DOĞRULANDI (`biascheck.py`).**

İşaretli (signed) adım-başı yön hatası üzerinde 4 test:

```
sistematik yanlilik (medyan/ortalama)   : -3.94 / -6.91 deg  (kucuk ama anlamli, p<0.001)
ardisik adimlar otokorelasyonu (lag-1)  : +0.38             (bagimsiz olsa ~0 olurdu)
|hata| ~ inlier_count korelasyonu       : +0.11             (zayif)
|hata| ~ R_H orani korelasyonu          : +0.23             (zayif-orta)
en uzun ayni-yonlu hata dizisi          : 49 kare  (ortalama 5.7; rastgele icin beklenen ~2.0)
```

Sonuç: ne saf rastgele gürültü (otokorelasyon ~0 olmalıydı), ne de sabit/
büyük bir yanlılık (bias küçük, std'nin çoğunu açıklamıyor). Hata
**sürüklenen/kalıcı** — bir kare küçük hata yapınca `pose_graph.py`'nin
zincirleme çarpımı (`T_world = T_world @ T_local`) bunu birkaç kare
boyunca taşıyor (klasik dead-reckoning random-walk). `R_H` oranıyla zayıf
korelasyon (0.23) bulunmuştu, aşağıda daha dikkatli test edildi.

**Hipotez D — `R_H` eşiğine (0.45) yakınlık daha güvenilmez mi:
REDDEDİLDİ (`rhcheck.py`, 21 Eylül).**

Not: 0.45 ORB-SLAM'in H/F seçim sezgisinden gelen bir literatür sabiti,
bu datasete özel tune edilmemiş — o yüzden bu hipotez tutsaydı bile
sonucu (belki eşik ayarı) dikkatle genellenmeliydi (kullanıcı bunu
sorguladı, haklı çıktı).

`|R_H - 0.45|` (eşiğe uzaklık) ile yön hatası arasında **anlamlı ilişki
yok**:

```
korelasyon(|R_H-0.45|, |yon_hatasi|)  = -0.15  (zayif, beklenenin tersi yonde)
esige yakin  (|R_H-0.45|<0.10)  n=203  |yon_hata| medyan = 20.43
esige uzak   (|R_H-0.45|>=0.10) n=240  |yon_hata| medyan = 20.10   <- neredeyse ayni
```

Mekanizma arayışı ilginç bir şey buldu ama beklenenin tersi: eşiğe yakın
kareler **daha fazla** eşleşen noktaya sahip (medyan 839 vs 561,
korelasyon -0.54) — "belirsiz/az veri" değil, "çok veriyle doğru ölçülmüş
ara durum". Dilim profilinde hata eşikten uzaklaştıkça düzgün
azalmıyor/artmıyor, dalgalanıyor; en uzak (en E-ağırlıklı) dilimde bile
hata yüksek kalıyor (28°).

**Sonuç:** önceki zayıf 0.23 korelasyon muhtemelen "eşiğe yakınlık"
değil, daha dağınık bir etkinin (genel nokta sayısı ya da E-ağırlıklı
olma) yan ürünüydü. Bu iz kapandı — ucuz mekanizma denemeleri tükendi.

**Sonuç: BA (motion-only, pencereli/multi-frame) bu sürüklenme türü için
makul bir aday** — sabit yanlılığı ya da ölçeği düzeltmez, ama zincirleme
sürüklenmeyi (ardışık tek-çift bağımlılığı yerine çoklu-komşu tutarlılığı
zorlayarak) azaltması beklenir. Sıralama değişmedi: önce Görev 2 (ölçek),
sonra BA (Görev 3) — ölçek kaymışken BA'nın şekil iyileştirmesi nihai
metriğe yansımaz.

**Ucuz ön-test — kör pencereli R yumuşatma (`windowcheck.py`, 21 Eylül):
BA'ya yatırım için destekleyici değil, ama neden anlaşıldı.**

Tam BA yazmadan önce, sadece komşu karelerin `R`'sini medyanla
yumuşatmanın (çeviri sabit kalarak) yardımcı olup olmadığına bakıldı:

```
pencere   Sim(3) ATE   HIZALANMAMIS mean   yon hatasi   otokorelasyon
1 (uretim)  26.06 m        224.85 m          20.24°        0.383
3           25.72 m        224.82 m (~ayni)  19.18°        0.370
7           24.09 m        238.00 m (kotu)   13.54°        0.372
15          30.12 m        264.41 m (kotu)   18.90°        0.633
31-61       surekli kotulesiyor
```

Pencere büyüdükçe Sim(3)/yön hatası bir noktaya kadar (7) iyi görünüyor
ama **hizalanmamış (asıl) metrik baştan itibaren kötüleşiyor** — kaba
medyan filtresi gürültüyü değil, gerçek dönüşleri de bulanıklaştırıyor.
Otokorelasyon da büyük pencerede artıyor (0.38→0.70) — kör ortalama
sürüklenmeyi çözmüyor, yeni bozulma ekliyor.

**Yorum:** bu, "komşuları kör kör ortala" yaklaşımının işe yaramadığını
gösteriyor — ama gerçek BA kör ortalama yapmaz, reprojection hatasına
(piksel kanıtına) bakarak optimize eder, bu yüzden gerçek dönüşleri
bulanıklaştırmaz. Yine de bu test BA'ya hemen büyük yatırım yapmayı
**güçlü şekilde desteklemiyor**; önce daha ucuz `R_H` eşiği ipucuna
(0.23 korelasyon, hipotez C'de bulunmuştu) bakılacak — bu ipucu da
Hipotez D'de kapatıldı, geriye gerçek BA kaldı.

### 2. Ölçek: `k` sabit varsayımı kırılıyor — Görev 1'deki ölçek sıçramasıyla AYNI KÖK PROBLEM

`s = k·Z`, `k = ω·Δt`. `ω` (açısal hız) sabit varsayılıyor ama drone hem
irtifayı hem hızı değiştiriyor (kullanıcı teyit etti — autonomous modda
ikisi de değişken). Ölçülen `k_ideal` uçuş boyunca 0.033–0.098, kullanılan
sabit ~0.047. Son dilimde oran 2.31.

**Kök neden doğrulandı (`scalecheck.py`, 21 Eylül):** retval gate sonrası
ölçek sıçramasının (Görev 1'deki gate tartışmasına bak) kaynağını
araştırırken aynı probleme çıkıldı.

- `k_factor` gate'ten neredeyse etkilenmedi (0.04691 — `kcheck.py`'deki
  ~0.047 ile aynı) → kalibrasyonun kendisi bozulmadı.
- Donmuş (invalid) kareler ölçek sıçramasının sebebi değil: toplam GT
  yolunun sadece **%0.28**'i (2.16 m / 775 m) — ihmal edilebilir.
- Asıl sebep: **adım uzunluğu sistematik olarak kısa tahmin ediliyor**,
  hem warmup'ta hem autonomous'ta:

  ```
  mod          GT/tahmin oran (medyan)
  warmup       1.067
  autonomous   1.307   <- daha kotu
  ```

  Toplam yol oranı (775 m GT / 663 m tahmin = 1.169) Sim3'ün bulduğu
  ölçek faktörüne (1.144) çok yakın — ölçek sıçraması gerçek ve bu
  kısalıktan geliyor.

- **Warmup'ta bile kısa çıkmasının sebebi:** `scale_recovery.py` warmup
  sırasında bile anlık `s_ref`'i değil, **çalışan medyanı**
  (`running = median(_scale_history)`, satır ~529) kullanıyor. Gerçek
  ölçek zaten değişken olduğu için bu yumuşatma sistematik olarak kısa
  tahmin üretiyor; autonomous'ta daha kötü çünkü o bölgede irtifa/hız
  warmup'takinden farklı.

**Sonuç:** ölçek sıçraması ayrı bir bug değilmiş — 4 dejenere E-karesi (retval=0) bu
kronik ölçek-altı tahmini yön hatasıyla tesadüfen kısmen maskeliyormuş;
onları temizleyince çıplak göründü. Tek problem: **sabit/yumuşatılmış
`k` gerçek zamanlı değişkenliği yakalayamıyor.**

**Aday çözümler:**

A) H çalışınca `decomposeHomographyMat`'ın döndürdüğü `t` aslında `t/d`
   (d = düzlem/irtifa mesafesi) — `‖t_metrik‖ = ‖t/d‖ · Z` ile kare-kare,
   sabitsiz bir ölçek elde edilebilir. Görev 1 (DLT) bitti, artık
   denenebilir. Sınırlama: H sadece 394 karenin 54'ünde (%12) seçiliyor.
   Nadir bakışta zemin genelde düzleme yakın olduğundan `R_H` eşiğinin
   (0.45) H'yi gereksiz yere az seçtirdiği düşünülebilir — ayrıca
   incelenebilir. **Henüz denenmedi.**
B) Ham optik akıştan ölçek (`s = akis_px · Z / f`, rotasyon düzeltmesiz).

   **Post-fix tekrar test edildi (`flowcheck2.py`, 21 Eylül): kalibrasyonsuz
   hâliyle KAZANMIYOR, ama ipucu var.**

   Üretim trajectory'siyle (mevcut `s=k·Z`) AYNI R, AYNI geçerlilik,
   sadece farklı ölçek büyüklüğü kullanılarak paralel bir "olsaydı ne
   olurdu" trajectory'si kuruldu (tek değişken: ölçek kaynağı):

   ```
                              uretim (k*Z)   akis (akis*Z/f)
   Sim(3) ATE                26.06 m        28.97 m
   HIZALANMAMIS mean (yarisma)  224.85 m       247.03 m   <- akis daha kotu
   toplam yol orani (GT/tahm)   1.169          0.885      <- zit yonde yanli
   adim olcegi orani medyan     1.163          0.945      <- akis GT'ye daha yakin!
   korelasyon(GT_adim, olcek)   +0.555         +0.665     <- akis daha iyi izliyor
   ```

   Yorum: akış GT'nin gerçek adım-adım değişkenliğini üretimden **daha iyi
   izliyor** (korelasyon 0.665>0.555, medyan oranı 1.0'a daha yakın) ama
   **ters yönde** ve **daha gürültülü** bir sistematik sapması var (üretim
   %17 kısa tahmin ediyor, akış %13 uzun tahmin ediyor — std'si de daha
   yüksek: 0.557 vs 0.433). Ham/kalibrasyonsuz haliyle bu daha büyük
   gürültü, daha iyi korelasyonun faydasını götürüyor.

   **Kalibrasyon çarpanı denendi (`flowcheck3.py`, 21 Eylül): KÜÇÜK AMA
   TUTARLI İYİLEŞME, doğrulandı.**

   Warmup FAZI üç varyantta da üretimin kendi ölçeğini kullanacak şekilde
   sabitlendi (adil kıyas — gerçek konuşlanmada warmup zaten referans
   sinyali kullanıyor), sadece AUTONOMOUS fazda ayrıştılar:

   ```
                                 uretim    akis-ham   akis-kal (c=0.9552)
   HIZALANMAMIS mean (yarisma)  224.85 m  224.62 m   223.79 m
   Sim(3) ATE                   26.06 m   24.66 m    24.78 m
   toplam yol orani (GT/tahm)   1.169     0.938      0.969  <- 1'e en yakin
   ```

   İki kez tekrar çalıştırıldı, sonuç bit-bit özdeş çıktı (RANSAC
   gürültüsü değil, kararlı). `c=0.9552` — ham akışın zaten hafif (~%5)
   fazla tahmin ettiğini gösteriyor, kalibrasyon bunu düzeltiyor.

   **Gürültü azaltma denemesi (`flowcheck4.py`): ETKİSİZ.** Akış
   medyanını tüm eşleşen noktalar yerine sadece RANSAC inlier noktalarıyla
   hesaplamak hemen hemen hiçbir şey değiştirmedi (224.58 vs 224.62) —
   medyan zaten outlier'a dayanıklı, sorun onlardan gelmiyor. Kalan
   gürültünün kaynağı muhtemelen sahne derinliği çeşitliliği (motion
   parallax) ya da rotasyon katkısı (düzeltmesi zaten denenip reddedilmiş,
   bkz. "Denendi, reddedildi" tablosu) — kolay bir sonraki adım yok.

   **ENTEGRE EDİLDİ (22 Eylül).** `core/scale_recovery.py`'ye warmup'ta
   `c` öğrenme mekanizması eklendi (`_flow_c_history`, `k_factor`'ün
   öğrenildiği aynı desen — hardcode değil). `config.yaml`:
   `optical_flow_scale: true`. Üretim ölçümü yukarıdaki "Mevcut durum"
   bölümünde (`c=0.9552`, warmup'ta 147 örnekten öğrenildi — flowcheck3
   ile bit-bit aynı). Commit'lendi.
C) `k`'yi tek sabit yerine irtifa ve/veya hız bin'lerine göre öğrenmek
   (warmup verisiyle). Daha fazla veri/karmaşıklık ister, henüz
   tasarlanmadı.

### 3. Bundle adjustment (motion-only, pencereli) — scipy'de 4 tur denendi, hiçbiri üretimi geçemedi

Yön gürültüsü için. BA ölçeği düzeltmez (Görev 2 zaten flow-scale ile
kısmen ele alındı).

**Araç kararı (22 Eylül):** Önce GTSAM düşünüldü (bu proje kullanıcının
mezuniyet sonrası şirketlere göstereceği bir portfolyo projesi, endüstri
standardı bir SLAM backend'i kullanmak sonuç kalitesinden bağımsız bir
sinyal değeri taşıyor) ama GTSAM'in PyPI'da HİÇBİR sürümde (4.0.2 → 4.3.0)
Windows wheel'i olmadığı doğrulandı (sadece `macosx_*`/`manylinux*`).
Karar: önce ucuz olan scipy'de konsepti doğrula (aynı formülasyon
hatasını GTSAM'e taşımamak için — tool switch + formulation fix'i AYNI
ANDA yapmamak, kural #2), sonuç netleşince GTSAM/WSL'e geç.

**`bacheck.py` — 4 varyant, hepsi diagnostic, core/*.py'ye entegre
edilmedi:**

```
                                    URETIM      v1        v2        v3        v4
Sim(3) ATE                          24.14 m     67.72 m   65.60 m   60.69 m   52.96 m
HIZALANMAMIS mean (bu diagnostic'in
  kendi kaba hizalamasiyla)        136.84 m       --      105.78 m  199.54 m  83.78 m
```

- **v1 (naif):** sadece ardışık çift eşleşmeleri, gerçek çoklu-kare track
  yok — her pencere içi düzeltme aslında `cv2.recoverPose`'un zaten
  çözdüğü şeyi daha gürültülü bir yoldan tekrar çözüyordu.
- **v2:** gerçek çok-kareli track (aynı fiziksel nokta pencere boyunca
  zincirleniyor, `frame_010→011` ve `frame_011→012` eşleşmelerinin AYNI
  frame_011 keypoint'ini paylaştığı doğrulanarak) + düşük-paralaks gate
  eklendi (83/90 pencere kullanılabildi, pencere başı ort. 44 track) —
  yine de kötü.
- **v3:** sadece rotasyon değil, öteleme YÖNÜ de (uzunluk sabit, zaten
  kalibre) birlikte optimize edildi — küçük bir iyileşme (65.6→60.7m)
  ama hâlâ 2.5x kötü.
- **v4:** "prensipli" regularizasyon eklendi — `residual = f * delta *
  sqrt(N_inlier)`, yani her adımın kendi orijinal inlier sayısından gelen
  güvenle orantılı bir "orijinal tahminden çok uzaklaşma" cezası (sabit,
  datasete-özel bir `lambda` DEĞİL). En iyi sonuç (52.96m) ama hâlâ 2.2x
  kötü. **Bilinen sorun:** regularizasyon residual'lerinin mutlak
  büyüklüğü (`f≈2680 × √N≈7 ≈ 18800` çarpanı) reprojection residual'lerinin
  (birkaç piksel) ölçeğinden çok büyük çıkıyor, ve ikisi aynı huber
  `f_scale`'i paylaşıyor — yani ölçeklendirme hâlâ kabaca ayarlı,
  "prensipli" türetim doğru ama mutlak birim kalibrasyonu eksik.

**Tutarlı bir trend var** (her düzeltme sonucu bir öncekinden iyileştirdi:
67.7→65.6→60.7→53.0m) ama 4 turda da üretim ATE'sini (24.1m) geçemedik.
Bu, "windowed motion-only BA fikri tamamen yanlış" değil, "bu 6-kareli
pencere + sınırlı paralaks + kaba ölçeklendirmeyle yeterince iyi
çözülmüyor" sinyali.

**Karar (kullanıcı, 22 Eylül):** Burada durup WSL üzerinden GTSAM'e
geçiliyor. Gerekçe: gürültü-modeli/regularizasyon ölçeklendirmesini
GTSAM doğal olarak (factor noise model'leriyle) doğru yapıyor, scipy'de
elle kalibre etmeye çalışmak zaman kaybı olmaya başladı. `core/*.py`'ye
hiçbir BA kodu entegre edilmedi — üretim hâlâ Umeyama Sim3 + flow-scale
durumunda (bkz. "Mevcut durum").

**WSL kurulumu (22-23 Eylül) — kendisi bir yan hikaye.** `wsl --install`
sadece `VirtualMachinePlatform`'u etkinleştirmiş, `Microsoft-Windows-
Subsystem-Linux` ve `HypervisorPlatform` devre dışı kalmıştı (bu iki
Windows optional feature `wsl --install`'ın normalde otomatik açması
gereken şeyler, bu makinede açmadı — sebep bulunamadı). `wsl -d Ubuntu`
sessizce sonsuza kadar askıda kalıyordu, hiçbir hata/çıktı vermiyordu.
Admin PowerShell'de `Enable-WindowsOptionalFeature` ile ikisi elle
açılıp reboot edildi, sonra çalıştı. Ubuntu 26.04, kullanıcı `root`
(hiç interaktif kurulum sihirbazı tetiklenmedi çünkü hep `wsl -d Ubuntu
-- <komut>` ile programatik çağrıldı). Repo `~/drone_semantic_slam`'a
klonlandı, ham veri (`data/raw_frames` 1.5GB, `detections.json`)
Windows'tan (`/mnt/c/...`) kopyalandı. `pip install gtsam` (4.3.0)
sorunsuz çalıştı (manylinux wheel). Pipeline'ın torch/ultralytics'e
ihtiyacı olmadığı doğrulandı (sadece `numpy, opencv, scipy, pyyaml`
gerekiyor — `utils/visualizer.py`'de `from torch import gt` gibi
kullanılmayan bir import var ama biz visualizer'ı hiç import etmedik).

**GTSAM sonucu (`bacheck_gtsam.py`, 23 Eylül): scipy'den farklı bir
başarısızlık modu, ama hâlâ başarısızlık.**

```
                                    URETIM      GTSAM BA
Sim(3) ATE                          22.54 m     22.59 m    <- pratikte AYNI
HIZALANMAMIS mean                  147.94 m    147.44 m    <- pratikte AYNI
```

scipy'deki gibi KÖTÜLEŞMEDİ (v1-v4: 53-68m), ama hiç de İYİLEŞMEDİ —
GTSAM'e verilen prior (`sigma = piksel_gurultu/(f*sqrt(N))`, N≈40-80)
o kadar sıkı çıktı (~0.005°) ki optimizer'a pratikte hareket alanı
kalmadı. Kök sorun: bu formül "kaç nokta destekliyor" bilgisini
**rastgele gürültüyü** ölçmek için kullanıyor, ama aradığımız şey
**sistematik** bir hata — nokta sayısı artınca rastgele gürültü güveni
yükselir ama sistematik hata görünmez kalır (kendi kendine tutarlılık
kontrolü sistematik yanlılığı göremez, çünkü kanıtın kendisi de aynı
yanlı mekanizmadan üretiliyor).

**Sonuç: scipy (gevşek regularizasyon → kötüleşir) ile GTSAM (sıkı,
prensipli prior → hiç değişmez) aynı temel gerçeği iki uçtan gösteriyor
— 6 karelik pencerede piksel kanıtı, aradığımız sistematik yön hatasını
ayırt edecek kadar güçlü değil.**

### Disambiguation'ı kaynağında düzeltme denemesi (`disambigcheck.py`, 23 Eylül): REDDEDİLDİ, ama netleştirici

Hipotez: BA'yı seçim SONRASI değil, essential matrix'ten R,t seçilirken
(4 aday arasından) düzeltmek — belirsiz kararlarda 1-2 kare ötesine
bakmak. Test etmeden önce "gerçekten belirsizlik var mı" diye ölçüldü:

`core/motion_estimator.py`'ye DOKUNMADAN (monkeypatch ile `_decompose_
essential`'ın gördüğü E/mask'i yakalayıp), üretimin kullandığı E'yi
`cv2.decomposeEssentialMat` ile 4 adaya ayırıp kendi cheirality+
reprojection oylamamızı yaptık (motion_estimator'ın H tarafındaki
mantığın aynısı):

```
E secilen adim sayisi                          : 390
Bizim oylamamiz uretimin secimiyle eslesiyor   : 389/390   <- yontem dogru
BELIRSIZ adim (kazanan/runner-up oyu yakin)    : 0 / 390   (%0)
Kazanan yon hatasi (medyan)                    : 25.51°
Runner-up (2. aday) yon hatasi                 : 153.78°   <- cok daha kotu
```

**Hiçbir karede gerçek bir "hangi aday doğru, belirsiz" durumu yok** —
doğru aday her zaman ezici bir farkla seçiliyor. Yani sorun **hangi
adayı seçtiğimiz değil**; doğru aday zaten her zaman doğru seçiliyor
ve buna rağmen ~25° hata kalıyor.

**Bu, 21 Eylül'deki `hecheck.py` bulgusunu da netleştiriyor:** o zaman
"düşük retval → yüksek hata" korelasyonu "yanlış aday seçiliyor" olarak
yorumlanmıştı. Gerçekte: düşük retval = az nokta bu adımı destekliyor =
**doğru aday da zayıf/belirsiz tahmin ediliyor** (yanlış aday seçimi
değil). Kanıt zayıfken doğru cevap da kötü oluyor — hangi cevabı
seçtiğimizle ilgisi yok.

**Genel sonuç (23 Eylül, üç bağımsız yaklaşım da aynı duvara çarptı):**
scipy BA, GTSAM BA, ve disambiguation-kaynağında-düzeltme — üçü de
farklı açılardan aynı şeyi doğruluyor: `frame_step=5` aralığındaki
essential matrix kanıtı, aradığımız sistematik yön hatasını düzeltmek
için yeterince güçlü/hassas değil. Bu bir kod hatası değil, mevcut
veri/geometriyle ulaşılan bir sınır. Yön hatası şu an **açık, iyi
teşhis edilmiş ama çözülmemiş** bir problem olarak bırakılıyor.

**Not (kullanıcı, 23 Eylül):** proje süresi 1 hafta değil **1.5 ay** —
bu, kök nedene inmeye devam etmeyi (sadece sunuma geçmeyi değil)
mantıklı kılıyor.

### Uzun-bazlı KLT track denemesi (23 Eylül) — İLK KEZ küçük ama gerçek bir kazanç, sonra bir bug

**Adım 1 — sabit 15 kare pencere + KLT (`longtrack_gtsam.py`, ilk hali):**
`bacheck_gtsam.py`'nin 6 kareli pencere+pairwise-eşleşme-zincirleme
track yöntemi 15+ kareye uzatılamaz (hayatta kalma olasılığı çarpımsal
düşer: ölçülen ~%68/adım hayatta kalma ile 20 karede ~%0.02 — pratikte
sıfır). Bunun yerine KLT (Lucas-Kanade optik akış) kullanıldı: bir
noktayı pencerenin ilk karesinde bulup, sonraki her karede yeniden
eşleştirmeye çalışmadan, görüntü gradyanlarıyla doğrudan takip etmek.
İleri-geri (forward-backward) tutarlılık kontrolüyle sessizce kayan
noktalar elendi.

**Sonuç: İLK KEZ BA üretimi geçti (küçük ama gerçek):**

```
                                    URETIM      BA (KLT 15-kare + GTSAM)
Sim(3) ATE                          22.54 m     21.35 m    <- %5.3 iyilesme
HIZALANMAMIS mean (bu diagnostic'in
  kendi kaba hizalamasiyla)        147.94 m    143.96 m    <- %2.7 iyilesme
```

31 pencerenin sadece 21'inde track hayatta kaldı — 238-434 kare
aralığında (raw ~frame_001190+) çoğu pencerede **hiç** track kalmadı.

**Kök neden teşhisi (`klt_debug.py`):** sorun doku/köşe eksikliği değil
(400 köşe her zaman bulunuyor, maske alanı ~%100) — **kare arası hareket
büyüklüğü**. "İyi" pencerede (frame 0-70) kümülatif kayma 14 hopta
yavaşça ~89px'e çıkıyor; "çöken" pencerelerde (frame 1190+) kayma
6-10. hopta 100-400px'e fırlıyor — muhtemelen o segmentte irtifa/hız
farklı (daha önce ölçek analizinde de bulunmuştu: drone irtifa/hız
değiştiriyor). KLT penceresini/piramit seviyesini büyütmek
(31→63px, 4→6 seviye) hafif iyileştirdi ama temel sorunu çözmedi —
bazı segmentlerde kayma 300-400px'e çıkıyor, sabit uzunluklu hiçbir
pencere bunu güvenilir takip edemez.

**Adım 2 — uyarlanabilir (adaptive) pencere denemesi: BUG BULUNDU, henüz
çözülmedi.** Fikir: pencere uzunluğunu sabit tutmak yerine, kümülatif
kayma bir eşiği (85px — "iyi" penceredeki hâlâ güvenilir takip
seviyesinden türetildi, veri setine özel ayarlanmadı) geçince ya da
hayatta kalan track sayısı çok azalınca pencereyi kapatmak, hareket
büyüdükçe kendiliğinden kısalması. **Gözlenen anormallik:** ilk pencere
(frame 0-35) 8 kareye çıktı (beklenen), ama ondan SONRAKİ hemen hemen
HER pencere tam olarak 4 karede (minimum sınır, `MIN_WINDOW`) kesildi —
bu kadar düzenli bir kesilme gerçek hareketle açıklanamaz, mantıkta bir
hata olmalı (muhtemelen kümülatif kayma hesabında ya da durdurma
koşulunun `i>=MIN_WINDOW-1` ile erken tetiklenmesinde). Ayrıca rapor
satırında kaldırılan bir sabite (`LONG_WINDOW`) referans kalmış, script
en sonda çöküyordu — bu düzeltildi (kozmetik), ama pencere-uzunluğu
bug'ı **çözülmedi**.

**Yarın ilk iş:** `longtrack_gtsam.py`'deki `build_long_tracks`
fonksiyonunun durdurma mantığını (`med_disp`/`n_alive` hesaplarını,
`klt_debug.py` ile birebir karşılaştırarak) debug etmek — neden ikinci
pencereden itibaren hep tam `MIN_WINDOW` uzunluğunda kesildiğini bulup
düzeltmek, sonra ölçmek. `git status` ile dosyaların gerçekten ne
içerdiğini kontrol etmeden devam etme (bkz. aşağıdaki ctrl+S dersi).

**Bundan sonrası için adaylar (öncelik kullanıcıyla konuşulacak):**
(a) uzun-bazlı KLT track'i düzgün çalışır hale getirip (bug'ı çöz)
ölçmeye devam etmek — şu ana kadarki TEK pozitif sinyal bu yönden
geldi, (b) bağımsız bir dış referans (IMU yok ama periyodik GNSS
yeniden-yakalama zaten "Faz B" olarak planlıydı) ile sistematik hatayı
gerçekten düzeltmek, (c) burada durup dürüstçe belgeleyip projenin
sunum/temizlik tarafına geçmek — ama artık 1.5 aylık süre olduğu için
(c) aceleye getirilmeyecek.

## Oturum notu (22 Eylül) — ctrl+S/overwrite ile kayıp ve kurtarma

Oturum başında `core/motion_estimator.py` (DLT/K fix + retval gate) ve bu
dosyanın (`tanı.md`, 493→384 satır) **aynı anda** (16:35:47 / 16:35:56)
eski bir editör sekmesinin üzerine "overwrite" ile kaydedilmesiyle kısmen
eski haline dönmüştü — muhtemelen bir "tümünü kaydet" işlemi, açık
sekmelerden biri güncel değişiklikleri hiç görmemiş bir buffer'dı.

Tespit: dosya mtime'ları diğer değişen dosyalarla (21 Eylül) uyuşmuyordu;
kod içeriği kontrol edilince DLT fix (`P1`/`P2`'den `K` çıkarılması) ve
`retval<=0` gate'in kaybolduğu doğrulandı. `core/pose_graph.py` ve
`core/scale_recovery.py` etkilenmemişti (mtime 21 Eylül'de kalmıştı).

Kurtarma: iki fix elle tekrar uygulandı, `python main.py --max-frames 900`
+ `python -m utils.evaluate_evo` ile ölçüldü, sonuç 21 Eylül'deki rakamlarla
(Sim3 ATE 26.15 m, ölçek 1.1242) bit-bit eşleşti → doğru şekilde kurtarıldı.
Commit'lendi ve push edildi (`9158a71`).

**Ders:** editörde bu dosyalar üzerinde çalışırken kaydetmeden önce
`git diff` ile ne olduğunu kontrol etmek ucuz bir sağlık kontrolü.

## Denendi, reddedildi — tekrar deneme

| Yaklaşım                                                | Sonuç                                                                                                                                                                                                                     |
| -------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| SO(3) izdüşümü                                          | ATE kötüleşti (R zaten geçerli, hata geometrik)                                                                                                                                                                           |
| 30° / 90° açı filtresi                                  | Geçerli pozları eledi                                                                                                                                                                                                     |
| Sabit hizalama açısı                                    | Açı uçuş boyunca 200° yayılıyor                                                                                                                                                                                           |
| Optik akıştan ölçek (ham)                               | Adım oranı 1.20 → 1.29, Sim(3) ölçek 0.76                                                                                                                                                                                 |
| Optik akış, `K·R·K⁻¹` rotasyon düzeltmesi               | Korelasyon 0.59 → −0.10                                                                                                                                                                                                   |
| force_2d + dönüş sonrası norm koruma                    | Şekli bozdu                                                                                                                                                                                                               |
| `t_norm` normalizasyonunu sona alma                     | **Etkisiz** — H hiç seçilmediği için çıktı değişmedi. 239→135 m düşüşü koordinat bayrağı değişikliğinden geldi                                                                                                            |
| Yön hatası ~ GT dönüş hızı korelasyonu (`turncheck.py`) | Korelasyon **-0.008** (yok). Düşük turn-rate grubunda bile medyan hata 31°, yüksek grupta 27° — hata dönüş anlarına özgü değil. Düşük turn-rate'te bile hata zaten yüksek → sistematik/sürekli bir şey, olaya-bağlı değil |
| Kör pencereli R yumuşatma (`windowcheck.py`)             | Pencere=7'ye kadar Sim(3)/yön hatası iyileşir ama HIZALANMAMIS metrik baştan itibaren kötüleşir; otokorelasyon büyür (0.38→0.70). Gerçek dönüşleri bulanıklaştırıyor.                                                    |
| `R_H` eşiğine (0.45) yakınlık ~ yön hatası (`rhcheck.py`)| Korelasyon **-0.15** (yok). Eşiğe yakın kareler paradoksal olarak daha fazla eşleşen noktaya sahip; mekanizma bulunamadı.                                                                                                 |

## Dokümanlar

`docs/` altında: `el_kitabi.docx` (kavramlar), `kod_notlari.md` (dosya bazlı
analiz), `proje_durumu.docx` (durum), yarışma şartnamesi. Gerekince oku,
hepsini her oturumda yükleme.
