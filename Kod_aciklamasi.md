# PERI Kod Akis Aciklamasi

Bu dokuman, `peri_V1` klasorundeki mevcut kodun nasil calistigini bastan sona anlatir. Buradaki amac "bu dosya ne ise yariyor" demek degil; tam tersine, verinin nereden gelip nereye gittigini, hangi fonksiyonun neyi alip neye cevirdigini, neden o anda o cevrimin yapildigini adim adim gostermektir.

Bu metni bir hikaye gibi oku:

1. Sen terminalde egitim komutunu veriyorsun.
2. Kod once butun ayarlari bir `TrainingConfig` nesnesine topluyor.
3. Sonra dataset kuruluyor.
4. CSV satiri bir `EMOTICRecord` nesnesine donusuyor.
5. Bu kayittan `full_image`, `person_crop`, `emotion`, `vad`, `meta` gibi alanlari olan bir `sample` cikiyor.
6. PAS kullaniliyorsa bu `sample` sonradan bir wrapper ile genisletiliyor.
7. DataLoader bu tek tek sample'lari batch haline getiriyor.
8. Model batch'i alip iki kola ayiriyor: context kolu ve body kolu.
9. Body koluna PAS bilgisi `Cont-In` bloklari ile enjekte ediliyor.
10. Son kisimda iki akistan gelen ozellikler birlestiriliyor ve tahmin uretiliyor.
11. Loss hesaplanip geriye dogru gradyan akitiliyor.
12. Epoch sonunda mAP ve VAD hatasi hesaplanip dosyalara yaziliyor.

Asagida bu akisi ayrintili sekilde acacagim.

## 1. En Dis Cember: Komut Verdigin Anda Ne Oluyor?

Giris noktasi `peri_V1/scripts/train.py`.

Sen su tip bir komut veriyorsun:

```text
python .\peri_V1\scripts\train.py --root .\emotic --mode paper_faithful ...
```

Bu komutun yaptigi ilk sey, terminalden gelen her parametreyi `argparse` ile okuyup `args` adli bir degiskende toplamak.

Orada bir kritik parca var:

- `--cont-in-stages layer1,layer2,layer3` gibi bir yazi gelir.
- `_parse_cont_in_stages()` bunu ayirir.
- Sonuc `("layer1", "layer2", "layer3")` gibi bir tuple olur.

Yani terminalde senin tek satirda yazdigin bilgi, Python tarafinda artik programin kullanabilecegi temiz bir veri yapisina donusur.

Sonra `train.py` bu parametrelerden bir `TrainingConfig` olusturur. Bu cok onemli, cunku programin geri kalaninin neredeyse tamami artik tek tek `args.xxx` ile degil, bu config nesnesi ile calisir. Yani komuttan gelen ham ayarlar, duzenli ve dogrulanmis tek bir kutuda toplanmis olur.

Ardindan su olur:

1. `trainer = Trainer(config)` olusturulur.
2. `trainer.fit()` cagrilir.
3. Gercek egitim akisi bu noktadan sonra `Trainer` sinifinin icinde ilerler.

Kisaca:

```text
Terminal komutu
-> argparse
-> TrainingConfig
-> Trainer
-> fit()
```

Bu zincir, sistemin en dis kabugudur.

## 2. TrainingConfig: Komuttan Gelen Bilgi Nasil Kesin Kurala Donusuyor?

`peri_V1/peri/training/config.py` icindeki `TrainingConfig`, bu projenin merkezi kontrol masasi gibidir.

Burada iki sey olur:

1. Parametreler saklanir.
2. Parametrelerin mantikli olup olmadigi kontrol edilir.

### 2.1. Parametreler sadece depolanmiyor, duzeltiliyor

Mesela `data_root`, `output_root`, `resume_from`, `precomputed_pas_root`, `npy_manifest_root` gibi path alanlari `Path(...).resolve()` ile mutlak yola cevrilir. Bunun anlami su:

- Program "bu yol neredeydi?" diye sonradan kararsiz kalmaz.
- Her sey tam yol olarak tutulur.

### 2.2. Burasi sadece ayar kutusu degil, bir guvenlik kapisi

`__post_init__()` icinde cok sayida kontrol yapiliyor:

- `batch_size` ve `epochs` sifirdan buyuk olmali
- `num_workers` negatif olamaz
- `scheduler_name` izin verilenlerden biri olmali
- `pas_sigma > 0` olmali
- `pas_rho` varsa `0 ile 1` arasinda olmali

Yani config, "ne gelirse kabul edeyim" diyen pasif bir yapi degil. Tam tersine, hata cikabilecek seyleri daha en basinda durduruyor.

### 2.3. `paper_faithful` modu neyi kitliyor?

Bu modda config, seni makaleye yakin tutmak icin ekstra sinir koyuyor:

- backend `npy` olmak zorunda
- `pretrained=True` olmak zorunda
- context `224`, body `128` olmak zorunda
- PAS fusion sadece `cont_in` veya `none` olabilir
- `cont_in_variant` kesinlikle `paper` olmali
- `cont_in_stages` kesinlikle `("layer1", "layer2", "layer3")` olmali
- emotion loss `dynamic_mse` olmali
- label smoothing kapali olmali

Yani kod sunu diyor:

```text
Makaledeki yola yakin kalmak istiyorsan, belli serbestlikleri kapatiyorum.
```

### 2.4. `uses_pas` neden kucuk ama onemli?

`uses_pas` ozelligi su sorunun kisa cevabidir:

```text
Bu kosuda PAS yolu aktif mi?
```

Eger `pas_fusion_mode != "none"` ise PAS kullaniliyordur. Bu tek satirlik karar, dataset wrapper kurulup kurulmayacagini, landmark extractor acilip acilmayacagini ve modele `pas_image` gidip gitmeyecegini belirler.

## 3. Trainer Nesnesi Kurulunca Neler Hazirlaniyor?

`peri_V1/peri/training/trainer.py` icindeki `Trainer`, tum sistemin saha sorumlusudur. Dataset kurdurur, modeli kurar, loss'u baglar, optimizer'i acar, epoch dongusunu yonetir, dosyalari yazar.

`Trainer(config)` cagrilinca daha egitim baslamadan su seyler hazirlaniyor:

1. `prepare_run_artifacts(config)` ile run klasorleri aciliyor.
2. `write_run_config(...)` ile config JSON olarak kaydediliyor.
3. `set_global_seed(...)` ile rastgelelik sabitleniyor.
4. `self.device` seciliyor.
5. `PERIModel(...)` kuruluyor.
6. `build_loss_module(...)` ile loss modulu kuruluyor.
7. `AdamW` optimizer kuruluyor.
8. Scheduler hazirlaniyor.
9. CUDA kullaniliyorsa AMP scaler kuruluyor.

Yani `fit()` daha cagrilmadan once, "egitime hazir bir makine" olusturuluyor.

## 4. Run Klasoru Nasil Olusuyor?

`peri_V1/peri/training/logging.py` icindeki `prepare_run_artifacts()` su klasor yapisini kuruyor:

```text
outputs/runs/<mode>/<experiment_name>/<run_name>/
```

Bu klasorun icine de su dosyalar gelecektir:

- `run_config.json`
- `dataset_summary.json`
- `training_history.json`
- `final_metrics.json`
- `summary.json`
- `checkpoints/best.pt`
- `checkpoints/last.pt`
- `plots/loss_curve.png`
- `plots/map_curve.png`
- `plots/vad_curve.png`
- `plots/per_class_ap.png`
- `plots/lr_curve.png`
- `tensorboard/`

Yani egitim sadece ekranda kaybolan loglar uretmez. Her sey duzenli bir klasor altina kalici olarak yazilir.

## 5. Ham Veri Nereden Geliyor?

Veri tarafi `peri_V1/peri/data` altinda baslar. En temel sabitler `emotic_constants.py` icinde tutulur:

- 26 duygu sinifi `EMOTION_COLUMNS`
- 3 VAD boyutu `VAD_COLUMNS`
- bbox kolonlari
- split dosya isimleri
- paper boyutlari:
  - context `224`
  - body `128`

Yani tum sistem boyunca herkes "duygu kolonlari hangileri", "body kac piksel", "official split CSV'nin adi ne" gibi bilgileri ayni yerden alir.

## 6. Dataset Kurulumu: Ilk Gercek Veri Nesnesi Nasil Doguyor?

`peri_V1/peri/data/factory.py` icindeki `create_emotic_dataset_from_config(config, split=...)` ilk gercek dataset'i kurar.

Burada su kararlar alinir:

1. `train`, `val`, `test` icin ayri dataset nesneleri uretilir.
2. Eger precomputed PAS kullaniyorsan, augmentation base dataset'te kapatilir.
3. Eger runtime PAS ureteceksen, base dataset `include_pas_source=True` ile ham goruntuleri de sample'a ekler.

Bu cok onemli bir tasarim karari:

- Runtime PAS yolunda, once goruntu augment edilir, sonra landmark ve PAS uretilir.
- Precomputed PAS yolunda, PAS zaten diskten hazir gelecegi icin augmentation daha sonra wrapper icinde birlikte uygulanir.

## 7. CSV Satiri Nasil `EMOTICRecord` Oluyor?

`peri_V1/peri/data/emotic_dataset.py` icindeki `_parse_npy_records()` veya `_parse_npy_records_from_manifest()` burada devreye girer.

Bu asamada henuz image tensor'u yok. Henuz sadece kayit var.

Bir CSV satirindan su bilgiler cekilir:

- `Filename`
- `Arr_name`
- `Crop_name`
- `Width`, `Height`
- `Age`, `Gender`
- `X_min, Y_min, X_max, Y_max`
- 26 duygunun hedef degeri
- `Valence`, `Arousal`, `Dominance`

Bunlar bir `EMOTICRecord` nesnesine donusturulur. Record bir tensor batch'i degildir. Record, bir ornegin kimligi ve etiketiyle birlikte kayda alinmis ham tarifidir.

## 8. `EMOTICDataset.__getitem__`: Kayit Artik Gercek Sample'a Donusuyor

Bir DataLoader dataset'ten index istediginde `EMOTICDataset.__getitem__(index)` cagrilir.

Akis su:

1. `record = self.records[index]`
2. bbox onarilir
3. full image ve person crop diskten okunur
4. gerekiyorsa augmentation uygulanir
5. full image `224x224`, crop `128x128` olacak sekilde resize edilir
6. bunlar tensor'a cevrilir
7. etiketler ve metadata ile birlikte bir `sample` dict'i olusturulur

### 8.1. BBox neden onariliyor?

`repair_bbox()` bbox'i sinirlar icine cekmek icin kullanilir. Sebep su:

- CSV'deki bbox bazen bozuk olabilir
- negatif olabilir
- goruntu sinirinin disina tasabilir

Eger bunu onarmazsan crop alma asamasinda bos veya hatali goruntu uretebilirsin.

### 8.2. Goruntuler nasil yukleniyor?

NPY backend icin:

- `full_image = _load_npy_image(full_path)`
- `person_crop = _load_npy_image(crop_path)`

Sonra `_resize_image()` ile boyutlar sabitlenir:

- context: `224x224`
- body: `128x128`

Sonra `_to_tensor()` ile:

- HWC -> CHW olur
- `uint8 [0,255]` -> `float [0,1]` olur

### 8.3. Bu sample dict'inde tam olarak ne var?

Temel sample su alanlari tasir:

- `full_image`
- `person_crop`
- `emotion`
- `vad`
- `bbox`
- `bbox_original`
- `meta`

`meta` icinde de kimlik ve takip bilgileri vardir:

- `sample_id`
- `split`
- `source_split`
- `filename`
- `full_image_name`
- `crop_name`
- `width`
- `height`
- `age`
- `gender`
- `bbox_original`
- `bbox_repaired`
- `bbox_notes`
- `augment_applied`

Eger runtime PAS kullanilacaksa iki sey daha eklenir:

- `pas_source_full_image`
- `pas_source_person_crop`

Yani bu noktada sample hazirdir ama `pas_image` henuz yoktur.

## 9. Base Dataset Bittigi Anda Elimizde Ne Var?

Artik elimizde, tek bir insan ornegini temsil eden bir `sample` vardir.

PAS kullanilmiyorsa bu sample neredeyse dogrudan DataLoader'a gidecektir.

PAS kullaniliyorsa henuz is bitmedi. Cunku sample'da `pas_image` yoktur. O ancak bir sonraki wrapper asamasinda eklenecektir.

## 10. PAS Yolu Nerede Devreye Giriyor?

Bu karar `peri_V1/peri/training/dataloaders.py` icindeki `build_dataloaders(config)` fonksiyonunda verilir.

Fonksiyon once her zaman su uc dataset'i kurar:

- train_dataset
- val_dataset
- test_dataset

Sonra `config.uses_pas` kontrol edilir.

### 10.1. Eger PAS kapaliysa

Dataset hic wrapper'a sarilmaz:

```text
EMOTICDataset
-> DataLoader
-> Model
```

### 10.2. Eger PAS aciksa

O zaman base dataset daha sonra `EMOTICPreprocessedDataset` ile sarilir.

Bu wrapper'in gorevi sunlardir:

- landmark bulmak
- PAS uretmek veya diskten PAS okumak
- sample'a `pas_image` ve `pas_mask` eklemek
- gerekiyorsa PAS ile uyumlu augmentation uygulamak

## 11. Runtime PAS ve Precomputed PAS Arasindaki Yol Ayrimi

Burasi en kritik kisimlardan biri. Cunku sistemin davranisi burada ikiye ayriliyor.

### 11.1. Runtime PAS yolu

Eger `precomputed_pas_root is None` ise:

1. `LandmarkExtractor(...)` olusturulur
2. `PASGenerator(...)` olusturulur
3. dataset wrapper'a bunlar verilir

Bu yolun mantigi su:

- Ornek gelir
- O ornegin crop'undan landmark cikarilir
- Landmark'lardan PAS maskesi o anda uretilir

### 11.2. Precomputed PAS yolu

Eger `precomputed_pas_root` doluysa:

1. train/val/test icin CSV index map'leri okunur
2. wrapper, sample'a bakip hangi `pas_xxxxxx.png` dosyasini acacagini bilir
3. PAS diskten hazir goruntu olarak cekilir

Yani ayni `pas_image` degiskeni iki farkli kaynaktan gelebilir:

- ya MediaPipe + Gaussian uretimi ile
- ya da diskten PNG okuyarak

## 12. `EMOTICPreprocessedDataset`: Base Sample'a PAS Ekleyen Katman

`peri_V1/peri/preprocess/landmarks.py` icindeki `EMOTICPreprocessedDataset`, base dataset'in ustune binen ikinci katmandir.

Bu wrapper'in `__getitem__` akisi su:

1. once `self.base_dataset[index]` ile base sample alinir
2. bu sample `augment_sample_with_landmarks_and_pas(...)` fonksiyonuna verilir
3. fonksiyon sample'a `landmarks`, `pas_mask`, `pas_image` ekler
4. eger wrapper seviyesinde augmentation aciksa, ayni transform full image, body crop ve maskeye birlikte uygulanir

Yani wrapper diyorki:

```text
Base dataset bana goruntuyu ve etiketi verdi.
Ben de bunun ustune PAS tarafini insa ettim.
```

## 13. Runtime PAS Yolunda Sample Nasil Genisliyor?

`augment_sample_with_landmarks_and_pas(...)` runtime yolunda su sekilde ilerler.

### 13.1. Landmark cache varsa once oraya bakar

Eger `landmark_cache_dir` verilmis ise:

- `sample_id` ile cache dosyasi aranir
- varsa landmark oradan okunur

Bu, MediaPipe'in her sample icin tekrar tekrar calismasini azaltir.

### 13.2. Cache yoksa extractor devreye girer

`LandmarkExtractor.extract(...)` cagrilir.

Bu fonksiyonun girdileri:

- `person_crop`
- bazen `full_image`
- `bbox`
- `meta`

Buradaki ana niyet su:

- once kisi crop'u uzerinde landmark bul
- gerekiyorsa baska yontemle tamamlamaya calis

### 13.3. LandmarkExtractor tam olarak ne yapiyor?

Bu sinif MediaPipe tarafini soyutlar.

`extract()` icinde su olur:

1. `person_crop` NumPy HWC RGB formatina cevrilir
2. once crop uzerinde detect denenir
3. `prefer_holistic=True` ise holistic modelden hem pose hem face landmark alinmaya calisilir
4. biri cikmazsa pose-only ve face-only modelleri fallback olarak denenir
5. experimental modda gerekirse full image bbox crop fallback'i kullanilabilir

Sonuc olarak eline su yapida bir dict gelir:

- `landmarks["pose"]`
- `landmarks["face"]`

Her biri su bilgileri tasir:

- `keypoints`
- `detected`
- `count`
- `source_image`
- `message`

Yani sadece "landmark var" demiyor. Nereden bulundu, bulundu mu, kac nokta var, hepsini tasiyor.

### 13.4. Landmark'lar PAS'a nasil donusuyor?

Burada `PASGenerator.generate(image, landmarks)` devreye girer.

Bu fonksiyonun girdileri:

- body crop goruntusu
- pose ve face landmark dict'leri

Icindeki mantik su:

1. bos bir `response` haritasi olusturulur
2. her pose noktasi icin merkeze Gaussian yerlestirilir
3. her face noktasi icin de aynisi yapilir
4. bunlar toplanmaz, piksel bazinda `maximum` ile birlestirilir
5. ortaya tek kanalli bir dikkat haritasi cikar
6. bu harita `rho` esigi ile maske haline getirilir
7. body crop bu maskeyle carpilarak `pas_image` uretilir

Yani `pas_image`, sifirdan uretilmis yeni bir fotograf degildir. Aslinda mevcut body crop'un, sadece landmark cevresindeki bolgeleri gorunur birakilmis halidir.

### 13.5. Boyut niye tekrar ayarlaniyor?

PAS body stream'e girecegi icin `person_crop` ile ayni H ve W'ye sahip olmali.

Bu yuzden:

- `resize_mask(...)`
- `resize_rgb_image(...)`

ile hedef boyuta getirilir. Sonra tensor'a cevrilir:

- `pas_mask`: `[1, H, W]`
- `pas_image`: `[3, H, W]`

Ve sample'a eklenir.

## 14. Precomputed PAS Yolunda Sample Nasil Genisliyor?

Ayni fonksiyon, bu sefer baska bir kola girer.

Akis su:

1. sample'in `filename`, `split`, `bbox` bilgileri `meta` icinden cekilir
2. bunlarla bir lookup key olusturulur
3. bu key ile index map'ten hangi PAS dosyasinin acilacagi bulunur
4. `pas_<split>/000123.png` gibi bir dosya acilir
5. bu PNG `pas_image` olur
6. RGB toplamindan sifir olmayan pikseller bulunur ve `pas_mask` uretilir

Burada landmark gercekten hesaplanmiyor. O yuzden sample'a bos landmark dict'leri yaziliyor.

Yani model acisindan elinde yine `pas_image` var, ama bu goruntunun kaynagi artik MediaPipe degil disk.

## 15. Augmentation Neden Iki Farkli Yerde Uygulaniyor?

Bu kisim ilk bakista garip gelir ama aslinda cok mantiklidir.

### 15.1. Runtime PAS'ta augmentation daha once

Runtime PAS'ta base dataset `augment=True` olabilir.

Yani:

1. goruntu once augment edilir
2. sonra landmark cikarilir
3. sonra PAS uretilir

Bu mantiklidir, cunku PAS zaten o yeni goruntuye gore sifirdan hesaplanir.

### 15.2. Precomputed PAS'ta augmentation daha sonra

Precomputed PAS'ta PAS onceden hazirdir.

Eger sen body crop'u ayri, PAS goruntusunu ayri sekilde augment edersen bunlar hizasini kaybeder.

Bu yuzden kod sunu yapiyor:

1. base dataset'te augmentation kapali
2. wrapper sample'i PAS ile birlikte olusturuyor
3. sonra tek bir rastgele transform uretiliyor
4. ayni geometri hem `full_image`, hem `person_crop`, hem `pas_mask` uzerine uygulanuyor
5. en sonda `pas_image = person_crop * pas_mask` diye yeniden kuruluyor

Yani sistem diyor ki:

```text
Maskeyi ve kisiyi birlikte hareket ettirmezsem, PAS yanlis yere kayar.
```

## 16. DataLoader Bu Tek Tek Sample'lari Batch'e Nasil Ceviriyor?

`collate_emotic_batch()` her sample'i alip ayni anahtarlara gore batch yapar.

Ornek:

- 32 sample geldiyse
- her birinde `full_image` `[3,224,224]` ise
- batch sonunda `full_image` `[32,3,224,224]` olur

Ayni sey su alanlar icin de olur:

- `person_crop` -> `[B,3,128,128]`
- `pas_image` -> `[B,3,128,128]`
- `pas_mask` -> `[B,1,128,128]`
- `emotion` -> `[B,26]`
- `vad` -> `[B,3]`

String veya liste gibi alanlar ise stack edilmez; liste olarak kalir. Mesela `meta["filename"]` batch'te bir liste olur.

## 17. Model Tarafina Gecis: Batch Artik `PERIModel`'e Giriyor

`peri_V1/peri/models/peri_model.py` icindeki `PERIModel`, bu batch'i alip tahmine donusturen ana agdir.

Bu modelin ana fikri su:

- bir kol tum sahneyi gorur: `full_image`
- bir kol sadece kisiyi gorur: `person_crop`
- PAS varsa, kisi kolunun ara asamalarina ek bilgi olarak girer

Yani model tek bir resimden degil, ayni ornegin iki farkli gorunusunden ogreniyor:

1. genel baglam
2. kisinin kendisi

## 18. Modelin Icine Girerken Ilk Ne Oluyor?

`forward()` cagrilinca ilk olarak batch'ten su alanlar cekilir:

- `full_image`
- `person_crop`
- `pas_image`

Sonra `_ensure_batched()` ile bu tensorlerin gercekten `[B,3,H,W]` formatinda oldugu garanti edilir.

Bu neden onemli?

Cunku dataset'ten bazen tek sample gelebilir, bazen batch gelebilir. Modelin ic mantigi ise hep "batch varmis" gibi calismak ister.

## 19. Normalizasyon: Hangi Goruntu Nasil Hazirlaniyor?

Model icinde iki farkli goruntu turu farkli sekilde ele aliniyor.

### 19.1. `full_image` ve `person_crop`

Bunlar ImageNet mean/std ile normalize ediliyor:

```text
(image - mean) / std
```

Bunun sebebi su:

Context ve body backbonelari ResNet-18 ve ImageNet pretrained olarak geliyor. Bu agirliklar, goruntuyu belli bir dagilimda bekliyor.

### 19.2. `pas_image`

PAS ise ayni normalizasyondan gecmiyor.

Sebep su:

```text
PAS burada ImageNet on egitimli tam bir ucuncu backbone'a girmiyor.
Cont-In bloklarina yardimci sinyal olarak gidiyor.
```

Bu yuzden PAS kendi `[0,1]` araliginda kalir.

## 20. ResNet-18 Backbone Nasil Davraniyor?

`peri_V1/peri/models/backbones.py` icindeki `ResNet18Backbone`, torchvision kullanmadan yazilmis bir ResNet-18'dir.

Bir `BasicBlock` mantigi su:

1. giris gelir
2. conv-bn-relu
3. bir conv-bn daha
4. orijinal giris residual olarak eklenir
5. en sonda tekrar relu uygulanir

Backbone'un genel akisi:

1. `conv1`
2. `bn1`
3. `relu`
4. `maxpool`
5. `layer1`
6. `layer2`
7. `layer3`
8. `layer4`
9. `avgpool`
10. flatten

## 21. Context Kolu Tam Olarak Ne Uretiyor?

Context kolu `full_image` tensorunu alir.

Giris sekli:

```text
[B, 3, 224, 224]
```

Bu ResNet'ten gecince son pooled ozellik:

```text
[B, 512]
```

olur.

Bu vektor, sahnenin genel bilgisini tasir:

- cevre
- mekan
- arka plan
- diger insanlarin varligi
- olay ortaminin genel yapisi

Yani "bu kisi nerede, nasil bir sahnede" sorusunun yogunlastirilmis cevabidir.

## 22. Body Kolu Tam Olarak Ne Uretiyor?

Body kolu `person_crop` tensorunu alir.

Giris sekli:

```text
[B, 3, 128, 128]
```

Body kolu farkli bir yoldan gider cunku araya Cont-In bloklari sokulmustur.

`_forward_body_stream()` icindeki akis su:

1. `conv1 -> bn1 -> relu -> maxpool`
2. `layer1`
3. eger `layer1` icin Cont-In varsa PAS ile modulle
4. `layer2`
5. eger `layer2` icin Cont-In varsa PAS ile modulle
6. `layer3`
7. eger `layer3` icin Cont-In varsa PAS ile modulle
8. `layer4`
9. `avgpool`

Yani body kolu PAS'i en basta bir kez alip bitirmiyor. Ara seviyelerde tekrar tekrar devreye sokuyor.

## 23. `fusion.py` Merkezinde Ne Var?

`peri_V1/peri/models/fusion.py` icinde dort ana yapi var:

1. `FusionHead`
2. `PaperPASStageEncoder`
3. `ContInBlock`
4. `LatePASFusion`

Paper-faithful kosularda asil kritik olanlar ilk ucudur.

## 24. `FusionHead`: En Sondaki Karar Mekanizmasi

Bu sinifin girdisi bir ozellik vektorudur.

Paper path'te tipik olarak bu vektor sunlarin birlesimidir:

- context pooled feature: `[B,512]`
- body pooled feature: `[B,512]`

Birlesince:

```text
[B, 1024]
```

`FusionHead` icinde su olur:

1. `Linear(in_dim -> hidden_dim)`
2. `ReLU`
3. `Dropout(0.3)`

Buradan cikan ortak temsil `fused` olur.

Sonra iki ayri kafa vardir:

- `emotion_head`: `Linear -> Sigmoid`
- `vad_head`: tek `Linear`

Yani ayni ortak ozellikten iki farkli cikis uretilir:

- 26 duygunun olasiligi
- 3 VAD degeri

`Sigmoid` kullanilma nedeni, gorevin multi-label olmasidir. Ayni kiside birden fazla duygu etiketi ayni anda aktif olabilir.

## 25. `PaperPASStageEncoder`: Ayni PAS Goruntusunun Her Stage Icin Ayri Kodlanmasi

Bu sinifin varlik nedeni su sorudur:

```text
PAS goruntusunu layer1, layer2, layer3 gibi farkli seviyelere nasil uyduracagim?
```

Body backbone'daki stage cikislari ayni boyutta degildir.

Body girisi `128x128` ise yaklasik olarak:

- `layer1` cikisi: `[B, 64, 32, 32]`
- `layer2` cikisi: `[B, 128, 16, 16]`
- `layer3` cikisi: `[B, 256, 8, 8]`
- `layer4` cikisi: `[B, 512, 4, 4]`

Ama elindeki `pas_image` hala:

```text
[B, 3, 128, 128]
```

Dogrudan bunu `layer2` ile birlestiremezsin. Kanal sayisi ve uzaysal boyutlar uyusmaz.

`PaperPASStageEncoder` tam bunu cozer.

### 25.1. `layer1` icin ne yapar?

Ilk blok:

1. `Conv7x7 stride=2`
2. `BN`
3. `ReLU`
4. `MaxPool`

Bu, 128x128 PAS goruntusunu yaklasik 32x32 boyuta indirir ve 64 kanal uretir. Yani `layer1` seviyesine uygun bir PAS ozelligi cikarir.

### 25.2. `layer2` icin ne yapar?

Yukaridaki ilk bloktan sonra bir stride-2 conv daha ekler.

Boyut ve kanal artik `layer2` ile daha uyumlu olur:

- uzaysal boyut kuculur
- kanal sayisi `128` olur

### 25.3. `layer3` icin ne yapar?

Bir stride-2 adim daha eklenir, kanal `256` olur.

Yani ayni `pas_image`, `layer1` icin baska, `layer2` icin baska, `layer3` icin baska sekilde encode edilir.

Buradaki ince fikir su:

```text
PAS'i bir kez encode edip her yere dagitmak yerine,
her stage kendi derinligine uygun PAS temsilini uretsin.
```

## 26. `ContInBlock`: PAS Bilgisinin Body Ozelligine Enjekte Edildigi Yer

Bu blok iki seyi girdi olarak alir:

1. `body_features`
2. `pas_signal`

Ornek olarak `layer2` asamasini dusunelim.

O anda:

- `body_features` sekli yaklasik `[B,128,16,16]`
- `pas_signal` sekli `[B,3,128,128]`

### 26.1. Ilk adim: PAS'i uygun stage temsilina cevir

`pas_features = self.pas_encoder(pas_signal)`

Eger bu blok `layer2` icinse, encoder PAS'tan `[B,128,16,16]` civari bir temsil cikarir. Yani artik PAS de body ile ayni seviyede konusmaya baslar.

### 26.2. Gerekirse boyutlari birebir esitle

Kod, uzaysal boyutlar tam eslesmezse `F.interpolate(...)` ile PAS feature'i body feature boyutuna getirir.

### 26.3. Sonra ikisini kanal ekseninde birlestir

`torch.cat([body_features, pas_features], dim=1)`

Mesela:

```text
[B,128,16,16] + [B,128,16,16]
-> [B,256,16,16]
```

Artik tek bir tensor icinde hem beden feature'i hem de PAS feature'i yanyana duruyor.

### 26.4. Bu birlesik tensoru modulle et

`self.modulation` icinde iki conv blok var.

Bu kisim sunu ogreniyor:

```text
Body feature'i PAS bilgisine bakarak nasil yeniden yazmaliyim?
```

Yani bu sadece "birlestir ve gec" degil. Model burada yeni bir ara temsil ogreniyor.

### 26.5. Paper varyantinda sonuc ne oluyor?

Paper varyantinda `return update` yapiliyor.

Demek ki blok su mantikla calisiyor:

```text
Eski body feature'i al
PAS ile birlikte yeniden isle
ve ortaya cikani yeni stage cikisi olarak kullan
```

Yani burada PAS, body ozelligine disaridan hafifce dokunan kucuk bir eklenti degil. Tam tersine, stage cikisinin kendisini yeniden bicimlendiren aktif bir belirleyici.

## 27. Body Kolu Icindeki Cont-In Akisini Gozunde Canlandiralim

Tek bir sample icin bunu cok somut dusunelim.

Elinde bir kisinin crop'u var:

- yuzu hafif egik
- omuzlari gorunuyor
- elleri gorunmuyor

Bir de PAS maskesi var:

- yuz cevresi aktif
- omuzlar civari aktif
- geri kalan cogunlukla sifir

Akis su sekildedir:

1. Body crop ResNet'e girer.
2. `layer1` ilk orta seviye beden ozelliklerini cikarir.
3. Ayni anda `pas_image`, `layer1`e uygun boyutta encode edilir.
4. Ikisi yanyana konur.
5. Cont-In "beden ozelligini, PAS'in isaret ettigi bolgelere daha duyarli hale getirilmis yeni bir bedene" cevirir.
6. Bu yeni tensor bir sonraki stage'e gider.
7. `layer2` ve `layer3`te ayni fikir yeniden uygulanir.

Yani PAS bir kere "bak buraya" deyip cekilmez. Uc farkli seviyede tekrar tekrar bedene mudahale eder.

## 28. `late` Fusion Ile `cont_in` Arasindaki Fark Ne?

Kodda `LatePASFusion` da var ama paper yolunda esas hedef bu degil.

`late` modda mantik su olurdu:

- PAS ayri ufak bir encoder'dan gecer
- en sonda bir vektor uretilir
- bu vektor context ve body pooled vektorlerine eklenir

Yani PAS sadece final karar oncesi masaya gelir.

`cont_in` yolunda ise PAS ara katmanlara girer.

Fark sudur:

- `late`: PAS en son fikir beyan eder
- `cont_in`: PAS beden temsilinin nasil olusacagini yolda belirler

## 29. `resolve_feature_concat`: En Son Birlesim

Model son asamaya geldiginde artik elinde su vektorler vardir:

- context pooled
- body pooled
- varsa late PAS pooled

`resolve_feature_concat()` sadece bunlari belirtilen sirada birlestirir.

Paper-faithful `cont_in` yolunda tipik durum:

```text
context [B,512]
body    [B,512]
-> concat -> [B,1024]
```

Sonra bu `FusionHead`e gider.

## 30. Modelin `forward()` Cikisi Tam Olarak Nedir?

Model su alanlari dondurur:

- `emotion_probs`
- `vad`
- `features`

`features` icinde de genelde:

- `context`
- `body`
- `fused`

vardir.

## 31. Loss Kismi: Tahmin Etmek Yetmez, Hata da Hesaplanmali

`peri_V1/peri/training/losses.py` icindeki `MultiTaskLoss` burada devreye girer.

Bu loss iki parcadan olusur:

1. emotion loss
2. vad loss

Ve sonra bunlar toplanir.

## 32. Emotion Loss Neden Siradan MSE Degil?

`DynamicWeightedMSELoss` kullaniliyor.

Mantigi su:

Her duygu sinifi batch icinde ayni siklikta gelmez. Bazi duygular cok az gorunur.

Kod once batch icindeki hedefleri inceler:

```text
class_probability = target.mean(dim=0)
```

Yani her duygunun bu batch'te ortalama ne kadar aktif olduguna bakar.

Sonra bunun uzerinden agirlik hesaplar:

```text
1 / log(p + c)
```

Bunun anlami su:

- batch'te az gorulen siniflar daha fazla agirlik alir
- cok gorulenler daha az agirlik alir

Sonra tahmin ile hedef arasindaki kare hata bu agirlikla carpilarak toplanir.

Yani model, nadir siniflari da ciddiye almak zorunda kalir.

## 33. VAD Loss Nasil Hesaplaniyor?

VAD tarafinda mantik daha basittir:

```text
abs(pred - target).mean()
```

Yani:

- valence hatasi
- arousal hatasi
- dominance hatasi

birlikte ortalanir.

Sonra toplam loss su olur:

```text
total_loss = emotion_loss + vad_weight * vad_loss
```

Varsayilan olarak `vad_weight = 0.5`.

## 34. Bir Batch Egitimde Nasil Isleniyor?

`Trainer._run_epoch()` icindeki bir batch akisini tek tek acalim.

1. DataLoader bir batch verir.
2. `_move_batch_to_device()` tensorleri CPU'dan GPU'ya tasir.
3. `_validate_batch()` NaN/Inf var mi bakar.
4. autocast gerekiyorsa acilir.
5. `outputs = self.model(batch)` cagrilir.
6. `losses = self.loss_module(outputs, targets)` hesaplanir.
7. training ise backward yapilir.
8. optimizer step atar.
9. gerekiyorsa scheduler step atar.
10. metric accumulator bu batch'in cikislarini toplar.

Yani her batch'te model sadece tahmin yapmiyor; ayni zamanda kendi hatasina bakip agirliklarini guncelliyor.

## 35. Backward Sirasinda Tam Olarak Ne Ogreniliyor?

Batch sonunda model sunu goruyor:

- Hangi duygulari fazla verdi
- Hangi duygulari eksik verdi
- VAD tarafinda ne kadar sasti

Bu hatalar gradyan olarak geriye akar.

Bu akisin etkiledigi yerler:

- `FusionHead`
- `ContInBlock` conv katmanlari
- `PaperPASStageEncoder`
- body backbone
- context backbone

Yani sadece en sondaki linear katmanlar degil, PAS'i ara seviyede kullanan kisimlar da ogrenme sinyali alir.

## 36. Metric Hesabi Ne Zaman ve Nasil Yapiliyor?

`peri_V1/peri/analysis/metrics.py` icindeki `BatchMetricAccumulator`, batch batch gelen tahminleri epoch boyunca biriktirir.

Epoch bitince:

- tum emotion olasiliklari birlestirilir
- tum emotion hedefleri birlestirilir
- tum VAD tahminleri birlestirilir
- tum VAD hedefleri birlestirilir

Sonra iki ana metric grubu hesaplanir:

1. multi-label emotion metric'leri
2. VAD metric'leri

### 36.1. Multi-label kisim

Burada:

- `precision`
- `recall`
- `f1`
- `map`

hesaplanir.

`map` icin her sinifta AP ayri hesaplanir, sonra ortalamasi alinir.

Mantik su:

1. Bir duygunun skorlarini buyukten kucuge sirala
2. Gercek pozitifler yukarida toplanabiliyor mu bak
3. Precision-recall alanini hesapla
4. Her duygu icin bunu ayri yap
5. Ortalamalarini al

### 36.2. VAD kisim

Her boyut icin ayri L1 hata:

- `vad_valence_l1`
- `vad_arousal_l1`
- `vad_dominance_l1`

ve bunlarin ortalamasi:

- `vad_error`

hesaplanir.

## 37. Bir Epoch Bitince Trainer Ne Yapiyor?

`fit()` icinde her epoch su sirayla olur:

1. train epoch kosar
2. val epoch kosar
3. validation mAP'e bakilir
4. eger en iyi sonucsa `best.pt` guncellenir
5. her durumda `last.pt` guncellenir
6. epoch kaydi `history` listesine eklenir
7. `training_history.json` yazilir
8. TensorBoard scalar'lari yazilir
9. scheduler uygunsa ilerletilir

Yani her epoch sonunda sistem "sadece devam et" demiyor; ayni zamanda kayit tutuyor, en iyiyi ayiriyor ve tekrar baslatilabilir bir durum sakliyor.

## 38. `best.pt` Ile `last.pt` Arasindaki Fark

- `last.pt`: en son gorulen durum
- `best.pt`: validation metrigine gore en iyi durum

Bu cok onemli. Cunku bazen egitim devam ederken model son epoch'ta daha kotu olabilir. Ama en iyi validation sonucu daha once alinmistir. `best.pt` tam olarak onu saklar.

## 39. Egitim Sonunda Ne Yaziliyor?

Eger `evaluate_test_after_train=True` ise:

1. `best.pt` tekrar yuklenir
2. test split uzerinde calistirilir
3. test metric'leri hesaplanir
4. `final_metrics.json` icine yazilir

Ardindan plot'lar olusur:

- loss egrisi
- mAP egrisi
- VAD egrisi
- gerekirse per-class AP
- learning rate egrisi

Ve `summary.json` ile bu run'in nasil bittigi yazilir.

## 40. Bir Ornegin Tum Yolculugunu Tek Parca Halinde Dusunelim

Simdi tum anlattiklarimi tek bir ornek uzerinden birbirine baglayalim.

Diyelim `train` split'inden tek bir kisi geliyor.

### Adim 1

CSV satirinda bu kisinin:

- dosya adi
- crop dosyasi
- bbox'i
- 26 duygu etiketi
- 3 VAD etiketi

var.

Bu satir `EMOTICRecord` olur.

### Adim 2

`EMOTICDataset.__getitem__` cagrilir.

Bu kayittan:

- full image okunur
- person crop okunur
- bbox gerekirse onarilir
- goruntuler resize edilir
- tensor'a cevrilir

Ve bir `sample` dict'i cikar.

### Adim 3

Eger runtime PAS kullaniyorsan:

- crop'tan landmark cikar
- Gaussian'larla maske uret
- crop ile maskeyi carp
- `pas_image` olustur

Eger precomputed PAS kullaniyorsan:

- sample'a denk gelen `pas_xxxxxx.png` dosyasini ac
- onu `pas_image` yap

### Adim 4

DataLoader bunu diger sample'larla birlestirir.

Artik:

- `full_image` -> `[B,3,224,224]`
- `person_crop` -> `[B,3,128,128]`
- `pas_image` -> `[B,3,128,128]`
- `emotion` -> `[B,26]`
- `vad` -> `[B,3]`

olmus olur.

### Adim 5

Model `full_image`'i context backbone'a yollar.

Buradan `[B,512]` context ozelligi gelir.

### Adim 6

Model `person_crop`'u body backbone'a yollar.

Ama `layer1`, `layer2`, `layer3` sonlarinda `pas_image` her defasinda stage'e uygun bicimde encode edilir ve body ozelligi yeniden bicimlendirilir.

En sonda body'den de `[B,512]` vektor gelir.

### Adim 7

Bu iki vektor birlesir:

```text
[B,512] + [B,512] -> [B,1024]
```

### Adim 8

`FusionHead` bu vektorden:

- `emotion_probs [B,26]`
- `vad [B,3]`

uretir.

### Adim 9

Gercek etiketlerle karsilastirilir:

- emotion loss hesaplanir
- vad loss hesaplanir
- toplam loss elde edilir

### Adim 10

Backward olur. Yani model, "bu batch'te nerede yanildim" bilgisini alir ve agirliklarini ona gore biraz duzeltir.

### Adim 11

Epoch sonuna kadar tum batch'ler boyle akar. Sonunda mAP hesaplanir ve checkpoint kaydedilir.

## 41. `fusion.py`yi Bir Cumlede Ozetlersek

Bu dosyanin gorevi, PAS'i en sona eklenen ucuncu bir bilgi parcasi yapmak degil; body stream'in ara feature'larini PAS goruntusune bakarak yeniden sekillendirmektir.

Daha da basit soylersek:

```text
Context kolu "sahne ne diyor?" der.
Body kolu "kisi neye benziyor?" der.
Cont-In ise "kisi ozelligini, PAS'in isaret ettigi bolgeleri dikkate alarak yeniden yaz" der.
```

## 42. Bu Kodu Okurken Zihninde Tutman Gereken En Onemli 10 Gercek

1. `train.py` sadece kapidir; asil is `Trainer` icinde doner.
2. `TrainingConfig`, parametreleri sadece saklamaz, ayni zamanda sinirlar.
3. `EMOTICRecord`, henuz goruntu degil; kaydin tarifidir.
4. `EMOTICDataset.__getitem__`, kaydi gercek tensor sample'a donusturur.
5. PAS kullaniliyorsa sample sonradan bir wrapper ile genisler.
6. Runtime PAS ve precomputed PAS ayni `pas_image` hedefine iki farkli yoldan ulasir.
7. Precomputed yolda augmentation sonradan uygulanir cunku maskeyle crop'un birlikte hareket etmesi gerekir.
8. Modelde iki ana akim vardir: context ve body.
9. `ContInBlock`, PAS'i body akisinin ara feature'larina stage stage enjekte eder.
10. Egitim sonunda secilen "en iyi model", son epoch olmak zorunda degildir; bu yuzden `best.pt` vardir.

## 43. Son Kisa Ozet

Bu kod tabani, EMOTIC ornegini once duzenli bir `sample` dict'ine ceviriyor, sonra gerekiyorsa PAS ile zenginlestiriyor, sonra bunu iki kollu bir ResNet tabanli modele veriyor, body kolunda PAS'i `Cont-In` bloklariyla ara seviyede kullaniyor, en sonda duygu ve VAD tahmini uretiyor, loss hesapliyor, metrikleri topluyor ve run artefact'larini diske yaziyor.

Bu sistemin omurgasi tek cumlede soyle:

```text
CSV kaydi -> sample dict -> PAS ile zenginlestirme -> batch -> iki akimli model -> Cont-In ile body modulasyonu -> fusion head -> loss -> metric -> checkpoint
```

Bu sirayi kafanda netlestirdiginde, kodun tamamina hakim olmak cok daha kolaylasir; cunku artik dosyalari tek tek degil, ayni uretim hattinin farkli istasyonlari olarak gormeye baslarsin.
