# DarkIR-CSF: Düşük Işıklı Görüntü İyileştirme için Çapraz Ölçekli Birleştirme Deneyleri

Bu repo, DarkIR mimarisi üzerine hafif bir Cross-Scale Fusion (CSF) modülü ekleyerek düşük ışıkta bulanık/gürültülü görüntü iyileştirmeyi araştıran ön çalışma niteliğinde bir projedir. Amaç, az veri senaryosunda CSF bloğunun etkisini görmek ve karşılaştırmalı sonuçlar üretmek.

## Önemli Not (Veri ve Sonuçlar)
- Tüm deneyler LOLBlur veri setinin küçültülmüş alt kümesiyle yapıldı: **1620 train**, **600 val**, **540 test**.
- CSF modu validation’da PSNR/L1 tarafında iyileşme gösterirken, tamamen ayrılmış test setinde orijinal DarkIR biraz daha iyi genelleme yaptı.
- Sonuçlar keşif niteliğinde; nihai SoTA iddiası yoktur.

## Proje Özeti
- Orijinal DarkIR’i yeniden uyguladım, LOLBlur alt kümesiyle eğittim.
- Encoder–decoder skip bağlantılarına hafif bir CSF bloğu ekledim.
- DarkIR (baseline) vs. DarkIR + CSF: PSNR, SSIM, LPIPS, parametre ve FLOP maliyeti kıyaslandı.
- Amaç: az veriyle model tasarım/eğitim/değerlendirme pratiği ve portföy için bir demo proje.

## Dizin Yapısı (özet)
```
DarkIR/
  train.py, testing.py, inference.py, inference_video.py
  archs/, losses/, utils/
  options/
    train/ LOLBlur.yml, LOLBlur_csf.yml
    test/  LOLBlur.yml, LOLBlur_CSF.yml
  data/datasets/LOLBlur/ (train, train_val, test)   # veri buraya konur
  models/ DarkIR_lolblur_original_best.pt, DarkIR_lolblur_best.pt, ...
```
`options/train` ve `options/test` içindeki yml dosyaları yol ve hiperparametre referanslarını içerir.

## Veri Seti
- Veri: LOLBlur (düşük ışık + blur/gürültü giriş, temiz hedef).
- Bölünüm: train 1620, val 600, test 540 çift. Test seti yalnızca final değerlendirme için.
- Telif/boyut nedeniyle veri repo içinde yok; resmi kaynaktan indirip `data/datasets/LOLBlur/` altına yerleştirin.

## Yöntem
**DarkIR (baseline)**  
- Encoder–decoder, depth-wise + dilated konvolüsyonlar.  
- ~3.32M parametre, 7.25 GMac (@3x256x256).

**DarkIR + CSF**  
- Skip bağlantılarında encoder/decoder özelliklerini 1x1 konv MLP ile ağırlıklı birleştiriyor.  
- ~3.38M parametre, 7.67 GMac (@3x256x256); ek maliyet küçük.

## Eğitim Ayarları
- Patch: 256x256 random crop, batch 4.
- Optim: AdamW (lr 1e-4, weight_decay 1e-4, betas 0.9/0.99), CosineAnnealing, eta_min 1e-6.
- Grad clip: 5.0, loss: Charbonnier.
- Augment: yatay/dikey çevirme.
- 100 epoch, CUDA varsa mixed precision opsiyonel.
- Donanım: yerel GPU veya Colab T4; kısıtlı VRAM yüzünden veri alt kümesi kullanıldı.

## Eğitimi Çalıştırma
```
# Orijinal DarkIR
python train.py -p ./options/train/LOLBlur.yml

# CSF’li DarkIR
python train.py -p ./options/train/LOLBlur_csf.yml   # use_csf: true
```
`options/train/*` içinde yol ve model isimlerini kendi ortamına göre güncelle.

## Test
```
# CSF açık test
python testing.py -p ./options/test/LOLBlur_CSF.yml

# Orijinal model testi (use_csf: false, doğru checkpoint yolu)
python testing.py -p ./options/test/LOLBlur.yml
```
- Metrikler konsola yazılır.
- İlk 8 örnek görseli `save.results_dir` (varsayılan `./images/results_test`) altına kaydeder.
- Unpaired test: `python testing_unpaired.py -p ./options/test/RealBlur_Night.yml`

## İnference (görsel kaydetme)
```
python inference.py -p ./options/inference/LOLBlur.yml -i ./path/to/images
# Çıktılar: ./images/results

python inference_video.py -p ./options/inference_video/Baseline.yml -i /path/to/video.mp4
# Çıktı video: ./videos/results
```
Config’teki checkpoint yolu ve `use_csf` bayrağını eğittiğin modele göre ayarla.

## Sonuçlar (özet)
- Validation (600 çift):
  - DarkIR: PSNR ~23.97, SSIM ~0.808, L1 ~0.0540 (epoch ~60)
  - DarkIR + CSF: PSNR ~24.32, SSIM ~0.797, L1 ~0.0478 (epoch ~20)
- Test (540 çift):
  - DarkIR: PSNR 22.25, SSIM 0.6928, LPIPS 0.3858
  - DarkIR + CSF: PSNR 22.70, SSIM 0.7185, LPIPS 0.3641
- Yorum: CSF validation’da kazanç sağlarken testte orijinal DarkIR daha iyi genelledi. Daha fazla veri/regularization/CSF tasarımı ile fark kapanabilir.

## Nitel Sonuçlar
- Testten seçilen input/gt/output örneklerini `images/results_test` (veya ayarladığınız `results_dir`) altında bulabilirsiniz. DarkIR vs. CSF görsel karşılaştırmaları buraya eklenebilir.

## Sınırlılıklar ve Gelecek Çalışmalar
- Küçük veri alt kümesi, tek veri seti (LOLBlur).
- CSF bloğu basit; optimum iddiası yok.
- Gelecek işler: tam veriyle eğitim, farklı LLIE setleri, ek regularization (perceptual/TV), farklı CSF varyantları, çoklu görev denemeleri.

## Hızlı Kullanım Özeti
```
# Eğitim
python train.py -p ./options/train/LOLBlur.yml
python train.py -p ./options/train/LOLBlur_csf.yml

# Test (metrik + örnek görsel)
python testing.py -p ./options/test/LOLBlur.yml        # use_csf ve checkpoint yolunu ayarla
python testing.py -p ./options/test/LOLBlur_CSF.yml

# İnference
python inference.py -p ./options/inference/LOLBlur.yml -i ./path/to/your/images
```
