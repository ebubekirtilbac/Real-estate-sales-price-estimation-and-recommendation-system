# Ev Fiyat Tahmin ve Öneri Sistemi

Bu proje, ev fiyatlarını tahmin etmek ve benzer evleri önermek için geliştirilmiş bir yapay zeka sistemidir. Streamlit kullanılarak geliştirilen web arayüzü sayesinde kullanıcılar kolayca ev fiyat tahminleri yapabilir ve benzer evleri keşfedebilirler.

## 📋 İçindekiler

1. [Proje Hakkında](#-proje-hakkında)
2. [Özellikler](#-özellikler)
3. [Kurulum](#-kurulum)
4. [Kullanım](#-kullanım)
5. [Veri Seti](#-veri-seti)
6. [Proje Yapısı](#-proje-yapısı)
7. [İletişim](#-iletişim)

## 📌 Proje Hakkında

Bu proje, ev fiyatlarını tahmin etmek ve benzer evleri önermek için geliştirilmiş bir yapay zeka sistemidir. Sistem, makine öğrenmesi modellerini kullanarak ev fiyatlarını tahmin eder ve kullanıcıların girdiği kriterlere göre benzer evleri önerir.

## 🚀 Özellikler

### 📊 Veri Analizi
- Keşifsel veri analizi (EDA)
- Fiyat dağılımı görselleştirmesi
- Özellik korelasyonları
- Eksik değer analizi

### 🛠️ Özellik Mühendisliği
- Aykırı değer tespiti
- Eksik değer doldurma
- Kategorik değişken kodlama
- Yeni özellik oluşturma

### 🤖 Model Eğitimi
- 11 farklı makine öğrenmesi modeli
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - LightGBM
  - CatBoost
- Model performans karşılaştırması
- Hiperparametre optimizasyonu
- Çapraz doğrulama

### 🔍 Öneri Sistemleri
- Fiyata göre ev önerileri
- Özelliklere göre ev önerileri
- Benzerlik skorları
- Detaylı ev bilgileri

## 🛠️ Kurulum

1. Python 3.8 veya üstü sürümünün yüklü olduğundan emin olun

2. Sanal ortam oluşturun ve aktifleştirin:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac için
# veya
.venv\Scripts\activate  # Windows için
```

3. Gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```

## 🚀 Kullanım

1. Uygulamayı başlatın:
```bash
streamlit run house_price_app.py
```

2. Tarayıcınızda açılan arayüzde:
   - Veri analizi sekmesinde veri setini inceleyin
   - Model eğitimi sekmesinde farklı modelleri deneyin
   - Öneri sistemleri sekmesinde ev arayın

## 📊 Veri Seti

### 📚 Veri Seti Hikayesi
Proje, Ames Housing veri setini kullanmaktadır. Bu veri seti, Iowa eyaletinin Ames şehrindeki konut satışlarını içermektedir. Veri seti, 2006-2010 yılları arasında satılan konutların detaylı özelliklerini ve satış fiyatlarını içerir. Veri seti, Dr. Dean De Cock tarafından derlenmiş ve makine öğrenmesi eğitimi için ideal bir veri seti olarak kabul edilmektedir.

### 📋 Veri Seti Özellikleri
- 80'den fazla özellik
- 1460 eğitim örneği
- 1459 test örneği
- Sayısal ve kategorik değişkenler

### 📝 Tüm Değişkenler ve Açıklamaları

#### 🏠 Temel Özellikler
- **SalePrice**: Evin satış fiyatı (hedef değişken)
- **MSSubClass**: Konut sınıfı (20: 1-kat, 30: 1.5-kat, vb.)
- **MSZoning**: Genel bölge sınıflandırması
- **LotArea**: Parsel alanı (metrekare)
- **Street**: Sokak erişim tipi
- **Alley**: Sokak erişim tipi
- **LotShape**: Parsel şekli
- **LandContour**: Parsel eğimi
- **Utilities**: Mevcut hizmetler
- **LotConfig**: Parsel yapılandırması
- **LandSlope**: Parsel eğimi
- **Neighborhood**: Fiziksel konum
- **Condition1**: Ana yol veya demiryoluna yakınlık
- **Condition2**: Ana yol veya demiryoluna yakınlık (ikinci durum)
- **BldgType**: Konut tipi
- **HouseStyle**: Konut stili
- **OverallQual**: Genel malzeme ve bitiş kalitesi (1-10)
- **OverallCond**: Genel durum değerlendirmesi (1-10)
- **YearBuilt**: İnşa yılı
- **YearRemodAdd**: Yenileme yılı
- **RoofStyle**: Çatı tipi
- **RoofMatl**: Çatı malzemesi
- **Exterior1st**: Dış kaplama malzemesi
- **Exterior2nd**: Dış kaplama malzemesi (ikinci)
- **MasVnrType**: Duvar kaplama tipi
- **MasVnrArea**: Duvar kaplama alanı
- **ExterQual**: Dış malzeme kalitesi
- **ExterCond**: Dış malzeme durumu
- **Foundation**: Temel tipi
- **BsmtQual**: Bodrum yüksekliği
- **BsmtCond**: Bodrum genel durumu
- **BsmtExposure**: Bodrum duvarları
- **BsmtFinType1**: Bodrum bitmiş alan kalitesi
- **BsmtFinSF1**: Tip 1 bitmiş alan
- **BsmtFinType2**: İkinci bitmiş alan tipi
- **BsmtFinSF2**: Tip 2 bitmiş alan
- **BsmtUnfSF**: Bitmemiş bodrum alanı
- **TotalBsmtSF**: Toplam bodrum alanı
- **Heating**: Isıtma tipi
- **HeatingQC**: Isıtma kalitesi ve durumu
- **CentralAir**: Merkezi klima
- **Electrical**: Elektrik sistemi
- **1stFlrSF**: Birinci kat alanı
- **2ndFlrSF**: İkinci kat alanı
- **LowQualFinSF**: Düşük kaliteli bitmiş alan
- **GrLivArea**: Üst kat(lar) yaşam alanı
- **BsmtFullBath**: Bodrum tam banyo
- **BsmtHalfBath**: Bodrum yarım banyo
- **FullBath**: Üst kat tam banyo
- **HalfBath**: Üst kat yarım banyo
- **BedroomAbvGr**: Bodrum dışı yatak odası
- **KitchenAbvGr**: Bodrum dışı mutfak
- **KitchenQual**: Mutfak kalitesi
- **TotRmsAbvGrd**: Üst kat toplam oda
- **Functional**: Ev fonksiyonelliği
- **Fireplaces**: Şömine sayısı
- **FireplaceQu**: Şömine kalitesi
- **GarageType**: Garaj konumu
- **GarageYrBlt**: Garaj inşa yılı
- **GarageFinish**: Garaj iç bitişi
- **GarageCars**: Garaj kapasitesi
- **GarageArea**: Garaj alanı
- **GarageQual**: Garaj kalitesi
- **GarageCond**: Garaj durumu
- **PavedDrive**: Asfaltlı yol
- **WoodDeckSF**: Ahşap güverte alanı
- **OpenPorchSF**: Açık veranda alanı
- **EnclosedPorch**: Kapalı veranda alanı
- **3SsnPorch**: Üç mevsim veranda alanı
- **ScreenPorch**: Ekranlı veranda alanı
- **PoolArea**: Havuz alanı
- **PoolQC**: Havuz kalitesi
- **Fence**: Çit kalitesi
- **MiscFeature**: Çeşitli özellikler
- **MiscVal**: Çeşitli özelliklerin değeri
- **MoSold**: Satış ayı
- **YrSold**: Satış yılı
- **SaleType**: Satış tipi
- **SaleCondition**: Satış koşulu

### 📦 Veri Seti İndirme
Veri seti ve detaylı açıklamaları şu adresten indirilebilir:
[Veri Seti ve Dokümantasyon](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)

## 🏗️ Proje Yapısı

```
ev-fiyat-tahmin/
├── house_price_app.py      # Ana uygulama dosyası
├── requirements.txt        # Gerekli kütüphaneler
├── README.md              # Proje dokümantasyonu
├── train.csv              # Eğitim veri seti
└── test.csv               # Test veri seti
```

## 📞 İletişim

Proje hakkında sorularınız için:
- Email: ebubekirtilbac@gmail.com
- LinkedIn: [Ebubekir Tilbaç](https://www.linkedin.com/in/ebubekirtilbac/)
- LinkedIn: [Serdar İster](https://www.linkedin.com/in/serdar-ister-5aa6a719/)
- LinkedIn: [Cem Koçer](https://www.linkedin.com/in/cem-ko%C3%A7er-60cmkcr/)

## 🙏 Teşekkürler

- [Miuul](https://www.miuul.com/) - Eğitim ve destekleri için
- [Kaggle](https://www.kaggle.com/) - Veri seti için
- Tüm katkıda bulunanlara teşekkürler 