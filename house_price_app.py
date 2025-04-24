import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import warnings
import os
from PIL import Image

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor

warnings.simplefilter(action='ignore', category=FutureWarning)

# Sayfa yapılandırması en başta yapılmalı
st.set_page_config(page_title="Ev Fiyat Tahmin Sistemi", layout="wide")

# Create tabs for each stage of the pipeline
tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Ana Sayfa", "EDA", "Özellik Mühendisliği", "Model Eğitimi", "Tahmin", 
    "Özellik Önemi", "Fiyata Göre Öneri", "Özelliklere Göre Öneri", "Evim Ne Kadar Eder?"
])

with tab0:
    st.title("Akıllı Emlak Sistemi")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        try:
            image1 = Image.open("image1.png")
            st.image(image1, width=400)
        except:
            st.write("image1.png dosyası bulunamadı")
    
    with col2:
        try:
            miuul = Image.open("miuul.png")
            st.image(miuul, width=200)
        except:
            st.write("miuul.png dosyası bulunamadı")
    
    st.markdown("""
    ### Hoş Geldiniz!
    
    Bu uygulama, ev fiyatlarını tahmin etmek ve benzer evleri önermek için geliştirilmiş bir yapay zeka sistemidir.
    """)
    
    # Özellikler bölümü
    st.markdown("## 🏠 Sistem Özellikleri")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📊 Keşifsel Veri Analizi
        - Veri seti istatistikleri
        - Eksik değer analizi
        - Fiyat dağılımı
        - Özellik korelasyonları
        """)
    
    with col2:
        st.markdown("""
        ### 🛠️ Özellik Mühendisliği
        - Aykırı değer tespiti
        - Eksik değer doldurma
        - Yeni özellik oluşturma
        - Kategorik değişken kodlama
        """)
    
    with col3:
        st.markdown("""
        ### 🤖 Model Eğitimi
        - 11 farklı makine öğrenmesi modeli
        - Model performans karşılaştırması
        - Hiperparametre optimizasyonu
        - Çapraz doğrulama
        """)
    
    # Öneri Sistemleri
    st.markdown("## 🔍 Öneri Sistemleri")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 💰 Fiyata Göre Öneri
        - Hedef fiyata en yakın evleri bulma
        - Detaylı ev özellikleri
        - Benzer fiyatlı evlerin karşılaştırması
        - Ev ID'leri ile kolay referans
        """)
    
    with col2:
        st.markdown("""
        ### 🏡 Özelliklere Göre Öneri
        - İstediğiniz özelliklere sahip evleri bulma
        - Toplam alan, oda sayısı, banyo sayısı
        - Yapım yılı, garaj kapasitesi
        - Arsa alanı ve genel kalite
        """)
    
    with col3:
        st.markdown("""
        ### 🏘️ Evim Ne Kadar Eder?
        - Kendi evinizin özelliklerini girin
        - 11 farklı model seçeneği
        - Detaylı fiyat tahmini
        - Benzer evlerle karşılaştırma
        """)
    
    # Nasıl Kullanılır?
    st.markdown("## 📝 Nasıl Kullanılır?")
    
    st.markdown("""
    1. **Keşifsel Veri Analizi**: Veri setini inceleyin ve temel istatistikleri görün
    2. **Özellik Mühendisliği**: Veri ön işleme adımlarını takip edin
    3. **Model Eğitimi**: 11 farklı modelin performanslarını karşılaştırın
    4. **Öneri Sistemleri**: 
       - Fiyata göre benzer evleri bulun
       - İstediğiniz özelliklere sahip evleri arayın
       - Kendi evinizin değerini tahmin edin
    """)
    
    # İletişim Bölümü
    st.markdown("## 📞 İletişim")
    
    st.markdown("""
    ### Proje Geliştiricileri
    
    - [Ebubekir Tilbaç](https://www.linkedin.com/in/ebubekirtilbac/)
    - [Serdar İster](https://www.linkedin.com/in/serdar-ister-5aa6a719/)
    - [Cem Koçer](https://www.linkedin.com/in/cem-ko%C3%A7er-60cmkcr/)
    """)

class DataManager:
    """Veri yönetimi işlemlerini yönetir"""
    @staticmethod
    def load_data():
        """Veri setlerini yükler"""
        try:
            train_df = pd.read_csv("train.csv")
            test_df = pd.read_csv("test.csv")
            return train_df, test_df
        except Exception as e:
            st.error(f"Veri yükleme hatası: {str(e)}")
            return None, None

    @staticmethod
    def preprocess_data(train_df, test_df):
        try:
            # 1. Outlier removal
            train_df = train_df.drop(train_df[(train_df["GrLivArea"] > 4000) & (train_df["SalePrice"] < 300000)].index)
            
            # 2. Handle missing values
            # Numeric columns
            numeric_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
            for col in numeric_cols:
                if col in train_df.columns:
                    train_df[col].fillna(train_df[col].median(), inplace=True)
                if col in test_df.columns:
                    test_df[col].fillna(train_df[col].median(), inplace=True)
            
            # Categorical columns
            categorical_cols = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", 
                              "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
                              "GarageType", "GarageFinish", "GarageQual", "GarageCond"]
            for col in categorical_cols:
                if col in train_df.columns:
                    train_df[col].fillna("None", inplace=True)
                if col in test_df.columns:
                    test_df[col].fillna("None", inplace=True)
            
            # 3. Feature engineering
            # KALACAK özellikler
            train_df["OverallQual_GrLivArea"] = train_df["OverallQual"] * train_df["GrLivArea"]
            test_df["OverallQual_GrLivArea"] = test_df["OverallQual"] * test_df["GrLivArea"]
            
            train_df["OverallQual_YearBuilt"] = train_df["OverallQual"] * train_df["YearBuilt"]
            test_df["OverallQual_YearBuilt"] = test_df["OverallQual"] * test_df["YearBuilt"]
            
            train_df["TotalSF"] = train_df["TotalBsmtSF"] + train_df["1stFlrSF"] + train_df["2ndFlrSF"]
            test_df["TotalSF"] = test_df["TotalBsmtSF"] + test_df["1stFlrSF"] + test_df["2ndFlrSF"]
            
            train_df["NEW_1st*GrLiv"] = train_df["1stFlrSF"] * train_df["GrLivArea"]
            test_df["NEW_1st*GrLiv"] = test_df["1stFlrSF"] * test_df["GrLivArea"]
            
            train_df["NEW_Garage*GrLiv"] = train_df["GarageArea"] * train_df["GrLivArea"]
            test_df["NEW_Garage*GrLiv"] = test_df["GarageArea"] * test_df["GrLivArea"]
            
            train_df["NEW_TotalSqFeet"] = train_df["GrLivArea"] + train_df["TotalBsmtSF"]
            test_df["NEW_TotalSqFeet"] = test_df["GrLivArea"] + test_df["TotalBsmtSF"]
            
            # 4. Label encode ordinal features
            qual_map = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}
            ord_cols = ["ExterQual", "ExterCond", "HeatingQC", "KitchenQual", "BsmtQual", "BsmtCond"]
            for col in ord_cols:
                if col in train_df.columns:
                    train_df[col] = train_df[col].map(qual_map).fillna(0).astype(int)
                if col in test_df.columns:
                    test_df[col] = test_df[col].map(qual_map).fillna(0).astype(int)
            
            # 5. One-hot encode nominal categorical features
            # First, ensure all categorical columns are strings
            categorical_cols = train_df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                train_df[col] = train_df[col].astype(str)
                if col in test_df.columns:
                    test_df[col] = test_df[col].astype(str)
            
            # Combine train and test for consistent encoding
            full_df = pd.concat([train_df.drop("SalePrice", axis=1), test_df], axis=0)
            full_df = pd.get_dummies(full_df, drop_first=True)
            
            # Split back into train/test
            n_train = len(train_df)
            X_train = full_df.iloc[:n_train, :].copy()
            X_test = full_df.iloc[n_train:, :].copy()
            y_train = train_df["SalePrice"].values
            
            return X_train, X_test, y_train
            
        except Exception as e:
            st.error(f"Veri ön işleme hatası: {str(e)}")
            st.write("Hata detayı:", str(e))
            return None, None, None

class ModelManager:
    """Model yönetimi işlemlerini yönetir"""
    @staticmethod
    def get_models():
        """Kullanılacak modelleri döndürür"""
        return {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(),
            'Lasso Regression': Lasso(),
            'Decision Tree': DecisionTreeRegressor(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': XGBRegressor(n_estimators=100, random_state=42),
            'LightGBM': LGBMRegressor(n_estimators=100, random_state=42),
            'CatBoost': CatBoostRegressor(n_estimators=100, random_state=42, verbose=False)
        }

    @staticmethod
    def train_model(model, X_train, y_train):
        """Modeli eğitir"""
        model.fit(X_train, y_train)
        return model

    @staticmethod
    def evaluate_model(model, X_test, y_test):
        """Model performansını değerlendirir"""
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return rmse, mae, r2

class HouseRecommender:
    """Ev öneri sistemi işlemlerini yönetir"""
    @staticmethod
    def get_user_input():
        """Kullanıcıdan özellik değerlerini alır"""
        col1, col2 = st.columns(2)
        with col1:
            overall_qual = st.slider("Genel Kalite (1-10)", 1, 10, 5)
            gr_liv_area = st.slider("Yaşam Alanı (sqft)", 300, 5000, 1500)
            total_bsmt_sf = st.slider("Bodrum Alanı (sqft)", 0, 3000, 1000)
        with col2:
            full_bath = st.slider("Tam Banyo Sayısı", 0, 4, 2)
            garage_cars = st.slider("Garaj Kapasitesi (Araç)", 0, 4, 2)
            year_built = st.slider("Yapım Yılı", 1870, 2010, 2000)
        
        return np.array([overall_qual, gr_liv_area, total_bsmt_sf, 
                        full_bath, garage_cars, year_built])

    @staticmethod
    def find_similar_houses(train, user_input):
        """Benzer evleri bulur"""
        feature_cols = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 
                       'FullBath', 'GarageCars', 'YearBuilt']
        X_features = train[feature_cols].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_features)
        user_input_scaled = scaler.transform(user_input.reshape(1, -1))
        
        knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
        knn.fit(X_scaled)
        distances, indices = knn.kneighbors(user_input_scaled)
        
        return indices[0], distances[0]

    @staticmethod
    def display_results(train, indices, distances):
        """Sonuçları gösterir"""
        st.write("**Size En Uygun 5 Ev Önerisi:**")
        for i, (idx, dist) in enumerate(zip(indices, distances)):
            st.write(f"**{i+1}. Ev**")
            st.write(f"- Fiyat: ${train.iloc[idx]['SalePrice']:,.0f}")
            st.write(f"- Genel Kalite: {train.iloc[idx]['OverallQual']}")
            st.write(f"- Yaşam Alanı: {train.iloc[idx]['GrLivArea']} sqft")
            st.write(f"- Bodrum Alanı: {train.iloc[idx]['TotalBsmtSF']} sqft")
            st.write(f"- Tam Banyo: {train.iloc[idx]['FullBath']}")
            st.write(f"- Garaj Kapasitesi: {train.iloc[idx]['GarageCars']}")
            st.write(f"- Yapım Yılı: {train.iloc[idx]['YearBuilt']}")
            st.write(f"- Benzerlik Skoru: {1/(1+dist):.2f}")
            st.write("---")

class Visualizer:
    """Görselleştirme işlemlerini yönetir"""
    @staticmethod
    def plot_price_distribution(train):
        """Fiyat dağılımını gösterir"""
        fig = px.histogram(train, x="SalePrice", title="Ev Fiyatları Dağılımı")
        st.plotly_chart(fig)

    @staticmethod
    def plot_feature_importance(model, feature_names):
        """Özellik önemini gösterir"""
        importance = pd.DataFrame({
            'Özellik': feature_names,
            'Önem': model.feature_importances_
        }).sort_values('Önem', ascending=False)
        
        fig = px.bar(importance, x='Özellik', y='Önem', 
                     title="Özellik Önemi")
        st.plotly_chart(fig)

    @staticmethod
    def plot_model_performance(models_performance):
        """Model performansını gösterir"""
        fig = go.Figure(data=[
            go.Bar(name='RMSE', x=list(models_performance.keys()), 
                  y=[p[0] for p in models_performance.values()]),
            go.Bar(name='MAE', x=list(models_performance.keys()), 
                  y=[p[1] for p in models_performance.values()]),
            go.Bar(name='R2', x=list(models_performance.keys()), 
                  y=[p[2] for p in models_performance.values()])
        ])
        fig.update_layout(barmode='group', title='Model Performans Karşılaştırması')
        st.plotly_chart(fig)

def main():
    """Ana uygulama fonksiyonu"""
    st.title("Ev Fiyat Tahmin ve Öneri Sistemi")
    
    # Veri yükleme ve ön işleme
    train, test = DataManager.load_data()
    if train is None or test is None:
        return
    
    X_train, X_test, y_train = DataManager.preprocess_data(train, test)
    
    if X_train is not None and X_test is not None:
        with tab1:
            st.header("Keşifsel Veri Analizi")
            if train is not None and test is not None:
                st.write(f"**Eğitim Seti:** {train.shape[0]} satır, {train.shape[1]} sütun")
                st.write(f"**Test Seti:** {test.shape[0]} satır, {test.shape[1]} sütun")
                
                st.subheader("Eğitim Verisi Örneği")
                st.dataframe(train.head())
                
                na_counts = train.isnull().sum()
                na_counts = na_counts[na_counts > 0].sort_values(ascending=False)
                st.subheader("Eksik Değerler (Eğitim Seti)")
                st.write(na_counts.astype(int))
                
                # SalePrice dağılımı
                fig = px.histogram(train, x="SalePrice", nbins=50, title="Satış Fiyatı Dağılımı")
                st.plotly_chart(fig)
                
                # Korelasyon ısı haritası
                corr = train.corr(numeric_only=True)
                top_feats = corr["SalePrice"].abs().sort_values(ascending=False).head(11).index
                fig = go.Figure(data=go.Heatmap(
                    z=corr.loc[top_feats, top_feats],
                    x=top_feats,
                    y=top_feats,
                    colorscale='YlGnBu',
                    text=np.round(corr.loc[top_feats, top_feats], 2),
                    texttemplate="%{text}",
                    textfont={"size": 10}
                ))
                fig.update_layout(title="Satış Fiyatı ile En İlişkili Özelliklerin Korelasyonu")
                st.plotly_chart(fig)

        with tab2:
            st.header("Özellik Mühendisliği ve Ön İşleme")
            
            # Veri ön işleme adımlarını göster
            st.subheader("1. Veri Temizleme ve Dönüşümler")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Eksik Değerler**")
                na_counts = train.isnull().sum()
                na_counts = na_counts[na_counts > 0].sort_values(ascending=False)
                st.write(na_counts.astype(int))
                
                st.write("**Aykırı Değerler**")
                st.write("Aykırı değer kriteri: GrLivArea > 4000 ve SalePrice < 300000")
            
            with col2:
                st.write("**Kategorik Değişkenler**")
                categorical_cols = train.select_dtypes(include=['object']).columns
                st.write(f"Toplam kategorik değişken sayısı: {len(categorical_cols)}")
                
                # Kategorik değişkenleri tabloya dönüştür
                cat_df = pd.DataFrame({
                    'Değişken': categorical_cols,
                    'Benzersiz Değer Sayısı': [train[col].nunique() for col in categorical_cols],
                    'Örnek Değerler': [', '.join(train[col].unique()[:3].astype(str)) for col in categorical_cols]
                })
                st.dataframe(cat_df)
            
            # Yeni özelliklerin oluşturulması
            st.subheader("2. Yeni Özelliklerin Oluşturulması")
            
            # KALACAK özelliklerin gösterimi
            st.write("**Önemli Yeni Özellikler**")
            
            new_features = {
                "OverallQual_GrLivArea": "Genel Kalite × Yaşam Alanı",
                "OverallQual_YearBuilt": "Genel Kalite × Yapım Yılı",
                "TotalSF": "Toplam Alan (Bodrum + 1. Kat + 2. Kat)",
                "NEW_1st*GrLiv": "1. Kat Alanı × Yaşam Alanı",
                "NEW_Garage*GrLiv": "Garaj Alanı × Yaşam Alanı",
                "NEW_TotalSqFeet": "Toplam Metrekare (Yaşam Alanı + Bodrum)"
            }
            
            for feature, description in new_features.items():
                st.write(f"- **{feature}**: {description}")
            
            # Özellik dağılımlarını göster
            st.subheader("3. Özellik Dağılımları")
            
            selected_feature = st.selectbox(
                "Görüntülemek istediğiniz özelliği seçin",
                list(train.columns)
            )
            
            if selected_feature in train.columns:
                fig = px.histogram(train, x=selected_feature, 
                                 title=f"{selected_feature} Dağılımı")
                st.plotly_chart(fig)
                
                # Korelasyon analizi - sadece sayısal değişkenler için
                if selected_feature != 'SalePrice' and pd.api.types.is_numeric_dtype(train[selected_feature]):
                    st.write("**Korelasyon Analizi**")
                    corr = train[selected_feature].corr(train['SalePrice'])
                    st.write(f"SalePrice ile korelasyon: {corr:.4f}")
                    
                    # Scatter plot
                    fig = px.scatter(train, x=selected_feature, y='SalePrice',
                                   title=f"{selected_feature} vs SalePrice")
                    st.plotly_chart(fig)
                elif not pd.api.types.is_numeric_dtype(train[selected_feature]):
                    st.warning("Bu değişken kategorik olduğu için korelasyon analizi yapılamaz.")
            
            # Kodlama işlemleri
            st.subheader("4. Kodlama İşlemleri")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Label Encoding (İkili Kategorik Değişkenler)**")
                binary_cols = [col for col in train.columns if train[col].dtypes == "O" and train[col].nunique() == 2]
                binary_df = pd.DataFrame({
                    'Değişken': binary_cols,
                    'Benzersiz Değerler': [', '.join(map(str, train[col].unique())) for col in binary_cols]
                })
                st.dataframe(binary_df)
            
            with col2:
                st.write("**One-Hot Encoding**")
                categorical_cols = [col for col in train.select_dtypes(include=['object']).columns 
                                 if col not in binary_cols]
                onehot_df = pd.DataFrame({
                    'Değişken': categorical_cols,
                    'Benzersiz Değer Sayısı': [train[col].nunique() for col in categorical_cols],
                    'Örnek Değerler': [', '.join(map(str, train[col].unique()[:3])) for col in categorical_cols]
                })
                st.dataframe(onehot_df)
            
            # Standardizasyon
            st.subheader("5. Standardizasyon")
            st.write("Sayısal değişkenler RobustScaler ile standardize edilecek.")
            num_cols = train.select_dtypes(include=['int64', 'float64']).columns
            st.write("Standardize edilecek değişken sayısı:", len(num_cols))
            
            # Son durum
            st.subheader("6. Son Durum")
            st.write(f"Başlangıç özellik sayısı: {train.shape[1]}")
            st.write(f"Toplam yeni özellik sayısı: {len(new_features)}")

        with tab3:
            st.header("Model Eğitimi ve Değerlendirme")
            # Split for validation
            X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
            
            # Daha hızlı eğitim için optimize edilmiş model listesi
            models = ModelManager.get_models()
            models_performance = {}

            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, (name, model) in enumerate(models.items()):
                st.write(f"### {name} Modeli Eğitiliyor")
                col1, col2 = st.columns(2)
                
                with col1:
                    status_text.text(f"{name} modeli eğitiliyor...")
                    start_time = time.time()

                    trained_model = ModelManager.train_model(model, X_train_sub, y_train_sub)
                    models_performance[name] = ModelManager.evaluate_model(trained_model, X_val, y_val)

                with col2:
                    # Make predictions
                    y_pred = trained_model.predict(X_val)
                    rmse, mae, r2 = models_performance[name]
                    
                    st.write("**Model Performansı**")
                    st.write(f"RMSE: {rmse:.4f}")
                    st.write(f"MAE: {mae:.4f}")
                    st.write(f"R² Score: {r2:.4f}")
                    st.write(f"Eğitim süresi: {time.time() - start_time:.2f} saniye")
                
                progress_bar.progress((i + 1) / len(models))
                st.write("---")

            status_text.text("Model eğitimi tamamlandı!")

            # Plot model performance
            Visualizer.plot_model_performance(models_performance)

            # En iyi model analizi
            best_model_name = min(models_performance, key=lambda k: models_performance[k][0])
            best_model = models[best_model_name]
            
            st.write(f"### En İyi Model: {best_model_name}")
            st.write(f"RMSE: {models_performance[best_model_name][0]:.4f}")
            
            # En iyi model parametreleri
            st.write("**Model Parametreleri:**")
            if hasattr(best_model, "get_params"):
                params = best_model.get_params()
                important_params = {k: v for k, v in params.items() 
                                 if not (k.endswith("_") or k.startswith("_"))}
                for param, value in important_params.items():
                    st.write(f"- {param}: {value}")
            
            # Özellik önemliliği
            if hasattr(best_model, "feature_importances_"):
                importances = best_model.feature_importances_
                importance_type = "Feature Importance"
            elif hasattr(best_model, "coef_"):
                importances = np.abs(best_model.coef_)
                importance_type = "Coefficient Magnitude"
            else:
                importances = None

            if importances is not None:
                feature_imp = pd.DataFrame({
                    'Özellik': X_train.columns,
                    'Önem': importances
                }).sort_values('Önem', ascending=False).head(20)

                fig = px.bar(feature_imp, x='Önem', y='Özellik', 
                           title=f"En Önemli 20 Özellik ({importance_type})",
                           orientation='h')
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig)

            # Save the best model to session state
            st.session_state['best_model'] = best_model
            st.session_state['best_model_name'] = best_model_name

        with tab4:
            st.header("Test Seti Tahminleri")
            if X_test is not None:
                model = st.session_state['best_model']
                test_preds = model.predict(X_test)
                output_df = pd.DataFrame({"Id": test["Id"], "SalePrice": test_preds})
                
                # Tahmin dağılımını göster
                fig = px.histogram(output_df, x="SalePrice", nbins=50, title="Tahmin Edilen Fiyat Dağılımı")
                st.plotly_chart(fig)
                
                st.write("Test seti için ilk birkaç tahmin:")
                st.dataframe(output_df.head(10))
                
                csv_data = output_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Tahminleri CSV Olarak İndir",
                    data=csv_data,
                    file_name="ev_fiyat_tahminleri.csv",
                    mime="text/csv"
                )

        with tab5:
            st.header("Özellik Önemi")
            if X_train is not None:
                model = st.session_state['best_model']
                
                if hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_
                    feat_names = X_train.columns
                elif hasattr(model, "coef_"):
                    importances = abs(model.coef_)
                    feat_names = X_train.columns
                else:
                    importances = None

                if importances is not None:
                    feat_imp_df = pd.DataFrame({"Özellik": feat_names, "Önem": importances})
                    feat_imp_df.sort_values("Önem", ascending=False, inplace=True)
                    top_feats = feat_imp_df.head(10)
                    
                    fig = px.bar(top_feats, x="Özellik", y="Önem", title=f"{st.session_state['best_model_name']} için En Önemli 10 Özellik")
                    st.plotly_chart(fig)
                    
                    st.write(f"**{st.session_state['best_model_name']}** için en önemli 10 özellik:")
                    st.write(top_feats)
                else:
                    st.write("Seçilen model için özellik önemi bilgisi mevcut değil.")

        with tab6:
            st.header("Fiyata Göre Öneri Sistemi")
            
            if 'X_train' in locals() and 'y_train' in locals():
                # Kullanıcıdan fiyat al
                target_price = st.number_input("Hedef Fiyat", min_value=0, value=200000)
                
                # Fiyata en yakın evleri bul
                price_diff = abs(y_train - target_price)
                closest_indices = np.argsort(price_diff)[:5]  # En yakın 5 evin indekslerini al
                
                st.write("### Önerilen Evler")
                for i, idx in enumerate(closest_indices):
                    st.write(f"**Ev {i+1} (ID: {train.iloc[idx]['Id']})**")
                    st.write(f"Fiyat: ${y_train[idx]:,.2f}")
                    st.write(f"Toplam Alan: {X_train.iloc[idx]['TotalSF']:.0f} sqft")
                    st.write(f"Yatak Odası: {X_train.iloc[idx]['BedroomAbvGr']}")
                    st.write(f"Banyo: {X_train.iloc[idx]['FullBath']}")
                    st.write(f"Yapım Yılı: {X_train.iloc[idx]['YearBuilt']}")
                    st.write(f"Garaj Kapasitesi: {X_train.iloc[idx]['GarageCars']}")
                    st.write("---")

        with tab7:
            st.header("Özelliklere Göre Öneri Sistemi")
            
            if 'X_train' in locals() and 'y_train' in locals():
                # Özelliğe göre öneri sistemi
                st.subheader("Özelliğe Göre Ev Önerileri")
                
                # Kullanıcıdan özellik değerlerini al
                user_input = HouseRecommender.get_user_input()
                
                # Eğitim verilerinden ilgili özellikleri seç
                feature_cols = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 
                              'FullBath', 'GarageCars', 'YearBuilt']
                X_features = train[feature_cols].values
                
                # Özellikleri normalize et
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_features)
                user_input_scaled = scaler.transform(user_input.reshape(1, -1))
                
                # KNN ile en benzer evleri bul
                indices, distances = HouseRecommender.find_similar_houses(train, user_input)
                
                # Sonuçları göster
                HouseRecommender.display_results(train, indices, distances)

        with tab8:
            st.header("Evim Ne Kadar Eder?")
            
            if 'X_train' in locals() and 'y_train' in locals():
                st.write("### Lütfen evinizin özelliklerini girin")
                
                # Kullanıcıdan ev özelliklerini al
                total_sf = st.number_input("Toplam Alan (sqft)", min_value=500, max_value=5000, value=2000, key="total_sf_tab8")
                bedrooms = st.number_input("Yatak Odası Sayısı", min_value=1, max_value=10, value=3, key="bedrooms_tab8")
                bathrooms = st.number_input("Banyo Sayısı", min_value=1, max_value=5, value=2, key="bathrooms_tab8")
                year_built = st.number_input("Yapım Yılı", min_value=1800, max_value=2023, value=2000, key="year_built_tab8")
                garage_cars = st.number_input("Garaj Kapasitesi (Araba Sayısı)", min_value=0, max_value=5, value=2, key="garage_cars_tab8")
                lot_area = st.number_input("Arsa Alanı (sqft)", min_value=1000, max_value=50000, value=10000, key="lot_area_tab8")
                overall_qual = st.number_input("Genel Kalite (1-10)", min_value=1, max_value=10, value=5, key="overall_qual_tab8")
                overall_cond = st.number_input("Genel Durum (1-10)", min_value=1, max_value=10, value=5, key="overall_cond_tab8")
                bsmt_sf = st.number_input("Bodrum Alanı (sqft)", min_value=0, max_value=5000, value=1000, key="bsmt_sf_tab8")
                gr_liv_area = st.number_input("Yaşam Alanı (sqft)", min_value=500, max_value=5000, value=2000, key="gr_liv_area_tab8")
                
                # Model seçimi
                model_choice = st.selectbox("Tahmin Modeli Seçin", [
                    "Linear Regression",
                    "Ridge Regression",
                    "Lasso Regression",
                    "ElasticNet",
                    "Random Forest",
                    "Gradient Boosting",
                    "AdaBoost",
                    "SVR",
                    "Decision Tree",
                    "XGBoost",
                    "LightGBM",
                    "CatBoost"
                ], key="model_choice_tab8")
                
                if st.button("Fiyat Tahmini Yap", key="predict_price_tab8"):
                    # Örnek ev oluştur
                    sample_house = pd.DataFrame(columns=X_train.columns)
                    sample_house.loc[0] = 0  # Tüm değerleri 0 ile başlat
                    
                    # Sayısal özellikleri ata
                    sample_house["TotalSF"] = total_sf
                    sample_house["BedroomAbvGr"] = bedrooms
                    sample_house["FullBath"] = bathrooms
                    sample_house["YearBuilt"] = year_built
                    sample_house["GarageCars"] = garage_cars
                    sample_house["LotArea"] = lot_area
                    sample_house["OverallQual"] = overall_qual
                    sample_house["OverallCond"] = overall_cond
                    sample_house["TotalBsmtSF"] = bsmt_sf
                    sample_house["GrLivArea"] = gr_liv_area
                    
                    # Seçilen model ile tahmin yap
                    if model_choice == "Linear Regression":
                        model = LinearRegression()
                    elif model_choice == "Ridge Regression":
                        model = Ridge(alpha=1.0)
                    elif model_choice == "Lasso Regression":
                        model = Lasso(alpha=1.0)
                    elif model_choice == "ElasticNet":
                        model = ElasticNet(alpha=1.0, l1_ratio=0.5)
                    elif model_choice == "Random Forest":
                        model = RandomForestRegressor(n_estimators=100, random_state=42)
                    elif model_choice == "Gradient Boosting":
                        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                    elif model_choice == "AdaBoost":
                        model = AdaBoostRegressor(n_estimators=100, random_state=42)
                    elif model_choice == "SVR":
                        model = SVR()
                    elif model_choice == "Decision Tree":
                        model = DecisionTreeRegressor(random_state=42)
                    elif model_choice == "XGBoost":
                        model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
                    elif model_choice == "LightGBM":
                        model = LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
                    else:  # CatBoost
                        model = CatBoostRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbose=False)
                    
                    model.fit(X_train, y_train)
                    prediction = model.predict(sample_house)[0]
                    
                    st.success(f"Tahmini Ev Değeri: ${prediction:,.2f}")
                    
                    # Güven aralığı göster
                    st.write("### Tahmin Güven Aralığı")
                    st.write("Bu tahmin ±%10 hata payı ile verilmiştir.")
                    st.write(f"Alt Sınır: ${prediction*0.9:,.2f}")
                    st.write(f"Üst Sınır: ${prediction*1.1:,.2f}")
                    
                    # Benzer evleri göster
                    st.write("### Benzer Evler")
                    price_diff = abs(y_train - prediction)
                    closest_indices = np.argsort(price_diff)[:3]  # En yakın 3 evin indekslerini al
                    
                    for i, idx in enumerate(closest_indices):
                        st.write(f"**Benzer Ev {i+1} (ID: {train.iloc[idx]['Id']})**")
                        st.write(f"Fiyat: ${y_train[idx]:,.2f}")
                        st.write(f"Toplam Alan: {X_train.iloc[idx]['TotalSF']:.0f} sqft")
                        st.write(f"Yatak Odası: {X_train.iloc[idx]['BedroomAbvGr']}")
                        st.write(f"Banyo: {X_train.iloc[idx]['FullBath']}")
                        st.write("---")
    else:
        st.error("Veri ön işleme başarısız oldu. Lütfen veri setlerini kontrol edin.")

if __name__ == "__main__":
    main()

