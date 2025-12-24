# 🤖 AI vs Human Text Detector  
## Proje-2: Makale Özetleri Üzerinden Metin Tespiti

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange)


### 🎯 Proje Özeti
Bu proje, metinlerin **insan** mı yoksa **yapay zeka** tarafından mı yazıldığını tespit eden bir makine öğrenmesi uygulamasıdır. 3 farklı ML modeli ile yüksek doğruluk oranı sağlar.

### 📊 Proje TaskBoard
[GitHub Projects Board](https://github.com/users/selinkincal/projects/3)

| Durum | Görevler |
|-------|----------|
| ✅ **Tamamlandı** | Veri toplama, Temizleme, ML Eğitimi, UI Geliştirme |
| 🔄 **Devam Eden** | Test Yazımı, Dokümantasyon |
| 📋 **Planlanan** | White Box Testler, Kod Kalite Analizi |

**Güncel Durum:** 6/11 görev tamamlandı (%55)

### 🏗️ Mimari Yapı
AI Human Detector Projesi
├── 📁 data/ # Veri setleri
│ ├── arxiv_3000.json # Human makale özetleri
│ ├── ai_3000.json # AI üretilmiş özetler
│ ├── clean_human.json # Temizlenmiş human verileri
│ └── clean_ai.json # Temizlenmiş AI verileri
├── 📁 models/ # Eğitilmiş modeller
│ ├── naive_bayes_model.pkl
│ ├── random_forest_model.pkl
│ ├── svm_model.pkl
│ └── tfidf_vectorizer.pkl
├── 📁 src/ # Kaynak kodlar
│ ├── clean_data.py # Veri temizleme
│ ├── prepare_ml_data.py # ML veri hazırlığı
│ ├── train_ml_models_final.py # Model eğitimi
│ └── app.py # Streamlit uygulaması
├── 📁 tests/ # Test dosyaları
│ ├── test_app.py # Unit testler
│ └── test_cases.md # Test case dokümanı
├── 📁 docs/ # Dokümantasyon
│ ├── sözleşme.docx # Yazılım şartnamesi
│ └── raporlar/ # Performans raporları
├── requirements.txt # Gereksinimler
├── README.md # Bu dosya
└── .gitignore # Git ignore dosyası


### 🚀 Hızlı Başlangıç

```bash
# 1. Repo'yu klonla
git clone https://github.com/selinkincal/ai-human_text_detection.git
cd ai-human-detector

# 2. Sanal ortam oluştur (opsiyonel)
python -m venv venv

venv\Scripts\activate     # Windows

# 3. Gereksinimleri yükle
pip install -r requirements.txt

# 4. Uygulamayı çalıştır
streamlit run app.py


📈 Model Performansı

Model	        Accuracy	Precision	Recall	F1-Score	Eğitim Süresi
Naive Bayes 	94.2%	      93.8%	    94.1%	 93.9%	        4.2s
Random Forest	96.8%	      96.5%	    96.7%	 96.6%	        28.5s
SVM            	98.1%	      97.9%	    98.0%	 97.9%	        45.8s

