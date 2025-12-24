# 🤖 AI vs Human Text Detection

## 📋 Proje Hakkında
Bu proje, makale özetlerinin **insan** mı yoksa **yapay zeka** tarafından mı yazıldığını tespit eden bir makine öğrenmesi uygulamasıdır. 3 farklı ML modeli (Naive Bayes, Random Forest, SVM) kullanır ve Streamlit ile web arayüzü sunar.

👨‍💻 Geliştirici
Selin KINCAL - 232703059
Gülsu BEŞE - 2327030
Sena Nur BAHÇEVAN - 232703057


## 🎯 Özellikler
- ✅ **3 ML Modeli:** Naive Bayes, Random Forest, SVM
- ✅ **6000 Örnek Veri:** 3000 Human + 3000 AI
- ✅ **Profesyonel Mimari:** MVC Pattern + Singleton + Factory Method
- ✅ **Kullanıcı Dostu Arayüz:** Streamlit ile modern UI/UX
- ✅ **Detaylı Analiz:** 3 modelin tahminleri ve güven skorları

## 📊 Model Performansı
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Naive Bayes | 94.2% | 93.8% | 94.1% | 93.9% |
| Random Forest | 96.8% | 96.5% | 96.7% | 96.6% |
| SVM | 98.1% | 97.9% | 98.0% | 97.9% |

## 🚀 Kurulum ve Çalıştırma

### 1. Gereksinimler

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


### 📊 Proje TaskBoard
[GitHub Projects Board](https://github.com/users/selinkincal/projects/3)



### 🏗️ Mimari Yapı
AI_Human_Detector/
├── 📁 data/              # Veri setleri
├── 📁 clean_data/
├── 📁 images /            
│
├── 📁 models/                       # EĞİTİLMİŞ MODELLER (Git'te yok)
│   ├── naive_bayes_model.pkl
│   ├── random_forest_model.pkl
│   ├── svm_model.pkl
│   └── tfidf_vectorizer.pkl     
|        
├── app.py                 # Ana uygulama (MVC Controller)
├── arxiv_test.py           # Arxiv'den insan özetleri çek
├── fetch_arxiv.py          # Alternatif Arxiv API ile veri çekme
├── fetch_pubmed.py         # PubMed'den veri çekme (opsiyonel)
├── fetch_semantic.py       # Semantic Scholar'dan veri çekme (opsiyonel)
├──ai_generate_offline.py   # Ollama ile AI özetleri üret
├── clean_data.py           # HTML/LaTeX temizleme
├── prepare_ml_data.py      # Verileri birleştir ve CSV'ye çevir
├── validate.py            # Veri kontrolü
├── check_data.py          # Temizlenmiş veriyi kontrol et
├── train_ml_models.py      # 3 ML modelini eğit (temel)
├── train_ml_models_final.py # 3 ML modelini eğit (gelişmiş)
├── requirements.txt    # Bağımlılıklar
└── README.md           # Proje dokümantasyonu
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


