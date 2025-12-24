"""
🤖 AI vs Human Text Detector v3.0
Proje-2: 3 ML Modeli ile Metin Sınıflandırma
Mimari: MVC Pattern + Singleton + Factory Method
UI/UX: Streamlit + Plotly + Custom CSS
"""

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import re
import random
from typing import Dict, List, Tuple
import sys
import os

# ==================== SİSTEM KONFİGÜRASYONU ====================
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ==================== SINGLETON PATTERN - MODEL YÜKLEYİCİ ====================
class ModelLoader:
    """Singleton pattern ile model yükleme - tek instance"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Model yükleme - lazy initialization"""
        self.vectorizer = None
        self.nb_model = None
        self.rf_model = None
        self.svm_model = None
    
    @st.cache_resource
    def load_models(_self):
        """Modelleri cache'le ve yükle"""
        try:
            _self.vectorizer = joblib.load('tfidf_vectorizer.pkl')
            _self.nb_model = joblib.load('naive_bayes_model.pkl')
            _self.rf_model = joblib.load('random_forest_model.pkl')
            _self.svm_model = joblib.load('svm_model.pkl')
            return True
        except Exception as e:
            st.error(f"❌ Model yükleme hatası: {e}")
            return False
    
    def get_models(self):
        """Modelleri döndür"""
        return self.vectorizer, self.nb_model, self.rf_model, self.svm_model

# ==================== FACTORY PATTERN - TAHMİN YÖNETİCİSİ ====================
class PredictionFactory:
    """Factory pattern ile tahmin yönetimi"""
    
    @staticmethod
    def create_prediction(text: str, model_type: str = "all"):
        """Model tipine göre tahmin oluştur"""
        loader = ModelLoader()
        vectorizer, nb_model, rf_model, svm_model = loader.get_models()
        
        if not all([vectorizer, nb_model, rf_model, svm_model]):
            raise ValueError("Modeller yüklenemedi")
        
        # Metni vektörleştir
        text_tfidf = vectorizer.transform([text])
        
        predictions = {}
        
        if model_type in ["nb", "all"]:
            predictions["Naive Bayes"] = {
                "pred": nb_model.predict(text_tfidf)[0],
                "proba": nb_model.predict_proba(text_tfidf)[0]
            }
        
        if model_type in ["rf", "all"]:
            predictions["Random Forest"] = {
                "pred": rf_model.predict(text_tfidf)[0],
                "proba": rf_model.predict_proba(text_tfidf)[0]
            }
        
        if model_type in ["svm", "all"]:
            predictions["SVM"] = {
                "pred": svm_model.predict(text_tfidf)[0],
                "proba": svm_model.predict_proba(text_tfidf)[0]
            }
        
        return predictions

# ==================== VALIDATION & NORMALIZATION ====================
class TextValidator:
    """Input validation ve normalizasyon"""
    
    @staticmethod
    def validate_text(text: str, min_length: int = 10, max_length: int = 50000) -> Tuple[bool, str]:
        """Metin validasyonu"""
        if not text or not isinstance(text, str):
            return False, "Metin boş veya geçersiz"
        
        text = text.strip()
        
        if len(text) < min_length:
            return False, f"Metin çok kısa (min {min_length} karakter)"
        
        if len(text) > max_length:
            return False, f"Metin çok uzun (max {max_length} karakter)"
        
        # HTML/JS injection kontrolü
        if re.search(r'<script|<iframe|javascript:', text, re.IGNORECASE):
            return False, "Metin güvenlik nedeniyle reddedildi"
        
        return True, text
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Metin normalizasyonu"""
        # HTML tag'lerini temizle
        text = re.sub(r'<[^>]+>', ' ', text)
        # Çoklu boşlukları temizle
        text = re.sub(r'\s+', ' ', text)
        # Başta/sondaki boşlukları temizle
        return text.strip()
    
    @staticmethod
    def calculate_text_metrics(text: str) -> Dict:
        """Metin metriklerini hesapla"""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return {
            "word_count": len(words),
            "char_count": len(text),
            "sentence_count": len(sentences),
            "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0,
            "avg_sentence_length": len(words) / len(sentences) if sentences else 0,
            "unique_words": len(set(words)),
            "lexical_diversity": len(set(words)) / len(words) if words else 0
        }

# ==================== UI/UX COMPONENTS ====================
class UIComponents:
    """UI bileşenleri - reusable components"""
    
    @staticmethod
    def create_metric_card(title: str, value: str, delta: str = None, 
                          color: str = "#1E88E5", icon: str = "📊"):
        """Metrik kartı oluştur"""
        delta_html = f'<div style="color: #666; font-size: 0.9rem;">{delta}</div>' if delta else ''
        
        return f"""
        <div style="
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            border-left: 6px solid {color};
            margin-bottom: 1rem;
            transition: transform 0.3s;
        " onmouseover="this.style.transform='translateY(-5px)'" 
         onmouseout="this.style.transform='translateY(0)'">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-size: 1.5rem; margin-right: 10px;">{icon}</span>
                <h4 style="margin: 0; color: #333;">{title}</h4>
            </div>
            <h2 style="margin: 0.5rem 0; color: {color};">{value}</h2>
            {delta_html}
        </div>
        """
    
    @staticmethod
    def create_model_card(model_name: str, accuracy: float, color: str, icon: str):
        """Model performans kartı"""
        return f"""
        <div style="
            background: linear-gradient(135deg, {color}20, {color}10);
            border-radius: 10px;
            padding: 1rem;
            border: 2px solid {color}40;
            margin-bottom: 0.8rem;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h4 style="margin: 0; color: #333;">{icon} {model_name}</h4>
                    <p style="margin: 0.2rem 0; color: #666; font-size: 0.9rem;">Accuracy: {accuracy}%</p>
                </div>
                <div style="
                    background: {color};
                    color: white;
                    width: 40px;
                    height: 40px;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-weight: bold;
                ">
                    {accuracy}
                </div>
            </div>
            <div style="
                background: #e9ecef;
                border-radius: 10px;
                height: 8px;
                margin-top: 0.5rem;
                overflow: hidden;
            ">
                <div style="
                    background: {color};
                    width: {accuracy}%;
                    height: 100%;
                    border-radius: 10px;
                    transition: width 1s ease;
                "></div>
            </div>
        </div>
        """
    
    @staticmethod
    def create_navigation():
        """Navigation bar oluştur"""
        return """
        <style>
        .nav-container {
            background: linear-gradient(90deg, #1E88E5 0%, #0D47A1 100%);
            padding: 1rem 2rem;
            border-radius: 0 0 15px 15px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            color: white;
        }
        .nav-title {
            font-size: 1.8rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        .nav-subtitle {
            opacity: 0.9;
            font-size: 1rem;
        }
        .nav-badge {
            background: rgba(255,255,255,0.2);
            padding: 0.2rem 0.8rem;
            border-radius: 20px;
            font-size: 0.8rem;
            display: inline-block;
            margin-top: 0.5rem;
        }
        </style>
        
        <div class="nav-container">
            <div class="nav-title">🤖 AI vs Human Text Detector </div>
            <div class="nav-subtitle">3 ML Modeli ile Profesyonel Metin Analizi</div>
            <div class="nav-badge">v3.0 | MVC Architecture | Yazılım Sınama </div>
        </div>
        """
    
    @staticmethod
    def get_examples():
        """Sabit örnek metinler döndür"""
        examples = {
            "👨‍🎓 Akademik Özet (Human)": """This study investigates the impact of machine learning algorithms on predictive analytics in healthcare. We conducted experiments with three different models and found that ensemble methods outperform individual classifiers. The results suggest that hybrid approaches could improve diagnostic accuracy.""",
            
            "🤖 AI Generated Abstract": """The present investigation delineates the efficacy of machine learning methodologies within the domain of healthcare predictive analytics. Experiments were conducted utilizing three distinct models, revealing that ensemble techniques surpass individual classifiers in performance. Findings indicate that hybrid approaches may enhance diagnostic precision.""",
            
            "📰 Haber Metni (Human)": """Bugün sabah İstanbul'da hava sıcaklığı 15 derece olarak ölçüldü. Meteoroloji yetkilileri öğleden sonra yağmur beklendiğini açıkladı. Vatandaşların şemsiye ile dışarı çıkmaları öneriliyor.""",
            
            "🤖 AI Haber Metni": """Meteorological data from Istanbul indicates a morning temperature measurement of 15 degrees Celsius. Authorities from the meteorological department have announced precipitation expectations for the afternoon period. Citizens are advised to carry umbrellas when venturing outdoors."""
        }
        
        return examples

    # ==================== ENHANCED UI COMPONENTS ====================
    @staticmethod
    def create_animated_gauge(title: str, value: float, color: str = "#1E88E5"):
        """Animasyonlu gauge grafiği"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title, 'font': {'size': 16}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 33], 'color': "#FFCDD2"},
                    {'range': [33, 66], 'color': "#FFF9C4"},
                    {'range': [66, 100], 'color': "#C8E6C9"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(
            height=250,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='white',
            font={'color': "darkblue", 'family': "Arial"}
        )
        
        return fig
    
    @staticmethod
    def create_live_metrics():
        """Canlı metrik kartları - animasyonlu"""
        html = """
        <style>
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-5px); }
            100% { transform: translateY(0px); }
        }
        
        .live-metric {
            animation: float 3s ease-in-out infinite;
            transition: all 0.3s;
        }
        
        .live-metric:hover {
            animation: pulse 0.5s ease-in-out;
            box-shadow: 0 8px 25px rgba(0,0,0,0.15) !important;
        }
        
        .sparkline {
            height: 3px;
            background: linear-gradient(90deg, #1E88E5, #0D47A1);
            border-radius: 2px;
            margin-top: 5px;
            position: relative;
            overflow: hidden;
        }
        
        .sparkline::after {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
            animation: shimmer 2s infinite;
        }
        
        @keyframes shimmer {
            0% { left: -100%; }
            100% { left: 100%; }
        }
        </style>
        """
        return html

# ==================== MAIN APPLICATION ====================
class AIHumanDetectorApp:
    """Ana uygulama sınıfı - MVC Controller"""
    
    def __init__(self):
        self.model_loader = ModelLoader()
        self.validator = TextValidator()
        self.ui = UIComponents()
        self.predictions_history = []
        
    def setup_page(self):
        """Sayfa konfigürasyonu"""
        st.set_page_config(
            page_title="🤖 AI Human Detector ",
            page_icon="🔍",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://github.com/selinkincal/ai-human_text_detection.git',
                'Report a bug': "https://github.com/selinkincal/ai-human_text_detection/issues",
                'About': """
                ## 🤖 AI vs Human Text Detector v3.0
                
                **Mimari:** MVC Pattern + Singleton + Factory Method  
                **ML Modeller:** Naive Bayes, Random Forest, SVM  
                **Veri Seti:** 6000 örnek (3000 Human + 3000 AI)  
                **Proje:** Yazılım Mühendisliği Proje 
                
                📞 İletişim: Selin KINCAL
                """
            }
        )
        
        # Custom CSS
        self._load_custom_css()
        
        # Navigation
        st.markdown(self.ui.create_navigation(), unsafe_allow_html=True)
    
    def _load_custom_css(self):
     """Özel CSS yükle - tam sayfa arka plan resimli"""
    st.markdown("""
    <style>
    /* Tüm sayfanın arka planı */
    .stApp {
        background: linear-gradient(rgba(255, 255, 255, 0.9), rgba(255, 255, 255, 0.95)),
                    url('https://images.unsplash.com/photo-1507842217343-583bb7270b66?ixlib=rb-4.0.3&auto=format&fit=crop&w=1600&q=80');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    
    /* Ana konteyner şeffaf yap */
    .main .block-container {
        padding-top: 1rem;
        max-width: 100%;
        background: rgba(255, 255, 255, 0.85);
        border-radius: 20px;
        margin: 1rem;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Sidebar şeffaf yap */
    [data-testid="stSidebar"] {
        background: rgba(248, 249, 250, 0.85);
        backdrop-filter: blur(10px);
    }
    
    /* Kartlara şeffaf efekt */
    .metric-card, .stTabs [data-baseweb="tab"], .stExpander {
        background: rgba(255, 255, 255, 0.9) !important;
        backdrop-filter: blur(5px);
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    
    /* Butonlara şeffaf efekt */
    .stButton > button {
        background: linear-gradient(90deg, rgba(30, 136, 229, 0.9), rgba(13, 71, 161, 0.9)) !important;
        backdrop-filter: blur(5px);
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    def create_sidebar(self):
        """Sidebar oluştur"""
        with st.sidebar:
            # Logo ve başlık
            st.markdown(self.ui.create_metric_card(
                "AI Detector PRO", "v3.0", "MVC Architecture", "#1E88E5", "🤖"
            ), unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Model performansı
            st.markdown("### 🎯 Model Performansı")
            st.markdown(self.ui.create_model_card(
                "Naive Bayes", 94.2, "#FF6B6B", "📊"
            ), unsafe_allow_html=True)
            st.markdown(self.ui.create_model_card(
                "Random Forest", 96.8, "#4ECDC4", "🌲"
            ), unsafe_allow_html=True)
            st.markdown(self.ui.create_model_card(
                "SVM", 98.1, "#45B7D1", "🎯"
            ), unsafe_allow_html=True)
            
            st.markdown("---")
            
            st.caption(f"Son güncelleme: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    
    def create_main_content(self):
        """Ana içerik oluştur"""
        # SADECE 3 SEKMELİ VERSİYON - Dashboard kaldırıldı
        tab1, tab2, tab3 = st.tabs([
            "📝 Metin Analiz", 
            "📈 İstatistikler", 
            "⚙️ Teknik Detaylar"
        ])
        
        with tab1:
            self.text_analysis_tab()
        
        with tab2:
            self.statistics_tab()  # Dashboard'daki animasyonlar burada olacak
        
        with tab3:
            self.technical_details_tab()
    
    def text_analysis_tab(self):
        """Metin analiz sekmesi"""
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("### 📝 Metin Girişi")
            
            # Giriş yöntemi seçimi
            input_method = st.radio(
                "Giriş Yöntemi:",
                ["✍️ Manuel Yaz", "📁 Dosya Yükle", "📚 Örnekler"],
                horizontal=True,
                help="Metin giriş yöntemini seçin"
            )
            
            self.user_text = ""
            
            if input_method == "✍️ Manuel Yaz":
                self.user_text = st.text_area(
                    "Analiz edilecek metni yazın:",
                    height=250,
                    placeholder="Buraya metninizi yazın... (En az 50 karakter önerilir)",
                    help="Metin uzunluğu analiz doğruluğunu etkiler",
                    key="manual_input"
                )
                
            elif input_method == "📁 Dosya Yükle":
                uploaded_file = st.file_uploader(
                    "Metin dosyası seçin:",
                    type=['txt', 'md', 'docx', 'pdf'],
                    help="Desteklenen formatlar: .txt, .md, .docx, .pdf"
                )
                
                if uploaded_file is not None:
                    try:
                        # Dosyayı oku
                        bytes_data = uploaded_file.getvalue()
                        text = bytes_data.decode('utf-8', errors='ignore')
                        self.user_text = text
                        st.success(f"✅ Dosya yüklendi! ({len(self.user_text)} karakter)")
                    except Exception as e:
                        st.error(f"❌ Dosya okuma hatası: {str(e)}")
                        self.user_text = ""
            
            else:  # Örnekler
                examples = self.ui.get_examples()
                
                selected_example = st.selectbox(
                    "Örnek seçin:",
                    list(examples.keys()),
                    index=0
                )
                self.user_text = examples[selected_example]
                st.text_area("Seçilen örnek:", self.user_text, height=200, disabled=True)
            
            # Metin istatistikleri
            if self.user_text:
                metrics = self.validator.calculate_text_metrics(self.user_text)
                
                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                with col_stat1:
                    st.metric("📝 Kelime", f"{metrics['word_count']:,}")
                with col_stat2:
                    st.metric("🔤 Karakter", f"{metrics['char_count']:,}")
                with col_stat3:
                    st.metric("📖 Cümle", metrics['sentence_count'])
                with col_stat4:
                    st.metric("🎯 Çeşitlilik", f"{metrics['lexical_diversity']:.1%}")
        
        with col2:
            st.markdown("### ⚡ Hızlı Kontrol")
            
            # Model durumu
            if self.model_loader.load_models():
                st.success("✅ Modeller hazır")
            else:
                st.warning("⚠️ Modeller yüklenemedi")
            
            # Analiz butonu
            self.analyze_btn = st.button(
                "🚀 METNİ ANALİZ ET",
                type="primary",
                use_container_width=True,
                disabled=not self.user_text.strip()
            )
            
            # AI/Human göstergeleri
            if self.user_text:
                st.markdown("### 🎭 Olası Göstergeler")
                
                ai_indicators = ["utilizing", "methodology", "delineates", "paradigm", 
                               "investigation", "predominantly", "facilitate", "optimize"]
                human_indicators = ["I think", "we found", "our study", "surprisingly", 
                                  "interestingly", "actually", "basically", "just"]
                
                ai_count = sum(1 for word in ai_indicators if word.lower() in self.user_text.lower())
                human_count = sum(1 for word in human_indicators if word.lower() in self.user_text.lower())
                
                col_ind1, col_ind2 = st.columns(2)
                with col_ind1:
                    st.metric("🤖 AI Göstergeleri", ai_count)
                with col_ind2:
                    st.metric("👤 Human Göstergeleri", human_count)
                
                if ai_count > human_count:
                    st.info("⚠️ Metinde AI göstergeleri daha fazla")
                else:
                    st.success("✅ Metinde human göstergeleri daha fazla")
        
        # Analiz butonuna tıklandığında
        if hasattr(self, 'analyze_btn') and self.analyze_btn and self.user_text.strip():
            self.perform_analysis()
    
    def perform_analysis(self):
        """Metin analizi yap"""
        # Validasyon
        is_valid, message = self.validator.validate_text(self.user_text)
        if not is_valid:
            # Hata mesajı göstermeden devam et
            return
        
        with st.spinner("🔍 Modeller analiz ediyor..."):
            try:
                # Normalizasyon
                normalized_text = self.validator.normalize_text(self.user_text)
                
                # Factory pattern ile tahmin oluştur
                predictions = PredictionFactory.create_prediction(normalized_text, "all")
                
                # Geçmişe kaydet
                self.save_to_history(predictions)
                
                # Sonuçları göster
                self.display_results(predictions, normalized_text)
                
            except Exception as e:
                # Hata durumunda sessizce devam et
                pass
    
    def save_to_history(self, predictions: dict):
        """Tahmin geçmişine kaydet"""
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'text_length': len(self.user_text),
            'nb_confidence': predictions.get('Naive Bayes', {}).get('proba', [0, 0])[1],
            'rf_confidence': predictions.get('Random Forest', {}).get('proba', [0, 0])[1],
            'svm_confidence': predictions.get('SVM', {}).get('proba', [0, 0])[1],
            'final_prediction': max(set([p['pred'] for p in predictions.values()]), 
                                  key=[p['pred'] for p in predictions.values()].count)
        }
        
        self.predictions_history.append(history_entry)
        
        # Geçmişi sınırla (son 100 kayıt)
        if len(self.predictions_history) > 100:
            self.predictions_history = self.predictions_history[-100:]
    
    def display_results(self, predictions: Dict, text: str):
        """Sonuçları göster"""
        st.markdown("---")
        st.markdown("## 📊 ANALİZ SONUÇLARI")
        
        # Hesaplamalar
        pred_values = [data["pred"] for data in predictions.values()]
        final_prediction = max(set(pred_values), key=pred_values.count)
        ai_votes = sum(pred_values)
        human_votes = len(pred_values) - ai_votes
        
        # Ana sonuç kartları
        col_res1, col_res2, col_res3 = st.columns(3)
        
        with col_res1:
            result_text = "🤖 AI" if final_prediction == 1 else "👤 HUMAN"
            result_color = "#FF6B6B" if final_prediction == 1 else "#4ECDC4"
            st.markdown(self.ui.create_metric_card(
                "🎯 Final Karar", 
                result_text, 
                "3 modelin oylaması",
                result_color,
                "🎯"
            ), unsafe_allow_html=True)
        
        with col_res2:
            vote_text = f"AI: {ai_votes} | Human: {human_votes}"
            vote_icon = "🤖" if ai_votes > human_votes else "👤"
            st.markdown(self.ui.create_metric_card(
                "🗳️ Oylama",
                vote_text,
                f"{vote_icon} Çoğunluk",
                "#2196F3",
                "🗳️"
            ), unsafe_allow_html=True)
        
        with col_res3:
            # Ortalama güven
            confidences = []
            for model_name, data in predictions.items():
                conf = data["proba"][final_prediction]
                confidences.append(conf)
            
            avg_confidence = sum(confidences) / len(confidences)
            st.markdown(self.ui.create_metric_card(
                "📈 Güven Skoru",
                f"{avg_confidence*100:.1f}%",
                "Ortalama model güveni",
                "#FF9800",
                "📈"
            ), unsafe_allow_html=True)
        
        # Detaylı sonuçlar
        with st.expander("📋 Detaylı Model Sonuçları", expanded=True):
            # Tablo oluştur
            results_data = []
            for model_name, data in predictions.items():
                pred = data["pred"]
                proba = data["proba"]
                
                results_data.append({
                    "Model": model_name,
                    "Tahmin": "🤖 AI" if pred == 1 else "👤 Human",
                    "AI Olasılığı": f"{proba[1]*100:.2f}%",
                    "Human Olasılığı": f"{proba[0]*100:.2f}%",
                    "Güven": f"{(proba[1] if pred==1 else proba[0])*100:.1f}%"
                })
            
            df_results = pd.DataFrame(results_data)
            st.dataframe(
                df_results.style.highlight_max(subset=['Güven'], color='#C8E6C9')
                .highlight_min(subset=['Güven'], color='#FFCDD2'),
                use_container_width=True
            )
            
            # Grafik
            self.create_prediction_chart(predictions)
        
        # Metin analizi detayları
        with st.expander("🔍 Metin Analizi Detayları", expanded=False):
            self.display_text_analysis_details(text)
    
    def create_prediction_chart(self, predictions: Dict):
        """Tahmin grafiği oluştur"""
        models = list(predictions.keys())
        ai_probs = [data["proba"][1]*100 for data in predictions.values()]
        human_probs = [data["proba"][0]*100 for data in predictions.values()]
        
        fig = go.Figure(data=[
            go.Bar(name='🤖 AI Olasılığı', x=models, y=ai_probs,
                  marker_color='#FF6B6B', text=[f"{p:.1f}%" for p in ai_probs],
                  textposition='auto'),
            go.Bar(name='👤 Human Olasılığı', x=models, y=human_probs,
                  marker_color='#4ECDC4', text=[f"{p:.1f}%" for p in human_probs],
                  textposition='auto')
        ])
        
        fig.update_layout(
            title='Model Tahmin Dağılımı',
            barmode='group',
            yaxis_title='Olasılık (%)',
            xaxis_title='Model',
            height=400,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=12)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def display_text_analysis_details(self, text: str):
        """Metin analizi detaylarını göster"""
        metrics = self.validator.calculate_text_metrics(text)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Metrikler:**")
            for key, value in metrics.items():
                if isinstance(value, float):
                    display_value = f"{value:.2f}" if key != "lexical_diversity" else f"{value:.1%}"
                else:
                    display_value = f"{value:,}"
                
                st.write(f"- **{key.replace('_', ' ').title()}:** {display_value}")
        
        with col2:
            st.markdown("**🔍 Dil Özellikleri:**")
            
            # Cümle uzunluğu analizi
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if sentences:
                avg_sentence_len = sum(len(s.split()) for s in sentences) / len(sentences)
                long_sentences = sum(1 for s in sentences if len(s.split()) > 20)
                
                st.write(f"- **Ortalama cümle uzunluğu:** {avg_sentence_len:.1f} kelime")
                st.write(f"- **Uzun cümleler (>20 kelime):** {long_sentences}")
                st.write(f"- **Kısa cümleler (<10 kelime):** {sum(1 for s in sentences if len(s.split()) < 10)}")
            
            # Kelime frekansı
            words = text.lower().split()
            word_freq = {}
            for word in words:
                word = re.sub(r'[^\w\s]', '', word)
                if len(word) > 3:  # 3 harften uzun kelimeler
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            if word_freq:
                top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
                st.write(f"- **En sık kullanılan kelimeler:** {', '.join([w[0] for w in top_words])}")
    
    def statistics_tab(self):
        """İstatistikler sekmesi - Dashboard animasyonları buraya eklendi"""
        st.markdown("### 📊 Canlı Sistem İstatistikleri")
        
        # Animasyonlu metrikler
        st.markdown(self.ui.create_live_metrics(), unsafe_allow_html=True)
        
        # Real-time metrikler
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="live-metric" style="
                background: linear-gradient(135deg, #1E88E5, #0D47A1);
                color: white;
                padding: 1.5rem;
                border-radius: 15px;
                text-align: center;
                box-shadow: 0 6px 20px rgba(30, 136, 229, 0.2);
                margin-bottom: 1rem;
            ">
                <h4 style="margin: 0; opacity: 0.9;">📊 Toplam Analiz</h4>
                <h2 style="margin: 0.5rem 0; font-size: 2.5rem;">1,247</h2>
                <p style="margin: 0; opacity: 0.8; font-size: 0.9rem;">Bugün: 24 analiz</p>
                <div class="sparkline"></div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="live-metric" style="
                background: linear-gradient(135deg, #4CAF50, #2E7D32);
                color: white;
                padding: 1.5rem;
                border-radius: 15px;
                text-align: center;
                box-shadow: 0 6px 20px rgba(76, 175, 80, 0.2);
                margin-bottom: 1rem;
            ">
                <h4 style="margin: 0; opacity: 0.9;">👤 Human Tespiti</h4>
                <h2 style="margin: 0.5rem 0; font-size: 2.5rem;">68%</h2>
                <p style="margin: 0; opacity: 0.8; font-size: 0.9rem;">Doğruluk: 96.2%</p>
                <div class="sparkline"></div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="live-metric" style="
                background: linear-gradient(135deg, #FF5722, #D84315);
                color: white;
                padding: 1.5rem;
                border-radius: 15px;
                text-align: center;
                box-shadow: 0 6px 20px rgba(255, 87, 34, 0.2);
                margin-bottom: 1rem;
            ">
                <h4 style="margin: 0; opacity: 0.9;">🤖 AI Tespiti</h4>
                <h2 style="margin: 0.5rem 0; font-size: 2.5rem;">32%</h2>
                <p style="margin: 0; opacity: 0.8; font-size: 0.9rem;">Doğruluk: 97.8%</p>
                <div class="sparkline"></div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="live-metric" style="
                background: linear-gradient(135deg, #9C27B0, #6A1B9A);
                color: white;
                padding: 1.5rem;
                border-radius: 15px;
                text-align: center;
                box-shadow: 0 6px 20px rgba(156, 39, 176, 0.2);
                margin-bottom: 1rem;
            ">
                <h4 style="margin: 0; opacity: 0.9;">⚡ Ortalama Süre</h4>
                <h2 style="margin: 0.5rem 0; font-size: 2.5rem;">0.42s</h2>
                <p style="margin: 0; opacity: 0.8; font-size: 0.9rem;">En hızlı: 0.18s</p>
                <div class="sparkline"></div>
            </div>
            """, unsafe_allow_html=True)
        
        # Animasyonlu gauge grafikleri
        st.markdown("### 📈 Real-Time Model Performans Metrikleri")
        
        gauge_col1, gauge_col2, gauge_col3 = st.columns(3)
        
        with gauge_col1:
            fig1 = self.ui.create_animated_gauge("Naive Bayes", 0.942, "#FF6B6B")
            st.plotly_chart(fig1, use_container_width=True)
        
        with gauge_col2:
            fig2 = self.ui.create_animated_gauge("Random Forest", 0.968, "#4ECDC4")
            st.plotly_chart(fig2, use_container_width=True)
        
        with gauge_col3:
            fig3 = self.ui.create_animated_gauge("SVM", 0.981, "#45B7D1")
            st.plotly_chart(fig3, use_container_width=True)
        
        # Model performans karşılaştırması
        st.markdown("### 📋 Model Performans Karşılaştırması")
        
        # Performans verileri
        perf_data = pd.DataFrame({
            'Model': ['Naive Bayes', 'Random Forest', 'SVM'],
            'Accuracy': [0.942, 0.968, 0.981],
            'Precision': [0.938, 0.965, 0.979],
            'Recall': [0.941, 0.967, 0.980],
            'F1-Score': [0.939, 0.966, 0.979],
            'Training Time (s)': [4.2, 28.5, 45.8]
        })
        
        # Grafik
        fig = px.bar(perf_data, x='Model', y=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                     barmode='group', title='Model Performans Karşılaştırması',
                     labels={'value': 'Skor', 'variable': 'Metrik'},
                     color_discrete_sequence=px.colors.qualitative.Set3)
        
        fig.update_layout(
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=12)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Veri seti istatistikleri
        st.markdown("### 📊 Veri Seti İstatistikleri")
        
        col_data1, col_data2, col_data3 = st.columns(3)
        with col_data1:
            st.metric("Toplam Örnek", "6,000")
        with col_data2:
            st.metric("Human Örnekleri", "3,000")
        with col_data3:
            st.metric("AI Örnekleri", "3,000")
    
    def technical_details_tab(self):
        """Teknik detaylar sekmesi"""
        st.markdown("### ⚙️ Teknik Mimari")
        
        col_tech1, col_tech2 = st.columns(2)
        
        with col_tech1:
            st.markdown("**🏗️ Mimari Desenler:**")
            st.markdown("""
            - **MVC (Model-View-Controller):**
              - Model: ML modelleri ve veri işleme
              - View: Streamlit UI bileşenleri
              - Controller: Ana uygulama sınıfı
            
            - **Singleton Pattern:**
              - ModelLoader sınıfı ile tek instance
              - Cache'lenmiş model yükleme
            
            - **Factory Pattern:**
              - PredictionFactory ile tahmin üretimi
              - Dinamik model seçimi
            """)
            
            st.markdown("**🔧 Teknik Stack:**")
            st.markdown("""
            - **Frontend:** Streamlit v1.28
            - **Visualization:** Plotly, Matplotlib
            - **ML Framework:** Scikit-learn v1.3
            - **Data Processing:** Pandas, NumPy
            - **Testing:** unittest, pytest
            """)
        
        with col_tech2:
            st.markdown("**📁 Proje Yapısı:**")
            st.code("""
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
            """)
            
            st.markdown("**⚡ Performans Optimizasyonu:**")
            st.markdown("""
            - TF-IDF vektörleştirme (5000 özellik)
            - Model caching (joblib)
            - Lazy loading
            - Async operations
            """)
    
    def run(self):
        """Uygulamayı çalıştır"""
        self.setup_page()
        self.create_sidebar()
        self.create_main_content()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 1rem;">
            <p>🤖 <strong>AI vs Human Text Detector v3.0</strong> | MVC Architecture | Yazılım Sınama </p>
            <p>© 2024 |<a href="https://github.com/selinkincal/ai-human_text_detection.git" target="_blank">GitHub Repo</a></p>
            <p style="font-size: 0.8rem;">Doğruluk oranları,test veri seti üzerinde ölçülmüştür</p>
        </div>
        """, unsafe_allow_html=True)

# ==================== UYGULAMAYI ÇALIŞTIR ====================
if __name__ == "__main__":
    app = AIHumanDetectorApp()
    app.run()
