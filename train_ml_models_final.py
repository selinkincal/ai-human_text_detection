import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("                3 ML MODELİ EĞİTİMİ - PROJE-2")
print("=" * 70)

def load_data():
    """Verileri yükleyelim"""
    print("\n📥 VERİLER YÜKLENİYOR...")
    train_df = pd.read_csv("train_data.csv", encoding="utf-8")
    test_df = pd.read_csv("test_data.csv", encoding="utf-8")
    
    print(f"✓ Eğitim verisi: {len(train_df)} örnek")
    print(f"✓ Test verisi: {len(test_df)} örnek")
    print(f"✓ Human/Test oranı: {sum(train_df['label']==0)}/{sum(train_df['label']==1)}")
    
    X_train = train_df["text"]
    y_train = train_df["label"]
    X_test = test_df["text"]
    y_test = test_df["label"]
    
    return X_train, X_test, y_train, y_test

def create_tfidf_features(X_train, X_test):
    """TF-IDF vektörleştirme"""
    print("\n🔠 TF-IDF VEKTÖRLEŞTİRME...")
    
    # TF-IDF vektörleştirici
    vectorizer = TfidfVectorizer(
        max_features=5000,       # En sık kullanılan 5000 kelime
        min_df=5,                # En az 5 dokümanda geçsin
        max_df=0.8,              # En fazla %80 dokümanda geçsin
        stop_words='english',    # İngilizce stop words
        ngram_range=(1, 2),      # 1-gram ve 2-gram
        sublinear_tf=True        # TF ölçeği
    )
    
    print("  Vektörleştirici eğitiliyor...")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"  Kelime dağarcığı: {len(vectorizer.get_feature_names_out())} kelime")
    print(f"  Eğitim boyutu: {X_train_tfidf.shape}")
    print(f"  Test boyutu: {X_test_tfidf.shape}")
    
    # En önemli kelimeleri göster
    feature_names = vectorizer.get_feature_names_out()
    feature_importance = np.asarray(X_train_tfidf.mean(axis=0)).ravel().tolist()
    top_indices = np.argsort(feature_importance)[-10:]
    
    print(f"\n  🔝 En önemli 10 kelime:")
    for idx in reversed(top_indices):
        print(f"     - {feature_names[idx]}")
    
    return vectorizer, X_train_tfidf, X_test_tfidf

def train_naive_bayes(X_train, y_train, X_test, y_test):
    """Naive Bayes modeli eğitimi"""
    print("\n" + "=" * 50)
    print("1. NAIVE BAYES MODELİ")
    print("=" * 50)
    
    start_time = time.time()
    
    # Model oluştur ve eğit
    print("  Model eğitiliyor...")
    nb_model = MultinomialNB(alpha=0.1)
    nb_model.fit(X_train, y_train)
    
    # Tahminler
    y_pred = nb_model.predict(X_test)
    y_pred_proba = nb_model.predict_proba(X_test)[:, 1]
    
    # Metrikler
    train_time = time.time() - start_time
    
    return evaluate_model("Naive Bayes", nb_model, y_test, y_pred, y_pred_proba, train_time)

def train_random_forest(X_train, y_train, X_test, y_test):
    """Random Forest modeli eğitimi"""
    print("\n" + "=" * 50)
    print("2. RANDOM FOREST MODELİ")
    print("=" * 50)
    
    start_time = time.time()
    
    # Model oluştur ve eğit
    print("  Model eğitiliyor...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    rf_model.fit(X_train, y_train)
    
    # Tahminler
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    
    # Metrikler
    train_time = time.time() - start_time
    
    return evaluate_model("Random Forest", rf_model, y_test, y_pred, y_pred_proba, train_time)

def train_svm(X_train, y_train, X_test, y_test):
    """SVM modeli eğitimi"""
    print("\n" + "=" * 50)
    print("3. SUPPORT VECTOR MACHINE (SVM) MODELİ")
    print("=" * 50)
    
    start_time = time.time()
    
    # Model oluştur ve eğit
    print("  Model eğitiliyor (bu biraz zaman alabilir)...")
    svm_model = SVC(
        kernel='linear',
        C=1.0,
        probability=True,
        random_state=42,
        class_weight='balanced'
    )
    
    svm_model.fit(X_train, y_train)
    
    # Tahminler
    y_pred = svm_model.predict(X_test)
    y_pred_proba = svm_model.predict_proba(X_test)[:, 1]
    
    # Metrikler
    train_time = time.time() - start_time
    
    return evaluate_model("SVM", svm_model, y_test, y_pred, y_pred_proba, train_time)

def evaluate_model(model_name, model, y_true, y_pred, y_pred_proba, train_time):
    """Model performansını değerlendir"""
    
    # Temel metrikler
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    auc = roc_auc_score(y_true, y_pred_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Sonuçları yazdır
    print(f"\n📊 {model_name} PERFORMANS METRİKLERİ:")
    print(f"  ⭐ Accuracy:  {accuracy:.4f} (%{accuracy*100:.2f})")
    print(f"  📍 Precision: {precision:.4f}")
    print(f"  🔍 Recall:    {recall:.4f}")
    print(f"  ⚡ F1-Score:  {f1:.4f}")
    print(f"  📈 ROC-AUC:   {auc:.4f}")
    print(f"  ⏱️  Eğitim süresi: {train_time:.2f} saniye")
    
    print(f"\n📋 CLASSIFICATION REPORT:")
    print(classification_report(y_true, y_pred, target_names=['Human', 'AI'], digits=4))
    
    print(f"🔢 CONFUSION MATRIX:")
    print(f"                  Predicted")
    print(f"                 Human    AI")
    print(f"  Actual Human:  {cm[0, 0]:4d}     {cm[0, 1]:4d}")
    print(f"  Actual AI:     {cm[1, 0]:4d}     {cm[1, 1]:4d}")
    
    # Hata analizi
    human_accuracy = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    ai_accuracy = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
    
    print(f"\n🎯 SINIF BAZINDA DOĞRULUK:")
    print(f"  Human doğruluğu: %{human_accuracy*100:.2f}")
    print(f"  AI doğruluğu:    %{ai_accuracy*100:.2f}")
    
    # ROC curve verileri
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    
    return {
        "model_name": model_name,
        "model": model,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "train_time": train_time,
        "confusion_matrix": cm.tolist(),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "human_accuracy": human_accuracy,
        "ai_accuracy": ai_accuracy
    }

def save_models_and_results(results, vectorizer):
    """Modelleri ve sonuçları kaydet"""
    print("\n" + "=" * 50)
    print("💾 MODELLER VE SONUÇLAR KAYDEDİLİYOR...")
    print("=" * 50)
    
    # 1. Modelleri kaydet
    for result in results:
        model_name = result["model_name"].lower().replace(" ", "_")
        filename = f'{model_name}_model.pkl'
        joblib.dump(result["model"], filename)
        print(f"✓ {filename} kaydedildi ({os.path.getsize(filename):,} bytes)")
    
    # 2. Vektörleştiriciyi kaydet
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    print(f"✓ tfidf_vectorizer.pkl kaydedildi ({os.path.getsize('tfidf_vectorizer.pkl'):,} bytes)")
    
    # 3. Sonuçları JSON olarak kaydet
    results_summary = []
    for result in results:
        results_summary.append({
            "model": result["model_name"],
            "accuracy": float(result["accuracy"]),
            "precision": float(result["precision"]),
            "recall": float(result["recall"]),
            "f1_score": float(result["f1"]),
            "roc_auc": float(result["auc"]),
            "train_time_seconds": float(result["train_time"]),
            "human_accuracy": float(result["human_accuracy"]),
            "ai_accuracy": float(result["ai_accuracy"])
        })
    
    with open('model_results.json', 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    print("✓ model_results.json kaydedildi")
    
    # 4. En iyi modeli belirle
    best_model_idx = np.argmax([r["f1"] for r in results])
    best_model = results[best_model_idx]
    
    print(f"\n🏆 EN İYİ PERFORMANS: {best_model['model_name']}")
    print(f"   F1-Score: {best_model['f1']:.4f} (%{best_model['f1']*100:.2f})")
    print(f"   Accuracy: {best_model['accuracy']:.4f} (%{best_model['accuracy']*100:.2f})")
    print(f"   ROC-AUC:  {best_model['auc']:.4f}")

def create_performance_plots(results):
    """Performans grafikleri oluştur"""
    print("\n" + "=" * 50)
    print("📈 PERFORMANS GRAFİKLERİ OLUŞTURULUYOR...")
    print("=" * 50)
    
    try:
        # 1. Model karşılaştırma grafiği
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Human vs AI Text Detection - Model Performans Karşılaştırması', fontsize=16)
        
        model_names = [r["model_name"] for r in results]
        metrics_data = {
            "Accuracy": [r["accuracy"] for r in results],
            "Precision": [r["precision"] for r in results],
            "Recall": [r["recall"] for r in results],
            "F1-Score": [r["f1"] for r in results],
            "ROC-AUC": [r["auc"] for r in results],
            "Eğitim Süresi (s)": [r["train_time"] for r in results]
        }
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for idx, (metric_name, values) in enumerate(metrics_data.items()):
            ax = axes[idx // 3, idx % 3]
            bars = ax.bar(model_names, values, color=colors[:len(model_names)])
            ax.set_title(metric_name, fontweight='bold')
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            
            if metric_name == "Eğitim Süresi (s)":
                ax.set_ylabel('Saniye')
            else:
                ax.set_ylim([0.9, 1.0])
                ax.set_ylabel('Skor')
            
            # Değerleri üste yaz
            for bar, value in zip(bars, values):
                height = bar.get_height()
                if metric_name == "Eğitim Süresi (s)":
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{value:.2f}s', ha='center', va='bottom', fontsize=9)
                else:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                           f'{value:.4f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ model_performance_comparison.png kaydedildi")
        
        # 2. ROC Curves
        plt.figure(figsize=(10, 8))
        for result in results:
            plt.plot(result["fpr"], result["tpr"], 
                    label=f'{result["model_name"]} (AUC = {result["auc"]:.4f})',
                    linewidth=3)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Guess', linewidth=2)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Model Karşılaştırması', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        print("✓ roc_curves.png kaydedildi")
        
        # 3. Confusion Matrix'ler
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Confusion Matrices - Model Karşılaştırması', fontsize=16, fontweight='bold')
        
        for idx, result in enumerate(results):
            ax = axes[idx]
            cm = np.array(result["confusion_matrix"])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Human', 'AI'], 
                       yticklabels=['Human', 'AI'],
                       cbar_kws={'shrink': 0.8},
                       ax=ax)
            ax.set_title(f'{result["model_name"]}\nAccuracy: {result["accuracy"]*100:.2f}%', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Predicted Label', fontsize=11)
            ax.set_ylabel('True Label', fontsize=11)
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        print("✓ confusion_matrices.png kaydedildi")
        
        # 4. Sınıf bazlı doğruluk
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(model_names))
        width = 0.35
        
        human_acc = [r["human_accuracy"] for r in results]
        ai_acc = [r["ai_accuracy"] for r in results]
        
        rects1 = ax.bar(x - width/2, human_acc, width, label='Human', color='#4CAF50')
        rects2 = ax.bar(x + width/2, ai_acc, width, label='AI', color='#FF5722')
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Sınıf Bazlı Doğruluk Oranları', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.set_ylim([0.9, 1.0])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Değerleri yaz
        for rects, acc_list in zip([rects1, rects2], [human_acc, ai_acc]):
            for rect, acc in zip(rects, acc_list):
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()/2., height + 0.005,
                       f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('class_accuracy_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ class_accuracy_comparison.png kaydedildi")
        
        plt.show()
        
    except Exception as e:
        print(f"⚠️ Görselleştirme hatası: {e}")
        print("Matplotlib kurulu değilse: pip install matplotlib seaborn")

def main():
    """Ana fonksiyon"""
    
    # 1. Verileri yükle
    X_train, X_test, y_train, y_test = load_data()
    
    # 2. TF-IDF vektörleştirme
    vectorizer, X_train_tfidf, X_test_tfidf = create_tfidf_features(X_train, X_test)
    
    # 3. Modelleri eğit
    print("\n" + "=" * 70)
    print("🚀 MODEL EĞİTİMLERİ BAŞLIYOR...")
    print("=" * 70)
    
    results = []
    
    # Naive Bayes
    nb_result = train_naive_bayes(X_train_tfidf, y_train, X_test_tfidf, y_test)
    results.append(nb_result)
    
    # Random Forest
    rf_result = train_random_forest(X_train_tfidf, y_train, X_test_tfidf, y_test)
    results.append(rf_result)
    
    # SVM
    svm_result = train_svm(X_train_tfidf, y_train, X_test_tfidf, y_test)
    results.append(svm_result)
    
    # 4. Grafikleri oluştur
    create_performance_plots(results)
    
    # 5. Modelleri ve sonuçları kaydet
    save_models_and_results(results, vectorizer)
    
    # 6. Özet rapor
    print("\n" + "=" * 70)
    print("✅ 3 ML MODELİ EĞİTİMİ TAMAMLANDI!")
    print("=" * 70)
    
    print("\n📁 OLUŞTURULAN DOSYALAR:")
    print("  📦 Modeller:")
    print("    - naive_bayes_model.pkl")
    print("    - random_forest_model.pkl")
    print("    - svm_model.pkl")
    print("    - tfidf_vectorizer.pkl")
    print("  🎨 Görseller:")
    print("    - model_performance_comparison.png")
    print("    - roc_curves.png")
    print("    - confusion_matrices.png")
    print("    - class_accuracy_comparison.png")
    print("  📊 Veriler:")
    print("    - model_results.json")
    
    print("\n🎯 PROJE-2 KRİTER DURUMU:")
    print("  ✓ User Story-1: Veri Toplama (5/5)")
    print("  ✓ User Story-2: Veri Temizleme (5/5)")
    print("  ✓ User Story-3: 3 ML Modeli Eğitimi (10/10)")
    print("  📋 User Story-4: Yazılım Entegrasyonu (10/10) - Sonraki adım")
    print("  📋 User Story-5: 3 Model Tahmini (10/10) - Sonraki adım")
    
    print(f"\n⏱️  Toplam eğitim süresi: {sum(r['train_time'] for r in results):.2f} saniye")
    print(f"🏆 En iyi model: {results[np.argmax([r['f1'] for r in results])]['model_name']}")
    
    print("\n" + "=" * 70)
    print("🎯 BİR SONRAKİ ADIM: STREAMLIT UYGULAMASI OLUŞTURMA")
    print("=" * 70)

if __name__ == "__main__":
    import os
    main()