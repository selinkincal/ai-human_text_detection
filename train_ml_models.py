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
print("                3 ML MODELİ EĞİTİMİ")
print("=" * 70)

def load_data():
    """Verileri yükleyelim"""
    print("\n📥 VERİLER YÜKLENİYOR...")
    train_df = pd.read_csv("train_data.csv", encoding="utf-8")
    test_df = pd.read_csv("test_data.csv", encoding="utf-8")
    
    print(f"✓ Eğitim verisi: {len(train_df)} örnek")
    print(f"✓ Test verisi: {len(test_df)} örnek")
    
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
        ngram_range=(1, 2)       # 1-gram ve 2-gram
    )
    
    print("  Vektörleştirici eğitiliyor...")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"  Kelime dağarcığı boyutu: {len(vectorizer.get_feature_names_out())}")
    print(f"  Eğitim verisi boyutu: {X_train_tfidf.shape}")
    print(f"  Test verisi boyutu: {X_test_tfidf.shape}")
    
    return vectorizer, X_train_tfidf, X_test_tfidf

def train_naive_bayes(X_train, y_train, X_test, y_test):
    """Naive Bayes modeli eğitimi"""
    print("\n" + "=" * 50)
    print("1. NAIVE BAYES MODELİ")
    print("=" * 50)
    
    start_time = time.time()
    
    # Model oluştur ve eğit
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
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1  # Tüm CPU çekirdeklerini kullan
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
    svm_model = SVC(
        kernel='linear',
        C=1.0,
        probability=True,  # Olasılık tahmini için
        random_state=42
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
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {auc:.4f}")
    print(f"  Eğitim süresi: {train_time:.2f} saniye")
    
    print(f"\n📋 CLASSIFICATION REPORT:")
    print(classification_report(y_true, y_pred, target_names=['Human', 'AI']))
    
    print(f"🔢 CONFUSION MATRIX:")
    print(f"  True Human  (Pred Human): {cm[0, 0]}")
    print(f"  True Human  (Pred AI):    {cm[0, 1]}")
    print(f"  True AI     (Pred Human): {cm[1, 0]}")
    print(f"  True AI     (Pred AI):    {cm[1, 1]}")
    
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
        "tpr": tpr.tolist()
    }

def plot_results(results, vectorizer):
    """Sonuçları görselleştir"""
    print("\n📈 GÖRSELLEŞTİRME...")
    
    # 1. Model karşılaştırma grafiği
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Model Performans Karşılaştırması', fontsize=16)
    
    # Metrikleri topla
    model_names = [r["model_name"] for r in results]
    metrics = ["accuracy", "precision", "recall", "f1", "auc", "train_time"]
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "Eğitim Süresi (s)"]
    
    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx // 3, idx % 3]
        values = [r[metric] for r in results]
        
        if metric == "train_time":
            bars = ax.bar(model_names, values, color=['skyblue', 'lightgreen', 'salmon'])
            ax.set_ylabel('Saniye')
        else:
            bars = ax.bar(model_names, values, color=['skyblue', 'lightgreen', 'salmon'])
            ax.set_ylim([0.5, 1.0])  # Metrikler için
        
        ax.set_title(metric_name)
        ax.set_xticklabels(model_names, rotation=45)
        
        # Değerleri üste yaz
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ model_performance_comparison.png kaydedildi")
    
    # 2. ROC Curve
    plt.figure(figsize=(10, 8))
    for result in results:
        plt.plot(result["fpr"], result["tpr"], 
                label=f'{result["model_name"]} (AUC = {result["auc"]:.3f})',
                linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Model Karşılaştırması')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
    print("✓ roc_curves.png kaydedildi")
    
    # 3. Confusion Matrix'leri
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, result in enumerate(results):
        ax = axes[idx]
        cm = np.array(result["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Human', 'AI'], 
                   yticklabels=['Human', 'AI'], ax=ax)
        ax.set_title(f'{result["model_name"]} - Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("✓ confusion_matrices.png kaydedildi")
    
    plt.show()

def save_models_and_results(results, vectorizer):
    """Modelleri ve sonuçları kaydet"""
    print("\n💾 MODELLER VE SONUÇLAR KAYDEDİLİYOR...")
    
    # 1. Modelleri kaydet
    for result in results:
        model_name = result["model_name"].lower().replace(" ", "_")
        joblib.dump(result["model"], f'{model_name}_model.pkl')
        print(f"✓ {model_name}_model.pkl kaydedildi")
    
    # 2. Vektörleştiriciyi kaydet
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    print("✓ tfidf_vectorizer.pkl kaydedildi")
    
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
            "train_time_seconds": float(result["train_time"])
        })
    
    with open('model_results.json', 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    print("✓ model_results.json kaydedildi")
    
    # 4. En iyi modeli belirle
    best_model_idx = np.argmax([r["f1"] for r in results])
    best_model = results[best_model_idx]
    
    print(f"\n🏆 EN İYİ MODEL: {best_model['model_name']}")
    print(f"   F1-Score: {best_model['f1']:.4f}")
    print(f"   Accuracy: {best_model['accuracy']:.4f}")

def main():
    """Ana fonksiyon"""
    
    # 1. Verileri yükle
    X_train, X_test, y_train, y_test = load_data()
    
    # 2. TF-IDF vektörleştirme
    vectorizer, X_train_tfidf, X_test_tfidf = create_tfidf_features(X_train, X_test)
    
    # 3. Modelleri eğit
    print("\n🚀 MODEL EĞİTİMLERİ BAŞLIYOR...")
    
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
    
    # 4. Sonuçları görselleştir
    try:
        plot_results(results, vectorizer)
    except Exception as e:
        print(f"Görselleştirme hatası: {e}")
        print("Matplotlib kurulu değilse: pip install matplotlib seaborn")
    
    # 5. Modelleri ve sonuçları kaydet
    save_models_and_results(results, vectorizer)
    
    print("\n" + "=" * 70)
    print("✅ 3 ML MODELİ EĞİTİMİ TAMAMLANDI!")
    print("=" * 70)
    
    print("\n📁 OLUŞTURULAN DOSYALAR:")
    print("  Modeller:")
    print("    - naive_bayes_model.pkl")
    print("    - random_forest_model.pkl")
    print("    - svm_model.pkl")
    print("    - tfidf_vectorizer.pkl")
    print("  Görseller:")
    print("    - model_performance_comparison.png")
    print("    - roc_curves.png")
    print("    - confusion_matrices.png")
    print("  Veriler:")
    print("    - model_results.json")
    
    print("\n🎯 BİR SONRAKİ ADIM: MODEL ENTEGRASYONU VE YAZILIM GELİŞTİRME")
    print("=" * 70)

if __name__ == "__main__":
    main()