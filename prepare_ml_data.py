import json
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import numpy as np

def combine_and_prepare_data():
    print("=" * 60)
    print("           VERİLERİ BİRLEŞTİRME VE ML HAZIRLIĞI")
    print("=" * 60)
    
    # 1. Temizlenmiş verileri yükle
    print("\n📥 VERİLER YÜKLENİYOR...")
    with open("clean_human.json", "r", encoding="utf-8") as f:
        human_data = json.load(f)
    
    with open("clean_ai.json", "r", encoding="utf-8") as f:
        ai_data = json.load(f)
    
    print(f"✓ Human verileri: {len(human_data)} örnek")
    print(f"✓ AI verileri: {len(ai_data)} örnek")
    
    # 2. Human verilerini hazırla (label: 0 = Human)
    print("\n🧑 HUMAN VERİLERİ HAZIRLANIYOR...")
    human_samples = []
    for i, item in enumerate(human_data):
        human_samples.append({
            "text": item["summary"],
            "label": 0,  # 0 = Human
            "source": "arxiv",
            "title": item["title"][:100]  # Başlık kısmını sakla (opsiyonel)
        })
    
    # 3. AI verilerini hazırla (label: 1 = AI)
    print("🤖 AI VERİLERİ HAZIRLANIYOR...")
    ai_samples = []
    for i, item in enumerate(ai_data):
        # Sadece AI tarafından üretilen özetleri kullan
        ai_samples.append({
            "text": item["ai_summary"],
            "label": 1,  # 1 = AI
            "source": "ai_generated",
            "title": item["title"][:100]
        })
    
    # 4. EK OPTİYON: AI dataset'inden HUMAN özetlerini de kullanabiliriz
    # (Daha fazla human verisi için - opsiyonel)
    """
    print("➕ AI DOSYASINDAN HUMAN ÖZETLERİ EKLENİYOR...")
    extra_human_samples = []
    for i, item in enumerate(ai_data):
        extra_human_samples.append({
            "text": item["human_summary"],
            "label": 0,  # Bu da human
            "source": "human_from_ai_dataset",
            "title": item["title"][:100]
        })
    
    human_samples.extend(extra_human_samples)
    print(f"  Eklenen human örnek: {len(extra_human_samples)}")
    """
    
    # 5. Tüm verileri birleştir
    print("\n🔄 VERİLER BİRLEŞTİRİLİYOR...")
    all_samples = human_samples + ai_samples
    random.seed(42)  # Tekrarlanabilirlik için
    random.shuffle(all_samples)
    
    # 6. DataFrame oluştur
    df = pd.DataFrame(all_samples)
    
    print(f"\n📊 TOPLAM VERİ SETİ:")
    print(f"  Toplam örnek sayısı: {len(df)}")
    
    # 7. Etiket dağılımı
    print("\n🎯 ETİKET DAĞILIMI:")
    label_counts = df["label"].value_counts().sort_index()
    for label, count in label_counts.items():
        label_name = "Human" if label == 0 else "AI"
        percentage = (count / len(df)) * 100
        print(f"  {label_name} ({label}): {count} örnek (%{percentage:.1f})")
    
    # 8. Metin uzunlukları analizi
    print("\n📏 METİN UZUNLUKLARI ANALİZİ:")
    df["text_length"] = df["text"].apply(len)
    
    print("\n  TÜM VERİLER:")
    print(f"    Ortalama: {df['text_length'].mean():.0f} karakter")
    print(f"    Minimum: {df['text_length'].min():.0f} karakter")
    print(f"    Maksimum: {df['text_length'].max():.0f} karakter")
    print(f"    Standart Sapma: {df['text_length'].std():.0f} karakter")
    
    print("\n  HUMAN METİNLERİ:")
    human_df = df[df["label"] == 0]
    print(f"    Ortalama: {human_df['text_length'].mean():.0f} karakter")
    print(f"    Min-Max: {human_df['text_length'].min():.0f} - {human_df['text_length'].max():.0f}")
    
    print("\n  AI METİNLERİ:")
    ai_df = df[df["label"] == 1]
    print(f"    Ortalama: {ai_df['text_length'].mean():.0f} karakter")
    print(f"    Min-Max: {ai_df['text_length'].min():.0f} - {ai_df['text_length'].max():.0f}")
    
    # 9. Train-Test Split (%80-%20)
    print("\n✂️  TRAIN-TEST AYIRMA (%80 Train, %20 Test)...")
    X = df["text"]
    y = df["label"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y  # Sınıf dağılımını koru
    )
    
    print(f"  Eğitim seti: {len(X_train)} örnek")
    print(f"  Test seti: {len(X_test)} örnek")
    
    # 10. CSV olarak kaydet
    print("\n💾 DOSYALAR KAYDEDİLİYOR...")
    
    # Tam dataset
    df.to_csv("full_dataset.csv", index=False, encoding="utf-8")
    
    # Train ve test setleri
    train_df = pd.DataFrame({"text": X_train, "label": y_train})
    test_df = pd.DataFrame({"text": X_test, "label": y_test})
    
    train_df.to_csv("train_data.csv", index=False, encoding="utf-8")
    test_df.to_csv("test_data.csv", index=False, encoding="utf-8")
    
    print("✓ full_dataset.csv - Tüm veriler")
    print("✓ train_data.csv - Eğitim verileri")
    print("✓ test_data.csv - Test verileri")
    
    # 11. Örnekler göster
    print("\n👁️  ÖRNEK VERİLER (Eğitim setinden 2 örnek):")
    for i in range(min(2, len(train_df))):
        label = train_df.iloc[i]["label"]
        label_name = "Human" if label == 0 else "AI"
        text_preview = train_df.iloc[i]["text"][:200] + "..." if len(train_df.iloc[i]["text"]) > 200 else train_df.iloc[i]["text"]
        print(f"\n  [{i+1}] {label_name} ({label}):")
        print(f"     {text_preview}")
    
    # 12. Dataset istatistikleri dosyası
    print("\n📈 İSTATİSTİKLER DOSYASI OLUŞTURULUYOR...")
    stats = {
        "total_samples": len(df),
        "human_samples": len(df[df["label"] == 0]),
        "ai_samples": len(df[df["label"] == 1]),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "avg_text_length": float(df["text_length"].mean()),
        "min_text_length": int(df["text_length"].min()),
        "max_text_length": int(df["text_length"].max()),
        "human_avg_length": float(human_df["text_length"].mean()),
        "ai_avg_length": float(ai_df["text_length"].mean())
    }
    
    with open("dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print("✓ dataset_stats.json - İstatistikler kaydedildi")
    
    print("\n" + "=" * 60)
    print("✅ VERİ HAZIRLAMA BAŞARIYLA TAMAMLANDI!")
    print("=" * 60)
    print("\n🎯 BİR SONRAKİ ADIM: 3 ML MODELİ EĞİTİMİ")
    print("\nOluşturulan dosyalar:")
    print("  - full_dataset.csv    : Tüm veri seti")
    print("  - train_data.csv      : Eğitim için")
    print("  - test_data.csv       : Test için")
    print("  - dataset_stats.json  : İstatistikler")
    print("\nToplam örnek sayısı:", len(df))
    print("=" * 60)

if __name__ == "__main__":
    combine_and_prepare_data()