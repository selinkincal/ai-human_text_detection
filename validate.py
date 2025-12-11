import json

with open("arxiv_3000.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Toplam kayıt: {len(data)}")

# boş veya çok kısa özet kontrolü
short_or_empty = [d for d in data if len(d["summary"].split()) < 20]
print(f"Çok kısa / boş özet sayısı: {len(short_or_empty)}")

# duplicate özet kontrolü
summaries = [d["summary"] for d in data]
duplicates = len(summaries) - len(set(summaries))
print(f"Duplicate özet sayısı: {duplicates}")

# örnek veri göster
print("\nÖrnek bir giriş:")
print(data[0])

