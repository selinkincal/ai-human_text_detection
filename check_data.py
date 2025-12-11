import json

# Temizlenmiş dosyaları kontrol et
with open("clean_human.json", "r", encoding="utf-8") as f:
    human_data = json.load(f)
    print(f"clean_human.json: {len(human_data)} örnek")

with open("clean_ai.json", "r", encoding="utf-8") as f:
    ai_data = json.load(f)
    print(f"clean_ai.json: {len(ai_data)} örnek")

# İlk örneği göster
print("\nİlk human örneği:")
print(json.dumps(human_data[0], indent=2, ensure_ascii=False)[:500] + "...")

print("\nİlk AI örneği:")
print(json.dumps(ai_data[0], indent=2, ensure_ascii=False)[:500] + "...")