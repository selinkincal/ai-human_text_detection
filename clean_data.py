import json
import re

def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.strip()

    # HTML ve LaTeX kalıntılarını temizle
    text = re.sub(r"<.*?>", " ", text)            # HTML tagleri
    text = re.sub(r"\\[a-zA-Z]+", " ", text)      # LaTeX komutları \alpha \begin vs.
    text = re.sub(r"\$.*?\$", " ", text)          # Matematik formülleri
    text = re.sub(r"[{}]", " ", text)             # Süslü parantezler

    # Boşlukları düzelt
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_file(input_path, output_path, is_ai=False):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned = []
    removed = 0

    for item in data:
        if is_ai:
            human_sum = clean_text(item["human_summary"])
            ai_sum = clean_text(item["ai_summary"])

            # Geçersiz ise atla
            if len(human_sum) < 50 or len(ai_sum) < 50:
                removed += 1
                continue

            cleaned.append({
                "title": item["title"],
                "human_summary": human_sum,
                "ai_summary": ai_sum
            })
        else:
            summary = clean_text(item["summary"])

            if len(summary) < 50:
                removed += 1
                continue

            cleaned.append({
                "title": item["title"],
                "summary": summary
            })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    print(f"Temizlendi → {output_path}")
    print(f"Kalan veri sayısı: {len(cleaned)}")
    print(f"Silinen veri sayısı: {removed}\n")


def main():
    print("📌 Human dataset temizleniyor...")
    clean_file("arxiv_3000.json", "clean_human.json")

    print("📌 AI dataset temizleniyor...")
    clean_file("ai_3000.json", "clean_ai.json", is_ai=True)

    print("✅ Temizleme tamamlandı!")


if __name__ == "__main__":
    main()
