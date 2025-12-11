import json
import subprocess
import time

MODEL_NAME = "mistral"   # veya "llama3.1:8b"
INPUT_FILE = "arxiv_3000.json"
OUTPUT_FILE = "ai_3000.json"
BLOCK_SIZE = 250  # Her 250 özetten sonra kaydedilecek

def ollama_generate(prompt):
    result = subprocess.run(
        ["ollama", "run", MODEL_NAME],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    return result.stdout.decode("utf-8")

def main():
    # Verileri yükle
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Daha önce kısmi çıktı varsa onu yükle
    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            output = json.load(f)
            start_index = len(output)
            print(f"📂 Önceki kayıttan devam ediliyor: {start_index}. özetten itibaren")
    except FileNotFoundError:
        output = []
        start_index = 0

    try:
        for i, item in enumerate(data[start_index:], start=start_index):
            print(f"📡 {i+1}/{len(data)} AI özeti üretiliyor...")

            prompt = f"""
Rewrite this scientific abstract in a new unique way, sounding like AI-generated text.
Do not shorten it. Keep similar length and structure.

Abstract:
{item['summary']}
"""
            ai_summary = ollama_generate(prompt).strip()

            output.append({
                "title": item["title"],
                "human_summary": item["summary"],
                "ai_summary": ai_summary
            })

            # Her BLOCK_SIZE özetten sonra kaydet
            if (i + 1) % BLOCK_SIZE == 0:
                with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                    json.dump(output, f, indent=2, ensure_ascii=False)
                print(f"💾 {i+1} özet kaydedildi, devam ediliyor...")

            time.sleep(0.2)

    except KeyboardInterrupt:
        # Ctrl+C ile durdurduğunda kalan kısmı kaydet
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n⏸ İşlem durduruldu, {len(output)} özet kaydedildi.")

    # İşlem tamamlandıysa son kaydı yap
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n✔ İşlem tamamlandı! {OUTPUT_FILE} oluşturuldu.")


if __name__ == "__main__":
    main()
