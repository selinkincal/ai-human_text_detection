import requests
import xmltodict
import json

def get_arxiv_data(max_results):
    base_url = "http://export.arxiv.org/api/query?"
    query = "search_query=all:cs"
    url = f"{base_url}{query}&start=0&max_results={max_results}"

    print(f"📡 API isteği gönderiliyor: {url}")
    response = requests.get(url)
    data = xmltodict.parse(response.text)

    entries = data["feed"].get("entry", [])
    print(f"✔ {len(entries)} makale alındı.")

    results = []
    for entry in entries:
        summary = entry.get("summary", "").strip()
        title = entry.get("title", "").strip()
        published = entry.get("published", "")
        authors = entry.get("author", [])
        
        if isinstance(authors, list):
            authors = [a["name"] for a in authors]
        else:
            authors = [authors["name"]]

        results.append({
            "title": title,
            "summary": summary,
            "published": published,
            "authors": authors,
        })

    file_name = f"arxiv_{max_results}.json"

    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"📁 Dosya kaydedildi: {file_name}")

# =====================
# ÇALIŞTIR
# =====================
get_arxiv_data(3000)

