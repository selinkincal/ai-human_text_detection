import requests
import csv
import time

def fetch_semantic_scholar_abstracts(query="machine learning", max_results=2000):
    abstracts = []
    url_template = "https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit=100&offset={offset}&fields=title,abstract"
    for offset in range(0, max_results, 100):
        url = url_template.format(query=query, offset=offset)
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Request failed at offset {offset}, retrying after 5 saniye...")
            time.sleep(5)
            response = requests.get(url)
        data = response.json()
        for paper in data.get("data", []):
            if paper.get("abstract"):
                abstracts.append({"source": "semantic_scholar", "title": paper["title"], "abstract": paper["abstract"]})
        time.sleep(0.5)  # Rate limit için
    return abstracts

print("Semantic Scholar verisi çekiliyor...")
semantic_data = fetch_semantic_scholar_abstracts()
print(f"Semantic Scholar’dan çekilen özet sayısı: {len(semantic_data)}")

with open("semantic_abstracts.csv", "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["source", "title", "abstract"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for item in semantic_data:
        writer.writerow(item)

print("Semantic Scholar verisi CSV’ye kaydedildi: semantic_abstracts.csv")
