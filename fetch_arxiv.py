# fetch_arxiv.py
import feedparser
import time
import csv

def fetch_arxiv_abstracts(category="cs.LG", target=2000):
    abstracts = []
    base_url = "http://export.arxiv.org/api/query?search_query=cat:{}&start={}&max_results={}"
    per = 200
    for start in range(0, target, per):
        url = base_url.format(category, start, per)
        print("Fetching:", url)
        feed = feedparser.parse(url)
        if not feed.entries:
            print("No entries returned at start", start)
            time.sleep(2)
            continue
        for entry in feed.entries:
            abstracts.append({"source":"arxiv_api","title": entry.title, "abstract": entry.summary})
        time.sleep(1)  # nazik davran
    return abstracts

print("Arxiv verisi çekiliyor...")
data = fetch_arxiv_abstracts(category="cs", target=2000)  # cs kategorisinden genel çek
print("Arxiv çekildi:", len(data))
# CSV yaz
with open("arxiv_api_abstracts.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["source","title","abstract"])
    writer.writeheader()
    for r in data:
        writer.writerow(r)
print("arxiv_api_abstracts.csv oluşturuldu.")
