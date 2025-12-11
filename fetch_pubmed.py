import requests
import csv
import time

def fetch_pubmed_abstracts(term="machine learning", max_results=2000):
    abstracts = []
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    
    params = {"db":"pubmed","term":term,"retmax":max_results,"retmode":"json"}
    response = requests.get(base_url, params=params)
    id_list = response.json().get("esearchresult", {}).get("idlist", [])
    
    for i, pmid in enumerate(id_list):
        params_fetch = {"db":"pubmed","id":pmid,"retmode":"xml"}
        res = requests.get(fetch_url, params=params_fetch)
        if "<AbstractText>" in res.text:
            start = res.text.find("<AbstractText>")+14
            end = res.text.find("</AbstractText>")
            abstract_text = res.text[start:end].strip()
            abstracts.append({"source": "pubmed", "title": pmid, "abstract": abstract_text})
        if i % 50 == 0:
            time.sleep(1)
    return abstracts

print("PubMed verisi çekiliyor...")
pubmed_data = fetch_pubmed_abstracts()
print(f"PubMed’den çekilen özet sayısı: {len(pubmed_data)}")

with open("pubmed_abstracts.csv", "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["source", "title", "abstract"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for item in pubmed_data:
        writer.writerow(item)

print("PubMed verisi CSV’ye kaydedildi: pubmed_abstracts.csv")
