import json

with open("ai_3000.json", "r", encoding="utf-8") as f:
    ai = json.load(f)
print("AI veri sayısı:", len(ai))

with open("arxiv_3000.json", "r", encoding="utf-8") as f:
    human = json.load(f)
print("İnsan veri sayısı:", len(human))
