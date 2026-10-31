import requests
from sqlalchemy import create_engine, text

PASSWORD = "aditya06"
engine = create_engine(f"postgresql+psycopg2://postgres:{PASSWORD}@localhost:5432/retinalai")

KEYWORDS = ("fundus", "retinopathy", "glaucoma", "macular", "retinal image", "retinal dataset", "ophthalm")

r = requests.get("https://zenodo.org/api/records", params={
    "q": "fundus",
    "sort": "newest", "size": 25})
r.raise_for_status()
new = 0
with engine.begin() as conn:
    for h in r.json()["hits"]["hits"]:
        m = h["metadata"]
        if not any(keyword in m["title"].lower() for keyword in KEYWORDS):
            continue
        res = conn.execute(text("""
            INSERT INTO datasets_seen (zenodo_id, title, published, link)
            VALUES (:id, :title, :pub, :link)
            ON CONFLICT (zenodo_id) DO NOTHING"""),
            {"id": str(h["id"]), "title": m["title"],
             "pub": m.get("publication_date"), "link": h["links"]["self_html"]})
        if res.rowcount:
            new += 1
            print("NEW:", h["id"], "|", m["title"])

print("Naye datasets:", new)