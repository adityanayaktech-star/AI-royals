import json
import os
from collections import Counter

from sqlalchemy import create_engine, text

PASSWORD = "aditya06"
engine = create_engine(
    f"postgresql+psycopg2://postgres:{PASSWORD}@localhost:5432/retinalai"
)

ROOT = os.path.join("data", "rajbari_images", "Retinal Dataset")
rows = []
for folder, _, files in os.walk(ROOT):
    for file_name in files:
        if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(folder, file_name)
        label = os.path.basename(folder)
        aug = "aug" in path.lower()
        rows.append(
            {
                "image_path": path,
                "label": label,
                "is_augmented": aug,
                "source": "Rajbari (Mendeley)",
            }
        )

with engine.begin() as conn:
    for row in rows:
        conn.execute(
            text(
                """
                INSERT INTO images (image_path, label, is_augmented, source)
                VALUES (:image_path, :label, :is_augmented, :source)
                ON CONFLICT (image_path) DO NOTHING
                """
            ),
            row,
        )

with open("data/images.json", "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2)

print("Loaded:", len(rows))
print(Counter(row["label"] for row in rows))