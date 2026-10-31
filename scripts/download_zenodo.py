import requests, os

RECORD = "19169587"
os.makedirs("data", exist_ok=True)

rec = requests.get(f"https://zenodo.org/api/records/{RECORD}").json()
print("Title:", rec["metadata"]["title"])
print("License:", rec["metadata"].get("license"))

for f in rec["files"]:
    path = os.path.join("data", f["key"])
    print("Downloading", f["key"], f["size"] // 1024**2, "MB")
    with requests.get(f["links"]["self"], stream=True) as r:
        r.raise_for_status()
        with open(path, "wb") as out:
            for chunk in r.iter_content(1024 * 1024):
                out.write(chunk)
print("Done")