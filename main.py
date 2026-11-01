import os
import sys
from pathlib import Path
import uvicorn

ROOT_DIR = Path(__file__).resolve().parent
APP_DIR = ROOT_DIR / "scripts" / "retinal-screening"

if __name__ == "__main__":
    print(f"Starting RetinaAI server from {APP_DIR} ...")
    print("Open http://127.0.0.1:8001 in your browser to view the retinal screening UI.")
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True, app_dir=str(APP_DIR))

