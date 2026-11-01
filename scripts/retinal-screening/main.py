import base64, io, os, threading
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from matplotlib import cm
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

CLASSES = [
    "AdvancedEnd-stage Glaucoma",
    "Dry Age-Related Macular Degeneration",
    "Mild Diabetic Retinopathy",
    "Mild Glaucoma",
    "Moderate Diabetic Retinopathy",
    "Moderate Glaucoma",
    "No Age-Related Macular Degeneration (Healthy regarding AMD)",
    "Proliferative Diabetic Retinopathy (PDR)",
    "Severe Diabetic Retinopathy",
    "Severe Glaucoma",
    "Wet Age-Related Macular Degeneration",
]

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = next((p for p in [os.path.join(HERE, "best.pth"),
                               os.path.join(HERE, "..", "best.pth")]
                   if os.path.exists(p)), None)
if MODEL_PATH is None:
    raise FileNotFoundError("best.pth not found")

tf = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()
lock = threading.Lock()


def gradcam(x):
    f = {}
    def hook(m, i, o):
        o.retain_grad()
        f["a"] = o
    h = model.layer4[-1].register_forward_hook(hook)
    try:
        out = model(x)
        c = out.argmax(1).item()
        model.zero_grad()
        out[0, c].backward()
    finally:
        h.remove()
    a = f["a"]
    w = a.grad.mean((2, 3), keepdim=True)
    cam = torch.relu((w * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear")[0, 0].detach().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam, c, F.softmax(out, 1)[0].detach().numpy()


def overlay_b64(img, cam):
    base = np.array(img.resize((224, 224))) / 255.0
    over = (0.55 * base + 0.45 * cm.jet(cam)[..., :3]).clip(0, 1)
    buf = io.BytesIO()
    Image.fromarray((over * 255).astype("uint8")).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


app = FastAPI(title="RetinaAI API")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/")
def index():
    html_path = os.path.join(HERE, "retinal-screening.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"message": "RetinaAI API is running. Access /docs for API documentation."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(415, "Only JPG or PNG images are supported")
    data = file.file.read()
    if len(data) > 15 * 1024 * 1024:
        raise HTTPException(413, "Image too large")
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Could not read image")
    x = tf(img).unsqueeze(0)
    with lock, torch.enable_grad():
        cam, c, p = gradcam(x)
    return {
        "prediction": CLASSES[c],
        "confidence": round(float(p[c]), 4),
        "probabilities": {CLASSES[i]: round(float(p[i]), 4) for i in np.argsort(-p)},
        "heatmap": overlay_b64(img, cam),
    }


if __name__ == "__main__":
    import uvicorn
    app_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Starting RetinaAI server at http://127.0.0.1:8001 ...")
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True, app_dir=app_dir)
