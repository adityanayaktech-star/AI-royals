import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt

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
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_image(path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE))

def clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(2.0, (8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)

def preprocess(img):
    x = img.astype("float32") / 255.0
    mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
    x = (x - mean) / std
    return torch.tensor(x.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(DEVICE)

def gradcam(model, x, target_layer):
    acts, grads = [], []
    h1 = target_layer.register_forward_hook(lambda m, i, o: acts.append(o))
    h2 = target_layer.register_full_backward_hook(lambda m, gi, go: grads.append(go[0]))
    model.eval()
    out = model(x)
    probs = torch.softmax(out, dim=1)[0].detach().cpu().numpy()
    k = int(out.argmax(1))
    model.zero_grad()
    out[0, k].backward()
    h1.remove(); h2.remove()
    w = grads[0].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * acts[0]).sum(1))[0].detach().cpu().numpy()
    cam = cam / (cam.max() + 1e-8)
    return cam, probs

def analyze(model, target_layer, path, out="report.png", use_clahe=True):
    img = load_image(path)
    inp = clahe(img) if use_clahe else img
    cam, probs = gradcam(model, preprocess(inp), target_layer)
    heat = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    color = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET),
                         cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img, 0.55, color, 0.45, 0)

    k = int(np.argmax(probs)); conf = probs[k] * 100
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.5))
    ax[0].imshow(img); ax[0].set_title("Original")
    ax[1].imshow(heat, cmap="jet"); ax[1].set_title("Heatmap")
    ax[2].imshow(overlay); ax[2].set_title("Overlay")
    bars = ax[3].barh(CLASSES, probs * 100, color="steelblue")
    bars[k].set_color("crimson"); ax[3].set_xlim(0, 100)
    ax[3].set_title("Class probabilities (%)")
    for a in ax[:3]: a.axis("off")

    msg = f"Prediction: {CLASSES[k]} ({conf:.1f}%)"
    if conf < 60:
        msg += "  |  WARNING: Uncertain, consult a doctor"
    fig.suptitle(msg, fontsize=14, color="crimson" if conf < 60 else "black")
    fig.text(0.5, 0.01, "For research/educational use only. Not a medical diagnosis.",
             ha="center", fontsize=9)
    plt.tight_layout(rect=[0, 0.04, 1, 0.94])
    plt.savefig(out, dpi=150)
    plt.close()
    return CLASSES[k], conf


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    model_path = next((p for p in [os.path.join(here, "best.pth"), os.path.join(here, "..", "best.pth"), "best.pth"] if os.path.exists(p)), None)
    if not model_path:
        raise FileNotFoundError("best.pth not found")

    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    print("Model loaded successfully.")
    sample_img = "sample_retina.jpg"
    if os.path.exists(sample_img):
        print(analyze(model, model.layer4[-1], sample_img))
