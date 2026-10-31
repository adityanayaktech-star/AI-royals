import streamlit as st, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, matplotlib.cm as cm
from PIL import Image
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

tf = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

@st.cache_resource
def load():
    m = models.resnet18(); m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
    m.load_state_dict(torch.load("best.pth", map_location="cpu")); return m.eval()
model = load()

def gradcam(x):
    f = {}
    def hook(m, i, o):
        o.retain_grad()
        f["a"] = o
    h = model.layer4[-1].register_forward_hook(hook)
    out = model(x)
    c = out.argmax(1).item()
    model.zero_grad()
    out[0, c].backward()
    h.remove()
    a = f["a"]
    w = a.grad.mean((2, 3), keepdim=True)
    cam = torch.relu((w * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear")[0, 0].detach().numpy()
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8), c, F.softmax(out, 1)[0].detach().numpy()
st.title("RetinaAI: Retinal Screening Assistant")
st.caption("Decision-support prototype. Does not replace an ophthalmologist.")
up = st.file_uploader("Upload retinal fundus image", type=["jpg", "jpeg", "png"])
if up:
    img = Image.open(up).convert("RGB")
    if st.button("Analyze"):
        x = tf(img).unsqueeze(0)
        cam, c, p = gradcam(x)
        base = np.array(img.resize((224, 224))) / 255.0
        over = (0.55 * base + 0.45 * cm.jet(cam)[..., :3]).clip(0, 1)
        c1, c2 = st.columns(2)
        c1.image(img, caption="Input", use_container_width=True)
        c2.image(over, caption="Grad-CAM heatmap", use_container_width=True)
        st.success(f"Prediction: {CLASSES[c]} (confidence {p[c]*100:.1f}%)")
        for i in np.argsort(-p)[:3]:
            st.write(f"{CLASSES[i]}: {p[i]*100:.1f}%")