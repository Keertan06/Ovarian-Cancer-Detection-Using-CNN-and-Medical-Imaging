import os, io, time, uuid, json
import numpy as np
from PIL import Image
from flask import Flask, render_template_string, request, url_for

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import cv2

try:
    import pydicom
    HAS_PYDICOM = True
except Exception:
    HAS_PYDICOM = False

DEVICE = "cpu"           
IMG_SIZE = 224

BIN_MODEL_PATH   = "models/ovarian_cancer_resnet50_torch.pth"   
STAGE_MODEL_PATH = "models/ov_stage_resnet50_best.pth"          
STAGE_LABEL_MAP  = "models/ov_stage_label_map.json"             

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def read_image_bytes(b: bytes, filename: str) -> Image.Image:
    """Read PNG/JPG/DICOM bytes and return RGB PIL image."""
    name = filename.lower()
    if name.endswith(".dcm"):
        if not HAS_PYDICOM:
            raise RuntimeError("Install DICOM support first: pip install pydicom")
        ds = pydicom.dcmread(io.BytesIO(b), force=True)
        arr = ds.pixel_array.astype(np.float32)
        vmin, vmax = np.percentile(arr, (1, 99))
        if vmax <= vmin:
            vmax = vmin + 1.0
        arr = np.clip((arr - vmin) / (vmax - vmin), 0, 1) * 255.0
        arr = arr.astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    return Image.open(io.BytesIO(b)).convert("RGB")

def _load_rgb_from_path(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

def build_binary_model():
    if not os.path.isfile(BIN_MODEL_PATH):
        return None
    model = models.resnet50(weights=None)
    nf = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(nf, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 1),
        nn.Sigmoid()  
    )
    state = torch.load(BIN_MODEL_PATH, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval().to(DEVICE)
    return model

def build_stage_model():
    """
    Loads a ResNet50 4-class stage model.
    Supports heads:
      - Linear: fc.weight / fc.bias
      - Sequential with 128 or 256 hidden: fc.0.* (in=2048,out=128/256), fc.3.* (hidden->4)
    Falls back to non-strict when shapes differ.
    """
    if not os.path.isfile(STAGE_MODEL_PATH):
        return None

    state = torch.load(STAGE_MODEL_PATH, map_location="cpu")
    keys = set(state.keys())
    has_linear = ("fc.weight" in keys and "fc.bias" in keys)
    has_seq = any(k.startswith("fc.0.") for k in keys) or any(k.startswith("fc.3.") for k in keys)

    base = models.resnet50(weights=None)
    nf = base.fc.in_features

    if has_linear:
        base.fc = nn.Linear(nf, 4)
        base.load_state_dict(state, strict=True)
    elif has_seq:
        hid = None
        if "fc.0.weight" in state:
            hid = int(state["fc.0.weight"].shape[0])
        if not hid:
            hid = 128
        base.fc = nn.Sequential(
            nn.Linear(nf, hid),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hid, 4)
        )
        try:
            base.load_state_dict(state, strict=True)
        except RuntimeError:
            base.load_state_dict(state, strict=False)
    else:
        base.fc = nn.Sequential(
            nn.Linear(nf, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 4)
        )
        base.load_state_dict(state, strict=False)

    base.eval().to(DEVICE)
    return base

def target_layer(model):
    
    return model.layer4[-1].conv3

def _force_2d(a: np.ndarray) -> np.ndarray:
    """Ensure a is HxW (drop extra dims safely)."""
    a = np.asarray(a)
    a = np.squeeze(a)
    if a.ndim == 3:
        if a.shape[-1] == 1:
            a = a[..., 0]
        elif a.shape[-1] == 3:
            a = cv2.cvtColor(a.astype(np.float32), cv2.COLOR_RGB2GRAY)
        else:
            a = a.mean(axis=-1)
    return a

def generate_gradcam(cam_model: torch.nn.Module, img_tensor: torch.Tensor) -> np.ndarray:
    """
    Grad-CAM on a logit (pre-sigmoid) to avoid saturation.
    Returns HxW CAM in [0,1].
    """
    acts, grads = [], []

    def fwd_hook(_m, _i, o): acts.append(o)
    def bwd_hook(_m, gi, go): grads.append(go[0])

    tl = target_layer(cam_model)
    h1 = tl.register_forward_hook(fwd_hook)
    h2 = tl.register_backward_hook(bwd_hook)

    out = cam_model(img_tensor)          
    logit = out.view(1, -1)[0, 0]        

    loss = -logit                        
    cam_model.zero_grad(set_to_none=True)
    loss.backward()

    h1.remove(); h2.remove()

    A = acts[0].detach()                 
    dA = grads[0].detach()               
    w = dA.mean(dim=(1,2))             

    cam = torch.relu((w[None, :, None, None] * A).sum(dim=1))   
    cam = F.interpolate(cam, size=img_tensor.shape[-2:], mode="bilinear", align_corners=False)[0]
    cam = cam.cpu().numpy().astype(np.float32)

    cam = _force_2d(cam)                
    cam -= cam.min()
    denom = np.percentile(cam, 99.0) + 1e-8
    cam = np.clip(cam / denom, 0, 1)

   
    cam = cv2.GaussianBlur(cam, (31, 31), 0)
    cam = np.power(cam, 0.7)
    return cam

def overlay_heatmap(pil_img: Image.Image, cam: np.ndarray, alpha=0.65) -> Image.Image:
    base = pil_img.convert("RGB")
    W, H = base.size

    cam = _force_2d(cam)  
    cam = cv2.resize(cam.astype(np.float32), (W, H))

    
    gray = cv2.cvtColor(np.array(base), cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    _, body = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    body = cv2.morphologyEx(body, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    body = (body > 0).astype(np.float32)

    cam *= body                           

    heat = (np.clip(cam, 0, 1) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(np.array(base), 1 - alpha, heat, alpha, 0)
    return Image.fromarray(overlay)


BIN_MODEL = build_binary_model()

CAM_MODEL = None
if BIN_MODEL is not None:
    cam_body = models.resnet50(weights=None)
    cam_body.load_state_dict(BIN_MODEL.state_dict(), strict=False)  
    nf = cam_body.fc.in_features
    cam_body.fc = nn.Sequential(  
        nn.Linear(nf, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 1)
    )
    try:
        cam_body.load_state_dict(BIN_MODEL.state_dict(), strict=False)
    except Exception:
        pass
    cam_body.eval().to(DEVICE)
    CAM_MODEL = cam_body

STAGE_MODEL = build_stage_model()


if os.path.isfile(STAGE_LABEL_MAP):
    with open(STAGE_LABEL_MAP, "r") as f:
        _map = json.load(f)
    STAGE_IDX_TO_NAME = {int(k): v for k, v in _map.items()}
else:
    STAGE_IDX_TO_NAME = {0: "Stage I", 1: "Stage II", 2: "Stage III", 3: "Stage IV"}

HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ovarian Cancer Detection • Staging</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
<style>
  body { background:#0f172a; color:#e2e8f0; }
  .card{ background:#111827; border:1px solid #1f2937; border-radius:16px; }
  .form-control, .form-range { background:#0b1220; border:1px solid #1f2937; color:#e2e8f0; }
  .form-label { color:#ffffff !important; } /* make both labels white */
  .btn-primary{ background:#2563eb; border-color:#2563eb; }
  .btn-primary:hover{ background:#1d4ed8; }
  .btn-success{ background:#16a34a; border-color:#16a34a; }
  .btn-success:hover{ background:#15803d; }
  .btn-lg{ padding:.9rem 1.4rem; font-weight:600; border-radius:14px; }
  .muted{ color:#94a3b8; }
  img.preview{ border-radius:12px; border:1px solid #1f2937; }
  .badge-res{ font-size:1rem; padding:.6rem .9rem; }
</style>
</head>
<body>
<div class="container py-5">
  <div class="row justify-content-center">
    <div class="col-lg-10">
      <h1 class="fw-bold mb-2">🩻 Ovarian Cancer Detection</h1>
      <p class="muted mb-4">Upload a CT slice (PNG / JPG / DICOM). Choose either <b>Cancer Detection</b> or <b>Stage Prediction</b>.</p>

      <div class="card p-4 shadow-sm">
        <form method="post" enctype="multipart/form-data" class="row g-3">
          <input type="hidden" name="cached" value="{{ cached }}">  <!-- keep last image -->
          <div class="col-md-6">
            <label class="form-label">Image file</label>
            <input class="form-control" type="file" name="file" accept=".png,.jpg,.jpeg,.dcm">
          </div>
          <div class="col-md-6">
            <label class="form-label">Decision threshold (non-cancer): <span id="thrLabel" class="fw-semibold">{{thr}}</span></label>
            <input type="range" min="0.05" max="0.95" step="0.01" class="form-range" name="thr" value="{{thr}}" oninput="thrLabel.innerText=this.value">
          </div>
          <div class="col-12 d-flex gap-2">
            <button class="btn btn-primary btn-lg" type="submit" name="action" value="binary">🔎 Cancer Detection</button>
            <button class="btn btn-success btn-lg" type="submit" name="action" value="stage">🎯 Stage Prediction</button>
          </div>
        </form>

        {% if error %}
          <hr class="my-4"><div class="alert alert-danger">{{ error }}</div>
        {% endif %}

        {% if msg %}
        <hr class="my-4">
        <div class="row g-4 align-items-start">
          <div class="col-lg-5">
            <div class="mb-2">
              {% if label_badge %}
                <span class="badge {{ label_badge }} badge-res">{{ label_text }}</span>
              {% endif %}
            </div>
            <div class="muted">{{ msg }}</div>
            {% if saved %}
            <figure class="mt-3">
              <img class="preview img-fluid" src="{{ saved }}" alt="Preview">
              <figcaption class="muted mt-2">Preview</figcaption>
            </figure>
            {% endif %}
          </div>
          {% if cam_saved %}
          <div class="col-lg-7">
            <figure>
              <img class="preview img-fluid" src="{{ cam_saved }}" alt="Grad-CAM">
              <figcaption class="muted mt-2">Grad-CAM overlay</figcaption>
            </figure>
          </div>
          {% endif %}
        </div>
        {% endif %}
      </div>

      <p class="muted mt-3">
        {% if not bin_ready %}⚠ Binary model not found: {{ bin_path }}{% endif %}
        {% if not stage_ready %}<br>⚠ Stage model not found: {{ stage_path }}{% endif %}
      </p>
    </div>
  </div>
</div>
</body>
</html>
"""

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    thr = 0.50
    msg = None
    saved = None
    cam_saved = None
    label_badge = ""
    label_text = ""
    error = None
    cached = request.form.get("cached", "")  

    bin_ready = BIN_MODEL is not None
    stage_ready = STAGE_MODEL is not None

    if request.method == "POST":
        action = request.form.get("action", "binary")
        thr = float(request.form.get("thr", 0.50)) if request.form.get("thr") else 0.50
        f = request.files.get("file")

        try:
            
            if f and f.filename:
                pil = read_image_bytes(f.read(), f.filename)
                os.makedirs("static/cache", exist_ok=True)
                cached_name = f"cache_{int(time.time())}_{uuid.uuid4().hex[:6]}.png"
                cached_path = os.path.join("static/cache", cached_name)
                pil.save(cached_path, "PNG")
                cached = cached_path
            elif cached and os.path.isfile(cached):
                pil = _load_rgb_from_path(cached)
            else:
                raise RuntimeError("Please upload a PNG/JPG/DICOM image first.")

            
            os.makedirs("static", exist_ok=True)
            fname = f"preview_{int(time.time())}_{uuid.uuid4().hex[:6]}.jpg"
            ppath = os.path.join("static", fname)
            pil.resize((820, 820)).save(ppath, "JPEG", quality=92)
            saved = url_for('static', filename=fname)

            x = TRANSFORM(pil).unsqueeze(0).to(DEVICE)

            if action == "binary":
                if not bin_ready:
                    error = f"Binary model not found at {BIN_MODEL_PATH}"
                else:
                    with torch.no_grad():
                        p_nc = float(BIN_MODEL(x).item())  
                    p_c = 1.0 - p_nc

                    is_non_cancer = p_nc > thr
                    conf  = p_nc if is_non_cancer else p_c

                    label_badge = "badge bg-success" if is_non_cancer else "badge bg-danger"
                    label_text  = "PREDICTION: NON-CANCEROUS" if is_non_cancer else "PREDICTION: CANCEROUS"
                    msg = (
                        f"Confidence: {conf*100:.1f}% • "
                        f"P(cancer)={p_c:.4f} • P(non-cancer)={p_nc:.4f} • "
                        f"thr(non-cancer)={thr:.2f}"
                    )

                    if not is_non_cancer and CAM_MODEL is not None:
                        cam = generate_gradcam(CAM_MODEL, x)
                        cam[cam < 0.18] = 0.0
                        if cam.max() > 0:
                            cam /= cam.max()
                        cam_img = overlay_heatmap(pil, cam, alpha=0.65)
                        cname = f"cam_{int(time.time())}_{uuid.uuid4().hex[:6]}.jpg"
                        cpath = os.path.join("static", cname)
                        cam_img.resize((820, 820)).save(cpath, "JPEG", quality=92)
                        cam_saved = url_for('static', filename=cname)

            elif action == "stage":
                if not stage_ready:
                    error = f"Stage model not found at {STAGE_MODEL_PATH}"
                else:
                    if bin_ready:
                        with torch.no_grad():
                            p_nc = float(BIN_MODEL(x).item())
                        p_c = 1.0 - p_nc
                    else:
                        p_nc, p_c = 0.5, 0.5

                    if p_nc > thr:
                        label_badge = "badge bg-success"
                        label_text  = "NON-CANCER — stage not computed"
                        msg = (f"Binary gate: P(non-cancer)={p_nc:.4f} > thr={thr:.2f} "
                               f"(P(cancer)={p_c:.4f}). We do not estimate FIGO stage for normal/benign.")
                        cam_saved = None
                    else:
                        with torch.no_grad():
                            logits = STAGE_MODEL(x)
                            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                            idx = int(np.argmax(probs))
                            stage_name = STAGE_IDX_TO_NAME.get(idx, f"Class {idx}")
                            msg = " | ".join([f"{STAGE_IDX_TO_NAME.get(i, f'Class {i}')}: {probs[i]*100:.1f}%"
                                              for i in range(len(probs))])
                            label_badge = "badge bg-info"
                            label_text  = f"STAGE PREDICTION: {stage_name}"

                        try:
                            cam = generate_gradcam(STAGE_MODEL, x)
                            if cam.max() > 0:
                                cam /= cam.max()
                            cam_img = overlay_heatmap(pil, cam, alpha=0.65)
                            cname = f"cam_{int(time.time())}_{uuid.uuid4().hex[:6]}.jpg"
                            cpath = os.path.join("static", cname)
                            cam_img.resize((820, 820)).save(cpath, "JPEG", quality=92)
                            cam_saved = url_for('static', filename=cname)
                        except Exception:
                            cam_saved = None
            else:
                error = "Unknown action."

        except Exception as ex:
            error = f"Failed to process image: {ex}"

    return render_template_string(
        HTML,
        msg=msg,
        saved=saved,
        cam_saved=cam_saved,
        thr=thr,
        label_badge=label_badge,
        label_text=label_text,
        error=error,
        bin_ready=bin_ready,
        stage_ready=stage_ready,
        bin_path=BIN_MODEL_PATH,
        stage_path=STAGE_MODEL_PATH,
        cached=cached  
    )

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
