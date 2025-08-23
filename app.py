# app.py  (design-only, no model files required)
import os, io, json, uuid, math, random
from typing import Tuple, Dict, List

from flask import Flask, render_template, request, send_file, url_for
from PIL import Image, ImageFilter, ImageOps
import numpy as np

# ---------- config ----------
IMG_SIZE   = int(os.environ.get("IMG_SIZE", 320))
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "web_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Try to import your real labels if present; otherwise use a simple fallback set.
try:
    from chestray_labels import LABELS  # optional
except Exception:
    LABELS = [
        "Pneumonia",
        "Tuberculosis",
        "COPD",
        "Pleural Effusion",
        "Atelectasis",
        "No Finding",
    ]

# ---------- tiny helpers (design-only “mock” inference) ----------
def _softmax_like(xs: np.ndarray) -> np.ndarray:
    xs = xs - xs.max()
    e = np.exp(xs)
    p = e / (e.sum() + 1e-9)
    return p

def _random_scores(n: int) -> np.ndarray:
    # stable random-ish scores for demo
    raw = np.random.default_rng().normal(loc=0.0, scale=1.0, size=n)
    probs = _softmax_like(raw)  # 0..1 and sums to 1
    return probs

def _risk_from_conf(conf: float) -> str:
    if conf >= 0.80: return "HIGH"
    if conf >= 0.55: return "MEDIUM"
    return "LOW"

def build_card_fallback(scores: Dict[str, float]) -> Dict:
    # Highest-probability condition = “decision”
    top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_label, top_p = top[0]
    # “No Finding” logic for demo
    decision = "NEGATIVE" if top_label.lower() in ("no finding", "normal", "healthy") else "POSITIVE"
    risk_tier = _risk_from_conf(top_p)

    # Build a simple, readable card structure similar to your original
    card = {
        "decision": decision,
        "risk_tier": risk_tier,
        "top_findings": [{"label": k, "score": float(v)} for k, v in top[:5]],
        "scores": {k: float(v) for k, v in scores.items()},
        "meta": {
            "img_size": IMG_SIZE,
            "note": "Design-only demo output (no model loaded)."
        }
    }
    return card

def _gaussian_blob(h: int, w: int, cx: float, cy: float, sigma: float) -> np.ndarray:
    """Create a 2D Gaussian blob centered at (cx,cy) in [0,1] coords."""
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    X, Y = np.meshgrid(x, y)
    g = np.exp(-(((X - cx) ** 2) + ((Y - cy) ** 2)) / (2 * sigma ** 2))
    return g

def _fake_cam_overlay(img_rgb: Image.Image, intensity: float = 0.6) -> Image.Image:
    """Generate a fake Grad‑CAM-like overlay for demo/design purposes."""
    # Work on a resized copy
    img = img_rgb.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    w, h = img.size

    # Combine a few Gaussian blobs at random positions
    blobs = np.zeros((h, w), dtype=np.float32)
    num_blobs = random.randint(1, 3)
    for _ in range(num_blobs):
        cx = random.uniform(0.25, 0.75)
        cy = random.uniform(0.25, 0.75)
        sigma = random.uniform(0.08, 0.18)
        blobs += _gaussian_blob(h, w, cx, cy, sigma).astype(np.float32)

    # Normalize to [0,1]
    blobs -= blobs.min()
    if blobs.max() > 0:
        blobs /= blobs.max()

    # Create a red heatmap (R channel from blobs, subtle G/B)
    heat_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    heat_rgb[..., 0] = (255 * blobs).astype(np.uint8)              # red
    heat_rgb[..., 1] = (64 * np.sqrt(blobs)).astype(np.uint8)      # a little green
    heat_rgb[..., 2] = (32 * (blobs ** 0.25)).astype(np.uint8)     # a touch of blue
    heat = Image.fromarray(heat_rgb, mode="RGB")

    # Blend heatmap onto original
    overlay = Image.blend(img.convert("RGB"), heat, alpha=float(np.clip(intensity, 0.0, 1.0)))
    return overlay

def run_inference_and_cam_demo(pil_img: Image.Image) -> Tuple[Dict, Image.Image, Image.Image]:
    """
    Design-only pipeline:
      - Resize image
      - Generate random label scores
      - Build a simple screening card
      - Create a fake Grad‑CAM overlay
    """
    img_rgb = pil_img.convert("RGB")
    img_resized = img_rgb.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

    # Random demo scores
    probs = _random_scores(len(LABELS))
    scores = {lbl: float(p) for lbl, p in zip(LABELS, probs)}

    # Build fallback card
    card = build_card_fallback(scores)

    # Fake “Grad‑CAM”
    cam_pil = _fake_cam_overlay(img_rgb, intensity=0.55)

    return card, img_resized, cam_pil

# ---------- Flask ----------
app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    # The template will render the upload UI. Results render only if variables are present.
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("image")
    if not file or file.filename == "":
        return render_template("index.html", error="Please choose an image (JPG/PNG).")

    # Read with PIL
    try:
        pil = Image.open(file.stream)
    except Exception:
        return render_template("index.html", error="Could not read image. Use JPG/PNG.")

    # DESIGN-ONLY: run demo inference + fake Grad‑CAM
    card, img_resized, cam_pil = run_inference_and_cam_demo(pil)

    # Persist images so the template can display them
    uid = uuid.uuid4().hex[:8]
    orig_path = os.path.join(UPLOAD_DIR, f"{uid}_orig.jpg")
    cam_path  = os.path.join(UPLOAD_DIR, f"{uid}_cam.jpg")
    img_resized.save(orig_path, quality=92)
    cam_pil.save(cam_path, quality=92)

    # Small suspected diagnosis summary (top-1)
    top1 = max(card["scores"].items(), key=lambda kv: kv[1])
    suspected = f"{top1[0]} ({top1[1]:.2f})"

    return render_template(
        "index.html",
        orig_url=url_for("serve_file", path=os.path.basename(orig_path)),
        cam_url=url_for("serve_file", path=os.path.basename(cam_path)),
        suspected=suspected,
        decision=card["decision"],
        risk_tier=card["risk_tier"],
        top_findings=card["top_findings"],
        card_json=json.dumps(card, indent=2)
    )

@app.route("/files/<path:path>")
def serve_file(path):
    return send_file(os.path.join(UPLOAD_DIR, path))

if __name__ == "__main__":
    # No auto-reloader to keep things simple/stable for design preview
    app.run(host="0.0.0.0", port=5000, debug=False)
