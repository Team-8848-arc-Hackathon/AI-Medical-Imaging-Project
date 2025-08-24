import os, io, json, uuid, logging
from datetime import datetime
from typing import List

from flask import Flask, render_template, request, send_file, url_for, jsonify
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

# ---------- Try to import your real modules; fall back if missing ----------
try:
    from model_chestray import ChestRayNet
    from chestray_labels import LABELS
    from report_builder import build_card
    REAL_MODEL_AVAILABLE = True
except Exception as e:
    logging.warning(f"[BOOT] Using dummy model because imports failed: {e}")
    REAL_MODEL_AVAILABLE = False

    class ChestRayNet(torch.nn.Module):
        def __init__(self, backbone="efficientnet_b0", pretrained=False): super().__init__()
        def forward(self, x): return torch.rand((x.shape[0], 14))
        # simple conv so Grad-CAM has a target
        def gradcam_target_layer(self): return torch.nn.Conv2d(3, 8, kernel_size=1)

    LABELS = ["Atelectasis","Cardiomegaly","Consolidation","Edema","Effusion",
              "Emphysema","Fibrosis","Hernia","Infiltration","Mass","Nodule",
              "Pleural Thickening","Pneumonia","Pneumothorax"]

    def build_card(probs_or_scores, thresholds_path=None, lang="en", patient_id="—", frontal_view=True):
        scores = {LABELS[i]: float(probs_or_scores[i]) for i in range(len(LABELS))}
        top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        decision = "Needs attention" if top[0][1] > 0.5 else "Likely normal"
        return {
            "decision": decision,
            "risk_tier": "High" if top[0][1] > 0.7 else ("Moderate" if top[0][1] > 0.5 else "Low"),
            "headline": "Automated screening summary",
            "scores": scores,
            "summary_en": {
                "impression": f"Top suspected: {top[0][0]} with score {top[0][1]:.2f}.",
                "consider": ([f"Also consider {top[1][0]}", f"Consider {top[2][0]}"] if len(top)>=3 else [])
            },
        }

# Grad‑CAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

# ------------ config ------------
WEIGHTS_PATH = os.environ.get("MODEL_WEIGHTS", "outputs/chestray_best.pt")
THRESH_PATH  = os.environ.get("THRESHOLDS", None)
IMG_SIZE     = int(os.environ.get("IMG_SIZE", 320))
UPLOAD_DIR   = os.environ.get("UPLOAD_DIR", "web_uploads")
USE_GOOGLE_TRANSLATE = os.environ.get("USE_GOOGLE_TRANSLATE", "1") == "1"

os.makedirs(UPLOAD_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("respiscan")

# ------------ device/model ------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChestRayNet(backbone="efficientnet_b0", pretrained=False)

# Load weights only if present and real model available
if REAL_MODEL_AVAILABLE and os.path.isfile(WEIGHTS_PATH):
    try:
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
        logger.info(f"[BOOT] Loaded weights: {WEIGHTS_PATH}")
    except Exception as e:
        logger.warning(f"[BOOT] Could not load weights ({WEIGHTS_PATH}): {e} (results may be random).")
else:
    if REAL_MODEL_AVAILABLE:
        logger.warning(f"[BOOT] MODEL_WEIGHTS not found at {WEIGHTS_PATH} (results may be random).")
    else:
        logger.warning("[BOOT] Running with dummy model (UI demo mode). Results are random.")

model.eval().to(device)

TFM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ------------ helpers ------------
def _translate_texts(texts: List[str], target: str = "ne") -> List[str]:
    if not texts or not USE_GOOGLE_TRANSLATE:
        return texts
    try:
        from google.cloud import translate_v2 as translate
        client = translate.Client()
        out = []
        for t in texts:
            try:
                r = client.translate(t, target_language=target, format_='text')
                out.append(r.get('translatedText', t))
            except Exception:
                out.append(t)
        return out
    except Exception as e:
        logger.warning(f"Translate fallback: {e}")
        return texts

def run_inference_and_cam(pil_img: Image.Image):
    img_rgb = pil_img.convert("RGB")
    t = TFM(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.sigmoid(model(t))[0].cpu().numpy()

    # Grad‑CAM on top‑1
    try:
        top_idx = int(np.argmax(probs))
        target_layer = model.gradcam_target_layer()
        cam = GradCAM(model=model, target_layers=[target_layer])
        heat = cam(input_tensor=t, targets=[ClassifierOutputTarget(top_idx)])[0]
        h, w = heat.shape
        img_resized = img_rgb.resize((w, h))
        rgb = np.asarray(img_resized).astype(np.float32) / 255.0
        overlay = show_cam_on_image(rgb, heat, use_rgb=True)
        overlay_pil = Image.fromarray(overlay)
    except Exception as e:
        logger.warning(f"GradCAM fallback: {e}")
        img_resized = img_rgb.resize((IMG_SIZE, IMG_SIZE))
        overlay_pil = img_resized

    return probs, img_resized, overlay_pil

def build_pdf(uid: str, patient: dict, card: dict, orig_path: str, cam_path: str) -> str:
    pdf_path = os.path.join(UPLOAD_DIR, f"{uid}_report.pdf")
    c = canvas.Canvas(pdf_path, pagesize=A4)
    w, h = A4

    c.setFillColorRGB(0.06, 0.49, 0.42); c.rect(0, h-2*cm, w, 2*cm, stroke=0, fill=1)
    c.setFillColorRGB(1, 1, 1); c.setFont("Helvetica-Bold", 16)
    c.drawString(2*cm, h-1.2*cm, "RespiScan AI – Screening Report")

    y = h-3.2*cm
    c.setFillColorRGB(0,0,0); c.setFont("Helvetica", 11)
    c.drawString(2*cm, y, f"Patient: {patient.get('name','—')}  |  Age: {patient.get('age','—')}  |  Sex: {patient.get('sex','—')}")
    y -= 0.6*cm
    c.drawString(2*cm, y, f"ID: {patient.get('id','—')}  |  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    y -= 1.0*cm; c.setFont("Helvetica-Bold", 12); c.drawString(2*cm, y, "Summary")
    y -= 0.5*cm; c.setFont("Helvetica", 11)
    c.drawString(2*cm, y, f"Decision: {card.get('decision','—')}  |  Risk tier: {card.get('risk_tier','—')}")
    y -= 0.5*cm
    c.drawString(2*cm, y, f"Headline: {card.get('headline','—')[:90]}")

    y -= 0.8*cm; c.setFont("Helvetica-Bold", 12); c.drawString(2*cm, y, "Top Findings")
    y -= 0.5*cm; c.setFont("Helvetica", 11)
    for lbl, sc in sorted(card.get('scores', {}).items(), key=lambda kv: kv[1], reverse=True)[:3]:
        c.drawString(2.2*cm, y, f"• {lbl}: {sc:.2f}"); y -= 0.45*cm

    y -= 0.2*cm
    try: c.drawImage(orig_path, 2*cm, y-7.0*cm, width=7.5*cm, height=7.0*cm, preserveAspectRatio=True, mask='auto')
    except Exception: pass
    try: c.drawImage(cam_path, 10.5*cm, y-7.0*cm, width=7.5*cm, height=7.0*cm, preserveAspectRatio=True, mask='auto')
    except Exception: pass

    c.setFont("Helvetica-Oblique", 9)
    c.drawString(2*cm, 1.5*cm, "This is decision support for health professionals, not a diagnosis.")
    c.showPage(); c.save()
    return pdf_path

# ------------ Flask ------------
app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("image")
    if not file or file.filename == "":
        return render_template("index.html", error="Please choose an image (JPG/PNG).")

    lang = request.form.get("lang", "en")  # 'en' or 'np'
    patient = {
        "name": request.form.get("patient_name", "Unknown"),
        "age": request.form.get("patient_age", "—"),
        "sex": request.form.get("patient_sex", "—"),
        "id": request.form.get("patient_id", uuid.uuid4().hex[:6].upper())
    }

    try:
        pil = Image.open(file.stream)
    except Exception:
        return render_template("index.html", error="Could not read image. Use JPG/PNG.")

    probs, img_resized, cam_pil = run_inference_and_cam(pil)

    card = build_card(
        probs_or_scores=probs,
        thresholds_path=THRESH_PATH,
        lang=("np" if lang == "np" else "en"),
        patient_id=patient["id"],
        frontal_view=True
    )

    impression_en = card.get("summary_en", {}).get("impression", "")
    consider_en = card.get("summary_en", {}).get("consider", [])
    if lang == "np":
        translated = _translate_texts([impression_en] + consider_en, target="ne")
        if translated:
            card["summary_np"] = {
                "impression": translated[0],
                "consider": translated[1:] if len(translated) > 1 else []
            }

    uid = uuid.uuid4().hex[:8]
    orig_path = os.path.join(UPLOAD_DIR, f"{uid}_orig.jpg")
    cam_path  = os.path.join(UPLOAD_DIR, f"{uid}_cam.jpg")
    img_resized.save(orig_path, quality=92)
    cam_pil.save(cam_path, quality=92)

    pdf_path = build_pdf(uid, patient, card, orig_path, cam_path)

    top3 = sorted(card["scores"].items(), key=lambda kv: kv[1], reverse=True)[:3]
    suspected = f"{top3[0][0]} ({top3[0][1]:.2f})" if top3 else "—"

    return render_template(
        "index.html",
        orig_url=url_for("serve_file", path=os.path.basename(orig_path)),
        cam_url=url_for("serve_file", path=os.path.basename(cam_path)),
        pdf_url=url_for("serve_file", path=os.path.basename(pdf_path)),
        decision=card.get("decision","—"),
        risk_tier=card.get("risk_tier","—"),
        suspected=suspected,
        headline=card.get("headline",""),
        summary_en=card.get("summary_en", {}),
        summary_np=card.get("summary_np", None),
        patient=patient,
        card_json=json.dumps(card, indent=2, ensure_ascii=False)
    )

@app.route("/files/<path:path>")
def serve_file(path):
    return send_file(os.path.join(UPLOAD_DIR, path))

@app.route("/translate", methods=["POST"])
def translate_api():
    data = request.get_json(force=True)
    texts = data.get("texts", [])
    target = data.get("target", "ne")
    return jsonify({"texts": _translate_texts(texts, target)})

@app.route("/feedback", methods=["POST"])
def feedback():
    msg = request.get_json(force=True).get("message", "").strip()
    if not msg:
        return jsonify({"ok": False, "error": "Empty message"}), 400
    with open(os.path.join(UPLOAD_DIR, "feedback.log"), "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}] {msg}\n")
    return jsonify({"ok": True})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
