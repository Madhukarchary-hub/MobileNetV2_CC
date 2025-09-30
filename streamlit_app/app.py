# app.py
from pathlib import Path
import json
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# --- ArcGIS Online posting helpers (AFTER streamlit import) ---
import requests
from datetime import datetime, timezone

TOKEN_URL = "https://www.arcgis.com/sharing/rest/generateToken"

def _cfg():
    """Read ArcGIS secrets lazily so imports never fail."""
    s = st.secrets
    return (
        s.get("AGOL_USERNAME", ""),
        s.get("AGOL_PASSWORD", ""),
        s.get("FEATURE_LAYER_URL", "")
    )

def get_agol_token():
    user, pwd, _ = _cfg()
    if not user or not pwd:
        raise RuntimeError("Missing ArcGIS credentials in .streamlit/secrets.toml")
    r = requests.post(
        TOKEN_URL,
        data={
            "username": user,
            "password": pwd,
            "client": "requestip",   # token bound to server IP
            "expiration": 60,        # minutes
            "f": "json",
        },
        timeout=20,
    )
    r.raise_for_status()
    j = r.json()
    if "token" not in j:
        raise RuntimeError(f"Token error: {j}")
    return j["token"]

def post_feature(lat, lon, species, conf, address="", trap_id="", image_url="", model_version="mnv2_cc"):
    _, _, layer_url = _cfg()
    if not layer_url:
        raise RuntimeError("Missing FEATURE_LAYER_URL in .streamlit/secrets.toml")
    token = get_agol_token()
    feature = {
        "geometry": {"x": float(lon), "y": float(lat), "spatialReference": {"wkid": 4326}},
        "attributes": {
            "species_pred": species,
            "pred_conf": float(conf),
            "created_at": int(datetime.now(timezone.utc).timestamp() * 1000),  # epoch ms
            "address": address,
            "trap_id": trap_id,
            "image_url": image_url,
            "model_version": model_version,
        },
    }
    resp = requests.post(
        layer_url + "/addFeatures",
        data={"f": "json", "token": token, "features": json.dumps([feature])},
        timeout=20,
    )
    resp.raise_for_status()
    j = resp.json()
    ok = j.get("addResults", [{}])[0].get("success", False)
    if not ok:
        raise RuntimeError(j)
    return j["addResults"][0]["objectId"]


# ----------------- YOUR MODEL SETUP -----------------
# Go one level up to project root, then into models/
ROOT = Path(__file__).resolve().parents[1]   # project root
MODEL_DIR = ROOT / "models"

BASE = Path(__file__).resolve().parent
MODEL_FILES = [BASE / "mnv2_cc_portable_legacy.h5"]  # ✅ use only the re-saved keras model
LABELS_FILE = BASE / "labels.json"
META_FILE = BASE / "mnv2_cc_meta.json"
IMG_SIZE = (224, 224)

@st.cache_resource
def load_artifacts():
    # model
    model_path = next((p for p in MODEL_FILES if p.exists()), None)
    if model_path is None:
        raise FileNotFoundError(
            "Model not found. Looked for:\n" + "\n".join(str(p) for p in MODEL_FILES)
        )
    model = tf.keras.models.load_model(model_path, compile=False)

    # labels
    labels_raw = json.loads(LABELS_FILE.read_text(encoding="utf-8"))
    labels = labels_raw["labels"] if isinstance(labels_raw, dict) and "labels" in labels_raw else labels_raw

    # default unknown threshold from meta (fallback 0.80)
    thr = 0.80
    if META_FILE.exists():
        try:
            thr = float(json.loads(META_FILE.read_text(encoding="utf-8")).get("unknown_threshold", thr))
        except Exception:
            pass
    return model, labels, thr

def preprocess(img: Image.Image):
    img = img.convert("RGB").resize(IMG_SIZE)
    x = np.array(img, dtype=np.float32)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    return np.expand_dims(x, 0)

def predict_one(model, labels, img: Image.Image, thr: float, margin: float):
    x = preprocess(img)
    probs = model(x, training=False).numpy().squeeze()
    top_idx = int(np.argmax(probs))
    top_prob = float(probs[top_idx])
    label = labels[top_idx]

    # liberal rule: accept if above thr OR sufficiently ahead of #2
    top2 = float(np.partition(probs, -2)[-2])
    is_known = (top_prob >= thr) or ((top_prob - top2) >= margin)
    decision = label if is_known else "Unknown"

    topk = sorted(
        [(labels[i], float(probs[i])) for i in range(len(labels))],
        key=lambda t: t[1],
        reverse=True
    )[:4]
    return decision, top_prob, topk

# ----------------- UI -----------------
st.set_page_config(page_title="BuzzWatch_cc", layout="centered")
st.title("BuzzWatch_cc")

model, labels, default_thr = load_artifacts()

with st.sidebar:
    st.header("Settings")
    thr = st.slider("Unknown threshold (top prob)", 0.50, 0.95, float(default_thr), 0.01)
    margin = st.slider("Liberal margin (top1 - top2)", 0.00, 0.20, 0.05, 0.01)
    st.caption("If top1 < threshold and not far ahead of top2 by this margin → **Unknown**")

st.write("Upload a mosquito photo or use the camera (mobile works).")

up = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
cam = st.camera_input("Camera")
img_file = cam or up

if img_file:
    image = Image.open(img_file)
    st.image(image, caption="Input", use_column_width=True)

    decision, top_prob, topk = predict_one(model, labels, image, thr, margin)
    st.subheader(f"Decision: **{decision}**")
    st.metric("Top probability", f"{top_prob:.3f}")
    st.write("Top scores:")
    st.table([{"class": n, "prob": f"{p:.3f}"} for n, p in topk])

    # --- ArcGIS form (inside this block so variables exist) ---
    with st.form("send_arcgis"):
        c1, c2 = st.columns(2)
        with c1:
            lat = st.number_input("Latitude", value=27.800000, format="%.6f")
        with c2:
            lon = st.number_input("Longitude", value=-97.396400, format="%.6f")
        addr = st.text_input("Address (optional)")
        trap = st.text_input("Trap ID (optional)")
        submitted = st.form_submit_button("Send to ArcGIS Dashboard")

    if submitted:
        try:
            species_to_post = decision  # can be "Unknown" or predicted label
            oid = post_feature(
                lat, lon, species_to_post, top_prob,
                address=addr, trap_id=trap, image_url="", model_version="mnv2_cc"
            )
            st.success(f"Posted to ArcGIS! OBJECTID={oid} • {species_to_post} ({top_prob:.2f})")
            st.info("Open your ArcGIS web map/dashboard and refresh to see the new point.")
        except Exception as e:
            st.error(f"Failed to post to ArcGIS: {e}")
else:
    st.info("Choose a file or open the camera.")
