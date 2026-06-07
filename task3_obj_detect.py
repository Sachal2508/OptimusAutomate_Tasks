"""
Task 3: Object Detection App
CodeAlpha AI Internship
================================================
Model    : MobileNet SSD (pre-trained on VOC — 20 object classes)
Backend  : OpenCV DNN (no TensorFlow/PyTorch needed)
Features : Upload image → detect objects → draw bounding boxes with
           labels + confidence scores → download annotated result
UI       : Streamlit

Folder structure (REQUIRED):
    your_folder/
    ├── object_detection_app.py   ← this file
    └── models/
        ├── deploy.prototxt
        └── MobileNetSSD_deploy.caffemodel

Run with : streamlit run object_detection_app.py
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Object Detection App",
    page_icon="🔍",
    layout="wide",
)

st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }
.stApp { background-color: #0d1117; color: #e6edf3; }
.stButton > button {
    background: linear-gradient(135deg, #1f4073, #1a3a5c);
    color: white; border: 1px solid #58a6ff44;
    border-radius: 8px; font-weight: 600;
}
.stButton > button:hover { border-color: #58a6ff; }
</style>
""", unsafe_allow_html=True)

# ── MobileNet SSD Classes (VOC 20) ───────────────────────────────
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

np.random.seed(42)
COLORS = {cls: tuple(int(c) for c in np.random.randint(80, 255, 3)) for cls in CLASSES}

# ── Resolve model paths with fallback locations ──────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_model_paths():
    # Support both layouts:
    # 1) src/models/deploy.prototxt + MobileNetSSD_deploy.caffemodel
    # 2) src/deploy.prototxt + MobileNetSSD_deploy.caffemodel
    candidates = [
        os.path.join(SCRIPT_DIR, "models"),
        SCRIPT_DIR,
    ]
    for base_dir in candidates:
        prototxt = os.path.join(base_dir, "deploy.prototxt")
        caffemodel = os.path.join(base_dir, "MobileNetSSD_deploy.caffemodel")
        if os.path.isfile(prototxt) and os.path.isfile(caffemodel):
            return base_dir, prototxt, caffemodel
    return None, None, None


MODEL_DIR, PROTOTXT, CAFFEMODEL = resolve_model_paths()


def resolve_yolo_weights_path():
    candidates = [
        os.path.join(SCRIPT_DIR, "yolov8n.pt"),
        os.path.join(os.path.dirname(SCRIPT_DIR), "yolov8n.pt"),
    ]
    for weights_path in candidates:
        if os.path.isfile(weights_path):
            return weights_path
    return None


YOLO_WEIGHTS = resolve_yolo_weights_path()

if not MODEL_DIR and (YOLO is None or YOLO_WEIGHTS is None):
    st.error("❌ Model files not found!")
    st.markdown("### Setup Instructions")
    st.markdown(f"**Script is running from:** `{SCRIPT_DIR}`")
    st.markdown("**Checked locations:**")
    st.markdown(f"- `{os.path.join(SCRIPT_DIR, 'models')}`")
    st.markdown(f"- `{SCRIPT_DIR}`")
    st.markdown(f"- `{os.path.dirname(SCRIPT_DIR)}` (for `yolov8n.pt`) ")
    st.markdown("**Make sure your folder looks like one of these:**")
    st.code("""
your_folder/
├── object_detection_app.py    ← this script
└── models/
    ├── deploy.prototxt
    └── MobileNetSSD_deploy.caffemodel

OR

your_folder/
├── object_detection_app.py
├── deploy.prototxt
└── MobileNetSSD_deploy.caffemodel
    """)
    st.markdown("**Missing files:**")
    for f in ["deploy.prototxt", "MobileNetSSD_deploy.caffemodel", "yolov8n.pt"]:
        st.markdown(f"- ❌ `{f}`")
    st.markdown("Add either MobileNetSSD files or `yolov8n.pt` in parent/script folder.")
    st.stop()

if MODEL_DIR and os.path.normpath(MODEL_DIR) == os.path.normpath(SCRIPT_DIR):
    st.sidebar.info("Using model files from script directory.")

# ── Load model ────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)
    return net


@st.cache_resource
def load_yolo_model(weights_path: str):
    return YOLO(weights_path)

net = None
yolo_model = None
if MODEL_DIR:
    try:
        net = load_model()
        st.sidebar.success("✅ Model loaded (MobileNet SSD, 119 layers)")
    except Exception as e:
        st.sidebar.warning(f"MobileNet SSD unavailable: {e}")

if YOLO is not None and YOLO_WEIGHTS is not None:
    try:
        yolo_model = load_yolo_model(YOLO_WEIGHTS)
        st.sidebar.success("✅ Model loaded (YOLOv8n)")
    except Exception as e:
        st.sidebar.warning(f"YOLO unavailable: {e}")

if net is None and yolo_model is None:
    st.error("No usable detection model could be loaded.")
    st.stop()

# ── Detection function ────────────────────────────────────────────
def _run_inference_on_image(image_bgr: np.ndarray, confidence_thresh: float, net, offset=(0, 0)):
    h, w = image_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image_bgr, (300, 300)),
        scalefactor=0.007843,
        size=(300, 300),
        mean=127.5,
    )
    net.setInput(blob)
    detections = net.forward()

    results = []

    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < confidence_thresh:
            continue
        class_id = int(detections[0, 0, i, 1])
        if class_id >= len(CLASSES):
            continue
        label = CLASSES[class_id]
        if label == "background":
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        x_off, y_off = offset
        x1, x2 = x1 + x_off, x2 + x_off
        y1, y2 = y1 + y_off, y2 + y_off

        results.append({"label": label, "confidence": confidence, "box": (x1, y1, x2, y2)})

    return results


def _apply_nms(results, iou_threshold=0.35):
    if not results:
        return []

    filtered = []
    labels = sorted(set(item["label"] for item in results))
    for label in labels:
        label_items = [item for item in results if item["label"] == label]
        boxes = []
        scores = []
        for item in label_items:
            x1, y1, x2, y2 = item["box"]
            boxes.append([x1, y1, max(1, x2 - x1), max(1, y2 - y1)])
            scores.append(float(item["confidence"]))

        indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=iou_threshold)
        if len(indices) == 0:
            continue

        for idx in np.array(indices).flatten():
            filtered.append(label_items[int(idx)])

    return filtered


def detect_objects(image_bgr: np.ndarray, confidence_thresh: float, net, use_tiled_fallback: bool = True):
    h, w = image_bgr.shape[:2]
    all_results = _run_inference_on_image(image_bgr, confidence_thresh, net)

    if use_tiled_fallback and len(all_results) == 0 and min(h, w) >= 200:
        tile_h = max(200, h // 2)
        tile_w = max(200, w // 2)
        y_starts = [0, max(0, h - tile_h)]
        x_starts = [0, max(0, w - tile_w)]
        for y in y_starts:
            for x in x_starts:
                tile = image_bgr[y:y + tile_h, x:x + tile_w]
                if tile.size == 0:
                    continue
                tile_results = _run_inference_on_image(tile, confidence_thresh, net, offset=(x, y))
                all_results.extend(tile_results)

    results = _apply_nms(all_results, iou_threshold=0.35)
    annotated = image_bgr.copy()

    for item in sorted(results, key=lambda value: -value["confidence"]):
        label = item["label"]
        confidence = item["confidence"]
        x1, y1, x2, y2 = item["box"]

        color = COLORS[label]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        text = f"{label}: {confidence:.1%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(annotated, text, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    return annotated, results


def detect_objects_yolo(image_bgr: np.ndarray, confidence_thresh: float, yolo_model):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    prediction = yolo_model.predict(source=rgb, conf=confidence_thresh, verbose=False)
    if not prediction:
        return image_bgr.copy(), []

    result = prediction[0]
    names = result.names
    detections = []
    annotated = image_bgr.copy()

    if result.boxes is None:
        return annotated, detections

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
        confidence = float(box.conf[0].cpu().numpy())
        class_id = int(box.cls[0].cpu().numpy())
        label = names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id)

        color = COLORS.get(label, (0, 200, 255))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        text = f"{label}: {confidence:.1%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(annotated, text, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        detections.append({"label": label, "confidence": confidence, "box": (x1, y1, x2, y2)})

    return annotated, detections

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    available_backends = []
    if yolo_model is not None:
        available_backends.append("YOLOv8n")
    if net is not None:
        available_backends.append("MobileNet SSD")

    backend = st.selectbox(
        "Detection backend",
        options=available_backends,
        index=0,
        help="YOLOv8n is generally more accurate, especially for small/multi-object scenes.",
    )

    conf_threshold = st.slider("Confidence Threshold", 0.05, 0.95, 0.25, 0.05,
                               help="Lower = more detections  |  Higher = more precise")
    use_tiled_fallback = st.toggle(
        "Enhanced small-object mode",
        value=True,
        help="If full-image detection finds nothing, run detection on image tiles to recover small objects.",
        disabled=(backend != "MobileNet SSD"),
    )
    st.markdown("---")
    st.markdown("### 🏷️ Detectable Classes")
    for cls in CLASSES[1:]:
        st.markdown(f"- {cls}")
    st.markdown("---")
    st.markdown("### ℹ️ How it works")
    st.markdown("""
**MobileNet SSD** is a lightweight Single Shot MultiBox Detector
pre-trained on Pascal VOC (20 classes).

1. Image resized to 300×300
2. Forward pass through 119-layer network
3. Bounding boxes drawn with label + confidence
    """)

# ── Main UI ───────────────────────────────────────────────────────
st.markdown("<h1 style='text-align:center;color:#58a6ff;'>🔍 Object Detection App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#8b949e;'>MobileNet SSD · OpenCV DNN · 20 Object Classes · Bounding Boxes with Confidence Scores</p>", unsafe_allow_html=True)
st.markdown("---")

uploaded = st.file_uploader(
    "Upload an image (JPG or PNG)",
    type=["jpg", "jpeg", "png"],
)

if not uploaded:
    st.info("👆 Upload any photo containing people, animals, vehicles, or furniture to detect objects.")
    badge_html = " ".join([
        f'<span style="background:#21262d;border:1px solid #30363d;padding:3px 10px;'
        f'border-radius:12px;color:#8b949e;font-size:0.82rem;margin:2px;display:inline-block;">{cls}</span>'
        for cls in CLASSES[1:]
    ])
    st.markdown("**Detectable classes:** " + badge_html, unsafe_allow_html=True)

if uploaded:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("Could not read the image. Please upload a valid JPG or PNG.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📷 Original Image")
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

    with st.spinner("🔍 Detecting objects..."):
        if backend == "YOLOv8n":
            annotated_bgr, detections = detect_objects_yolo(img_bgr, conf_threshold, yolo_model)
        else:
            annotated_bgr, detections = detect_objects(img_bgr, conf_threshold, net, use_tiled_fallback=use_tiled_fallback)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    with col2:
        st.markdown("#### 🎯 Detected Objects")
        st.image(annotated_rgb, use_container_width=True)

    st.markdown("---")

    m1, m2, m3, m4 = st.columns(4)
    unique_labels = list(set(d["label"] for d in detections))
    avg_conf = np.mean([d["confidence"] for d in detections]) if detections else 0
    m1.metric("Objects Detected", len(detections))
    m2.metric("Unique Classes",   len(unique_labels))
    m3.metric("Avg Confidence",   f"{avg_conf:.1%}" if detections else "—")
    m4.metric("Threshold Used",   f"{conf_threshold:.0%}")

    if detections:
        st.markdown("#### 📋 Detection Results")
        import pandas as pd
        table = []
        for i, d in enumerate(sorted(detections, key=lambda x: -x["confidence"]), 1):
            x1, y1, x2, y2 = d["box"]
            table.append({
                "#": i,
                "Object":         d["label"].capitalize(),
                "Confidence":     f"{d['confidence']:.2%}",
                "Box (x1,y1)":    f"({x1}, {y1})",
                "Box (x2,y2)":    f"({x2}, {y2})",
                "Width × Height": f"{x2-x1} × {y2-y1} px",
            })
        st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

        buf = io.BytesIO()
        Image.fromarray(annotated_rgb).save(buf, format="PNG")
        st.download_button(
            label="⬇️ Download Annotated Image",
            data=buf.getvalue(),
            file_name="detected_objects.png",
            mime="image/png",
            use_container_width=True,
        )
    else:
        st.warning(
            f"No objects detected above {conf_threshold:.0%} confidence. "
            "Try lowering the threshold. If available, switch backend to YOLOv8n for collage/small objects."
        )