"""
BirdVision AI — Streamlit frontend for the VGG7 Bird Species Classifier
=========================================================================

This app is a thin inference/demo layer on top of the existing, already
trained VGG7_imgclassi project. It does NOT retrain, redefine, or alter
the model, its weights, or its evaluation results in any way.

Expected project layout (unchanged):

    VGG7_imgclassi/
    ├── app.py                      <- this file
    ├── models/best_vgg7_birds.pth  <- existing trained weights (untouched)
    ├── results/...                 <- existing evaluation artifacts
    └── src/vgg7.py                 <- existing model architecture (imported, not copied)

IMPORTANT — one assumption you may need to adjust:
    This app imports the VGG7 class from `src/vgg7.py` and instantiates it as
    `VGG7(num_classes=20)`. If your actual class has a different constructor
    signature (e.g. no `num_classes` arg, or a different name), open
    `load_model()` below and adjust the single line marked with "ADAPT HERE".
    Everything else (preprocessing, inference, UI) works independently of that.

Run with:
    streamlit run app.py
"""

import os
import sys
import time

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# Paths (all relative to this file, never hard-coded to a user's machine)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_vgg7_birds.pth")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CONFUSION_MATRIX_PATH = os.path.join(RESULTS_DIR, "final_confusion_matrix.png")
INCORRECT_PREDICTIONS_PATH = os.path.join(RESULTS_DIR, "incorrect_predictions.png")
PER_CLASS_METRICS_PATH = os.path.join(RESULTS_DIR, "per_class_metrics.png")

if SRC_DIR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import the EXISTING architecture. We never redefine VGG7 here.
try:
    from src.vgg7 import VGG7
    VGG7_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - surfaced in the UI instead
    VGG7 = None
    VGG7_IMPORT_ERROR = str(exc)

# ---------------------------------------------------------------------------
# Fixed, documented evaluation results (do not edit — these are the actual
# reported numbers from the existing project's evaluation run).
# ---------------------------------------------------------------------------
METRICS = {
    "num_classes": 20,
    "val_accuracy": 87.00,
    "test_accuracy": 85.00,
    "test_images": 100,
    "correct": 85,
    "incorrect": 15,
    "precision": 87.81,
    "recall": 85.00,
    "f1": 84.99,
}

# Fallback class list — matches the alphabetical order PyTorch's ImageFolder
# assigns by default. If the checkpoint stores its own class list (see
# load_model), that takes priority over this fallback.
DEFAULT_CLASS_NAMES = [
    "ABBOTTS BABBLER",
    "ABBOTTS BOOBY",
    "ABYSSINIAN GROUND HORNBILL",
    "AFRICAN CROWNED CRANE",
    "AFRICAN EMERALD CUCKOO",
    "AFRICAN FIREFINCH",
    "AFRICAN OYSTER CATCHER",
    "AFRICAN PIED HORNBILL",
    "AFRICAN PYGMY GOOSE",
    "ALBATROSS",
    "ALBERTS TOWHEE",
    "ALEXANDRINE PARAKEET",
    "ALPINE CHOUGH",
    "ALTAMIRA YELLOWTHROAT",
    "AMERICAN AVOCET",
    "AMERICAN BITTERN",
    "AMERICAN COOT",
    "AMERICAN FLAMINGO",
    "AMERICAN GOLDFINCH",
    "AMERICAN KESTREL",
]

LOW_CONFIDENCE_THRESHOLD = 50.0  # purely a UI cue, not a scientific claim


# ---------------------------------------------------------------------------
# Model loading, preprocessing, prediction
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    """Load the existing trained VGG7 model exactly once per session."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if VGG7 is None:
        return None, DEFAULT_CLASS_NAMES, device, (
            f"Could not import VGG7 from src/vgg7.py: {VGG7_IMPORT_ERROR}"
        )

    if not os.path.exists(MODEL_PATH):
        return None, DEFAULT_CLASS_NAMES, device, (
            f"Model weights not found at '{MODEL_PATH}'. "
            "Make sure best_vgg7_birds.pth is inside the models/ folder."
        )

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
    except Exception as exc:
        return None, DEFAULT_CLASS_NAMES, device, f"Failed to read checkpoint file: {exc}"

    # A checkpoint may be a plain state_dict, or a dict bundling extra info
    # (state_dict + class names + training config, etc). Handle both.
    class_names = DEFAULT_CLASS_NAMES
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("class_names", "classes", "idx_to_class"):
            if key in checkpoint:
                stored = checkpoint[key]
                if isinstance(stored, dict):
                    class_names = [stored[i] for i in sorted(stored, key=lambda k: int(k))]
                else:
                    class_names = list(stored)
                break
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state_dict = checkpoint[key]
                break

    # ADAPT HERE if src/vgg7.py's VGG7 constructor differs from this signature.
    try:
        model = VGG7(num_classes=len(class_names))
    except TypeError:
        model = VGG7()

    try:
        model.load_state_dict(state_dict)
    except Exception as exc:
        return None, class_names, device, (
            f"Loaded checkpoint but could not apply it to VGG7's weights: {exc}"
        )

    model.to(device)
    model.eval()
    return model, class_names, device, None


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Apply the same preprocessing used at evaluation time (no augmentation)."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image.convert("RGB")).unsqueeze(0)


def predict_image(model, image_tensor, class_names, device, top_k: int = 3):
    """Run inference and return the top-k (label, probability%) predictions."""
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
    k = min(top_k, len(class_names))
    top_probs, top_idxs = torch.topk(probabilities, k=k)
    return [
        (class_names[idx.item()], prob.item() * 100.0)
        for prob, idx in zip(top_probs, top_idxs)
    ]


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&display=swap');

        :root {
            --forest-deep: #163527;
            --forest: #1F5B3F;
            --forest-mid: #2E7D57;
            --forest-soft: #6E9884;
            --cream: #F7F4EC;
            --card: #FFFFFF;
            --gold: #B08A4E;
            --ink: #1B2B22;
            --ink-muted: #5B6B60;
            --line: #E4DFCF;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, sans-serif;
            color: var(--ink);
        }

        .stApp {
            background: var(--cream);
        }

        h1, h2, h3, .bv-display {
            font-family: 'Fraunces', serif;
            color: var(--forest-deep);
            letter-spacing: -0.01em;
        }

        #MainMenu, footer, header {visibility: hidden;}
        .block-container {padding-top: 1.5rem; max-width: 1100px;}

        /* Hero */
        .bv-hero {
            background: linear-gradient(135deg, var(--forest-deep) 0%, var(--forest) 55%, var(--forest-mid) 100%);
            border-radius: 24px;
            padding: 3.2rem 3rem;
            color: #F4F1E6;
            margin-bottom: 2rem;
            position: relative;
            overflow: hidden;
        }
        .bv-hero::after {
            content: "";
            position: absolute;
            right: -60px; top: -60px;
            width: 260px; height: 260px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0) 70%);
        }
        .bv-hero h1 {
            color: #FBF9F2;
            font-size: 2.7rem;
            margin: 0.4rem 0 0.6rem 0;
        }
        .bv-hero p.bv-subtitle {
            font-size: 1.15rem;
            color: #D8E5DC;
            max-width: 640px;
            margin-bottom: 0.9rem;
        }
        .bv-hero p.bv-desc {
            color: #C3D3C8;
            max-width: 620px;
            font-size: 0.98rem;
        }
        .bv-badge {
            display: inline-block;
            background: rgba(255,255,255,0.12);
            border: 1px solid rgba(255,255,255,0.25);
            color: #F4F1E6;
            padding: 0.35rem 0.9rem;
            border-radius: 999px;
            font-size: 0.82rem;
            margin-bottom: 1rem;
            letter-spacing: 0.02em;
        }

        /* Cards */
        .bv-card {
            background: var(--card);
            border-radius: 18px;
            padding: 1.6rem 1.8rem;
            box-shadow: 0 6px 24px rgba(22, 53, 39, 0.07);
            border: 1px solid var(--line);
            margin-bottom: 1.2rem;
        }
        .bv-metric {
            background: var(--card);
            border-radius: 16px;
            padding: 1.3rem 1rem;
            text-align: center;
            box-shadow: 0 4px 16px rgba(22, 53, 39, 0.06);
            border: 1px solid var(--line);
        }
        .bv-metric .bv-metric-value {
            font-family: 'Fraunces', serif;
            font-size: 1.8rem;
            color: var(--forest-deep);
            font-weight: 600;
        }
        .bv-metric .bv-metric-label {
            color: var(--ink-muted);
            font-size: 0.85rem;
            margin-top: 0.15rem;
        }

        /* Prediction result */
        .bv-result {
            background: linear-gradient(180deg, #FFFFFF 0%, #F4F8F5 100%);
            border-radius: 20px;
            padding: 2.2rem;
            text-align: center;
            border: 1px solid var(--line);
            box-shadow: 0 10px 30px rgba(22, 53, 39, 0.09);
        }
        .bv-result .bv-eyebrow {
            color: var(--ink-muted);
            font-size: 0.9rem;
            margin-bottom: 0.5rem;
        }
        .bv-result .bv-species {
            font-family: 'Fraunces', serif;
            font-size: 2.1rem;
            color: var(--forest-deep);
            margin: 0.2rem 0 0.6rem 0;
            text-transform: capitalize;
        }
        .bv-result .bv-confidence {
            color: var(--forest-mid);
            font-weight: 600;
            font-size: 1.05rem;
        }

        .bv-bar-track {
            background: #EAE6D8;
            border-radius: 999px;
            height: 10px;
            width: 100%;
            overflow: hidden;
            margin: 0.35rem 0 0.9rem 0;
        }
        .bv-bar-fill {
            background: linear-gradient(90deg, var(--forest-mid), var(--gold));
            height: 100%;
            border-radius: 999px;
        }
        .bv-rank-row {
            display: flex;
            justify-content: space-between;
            font-size: 0.92rem;
            color: var(--ink);
            margin-bottom: 0.1rem;
        }
        .bv-rank-row .bv-rank-pct {
            color: var(--ink-muted);
            font-variant-numeric: tabular-nums;
        }

        .bv-section-title {
            margin-top: 2.6rem;
            margin-bottom: 0.9rem;
        }
        .bv-note {
            background: #FBF3E4;
            border: 1px solid #EBD8AE;
            color: #7A5A1E;
            border-radius: 12px;
            padding: 0.8rem 1rem;
            font-size: 0.92rem;
        }
        .bv-error {
            background: #FBEAE7;
            border: 1px solid #ECC1B8;
            color: #8C3A2A;
            border-radius: 12px;
            padding: 0.8rem 1rem;
            font-size: 0.92rem;
        }
        .bv-step {
            text-align: center;
            padding: 0.6rem;
        }
        .bv-step .bv-step-num {
            font-family: 'Fraunces', serif;
            font-size: 1.4rem;
            color: var(--gold);
            font-weight: 600;
        }
        .bv-step .bv-step-label {
            color: var(--ink);
            font-weight: 500;
            margin-top: 0.2rem;
        }
        .bv-tech-chip {
            display: inline-block;
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 0.5rem 1rem;
            margin: 0.25rem;
            font-size: 0.9rem;
            color: var(--ink);
        }
        .bv-footer {
            margin-top: 3rem;
            padding-top: 1.6rem;
            border-top: 1px solid var(--line);
            color: var(--ink-muted);
            font-size: 0.9rem;
        }
        .stButton>button {
            background: var(--forest-deep);
            color: #F4F1E6;
            border-radius: 12px;
            border: none;
            padding: 0.6rem 1.6rem;
            font-weight: 600;
            transition: background 0.2s ease;
        }
        .stButton>button:hover {
            background: var(--forest-mid);
            color: #FFFFFF;
        }
        [data-testid="stFileUploader"] {
            border-radius: 16px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# UI sections
# ---------------------------------------------------------------------------
def render_hero():
    st.markdown(
        """
        <div class="bv-hero">
            <div class="bv-badge">VGG7 &nbsp;•&nbsp; PyTorch &nbsp;•&nbsp; 20 species</div>
            <h1>BirdVision AI</h1>
            <p class="bv-subtitle">AI-powered bird species classification using a custom VGG7 deep learning model.</p>
            <p class="bv-desc">Upload a bird image and let our computer vision model identify the species in seconds.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_upload_and_predict(model, class_names, device, model_error):
    st.markdown('<h2 class="bv-section-title">Identify a bird</h2>', unsafe_allow_html=True)

    if model_error:
        st.markdown(f'<div class="bv-error">{model_error}</div>', unsafe_allow_html=True)
        return

    left, right = st.columns([1, 1], gap="large")

    with left:
        with st.container(border=True):
            st.markdown("**Upload a bird image**")
            st.caption("PNG, JPG or JPEG • Recommended image size: 224 × 224 or larger")
            uploaded_file = st.file_uploader(
                " ", type=["png", "jpg", "jpeg"], label_visibility="collapsed"
            )

            image = None
            if uploaded_file is not None:
                try:
                    image = Image.open(uploaded_file)
                    st.image(image, use_container_width=True, caption="Uploaded image")
                except Exception:
                    st.markdown(
                        '<div class="bv-error">This file doesn\'t look like a valid image. '
                        'Please upload a PNG or JPEG.</div>',
                        unsafe_allow_html=True,
                    )

            predict_clicked = st.button("Identify Bird", type="primary", disabled=image is None)

    with right:
        with st.container(border=True):
            if uploaded_file is None:
                st.markdown("**Prediction**")
                st.caption("Upload an image on the left, then click *Identify Bird* to see the result here.")
            elif predict_clicked and image is not None:
                with st.spinner("Running the VGG7 model..."):
                    start = time.time()
                    tensor = preprocess_image(image)
                    results = predict_image(model, tensor, class_names, device, top_k=3)
                    elapsed = time.time() - start
                render_prediction_result(results, elapsed)
            else:
                st.markdown("**Ready to predict**")
                st.caption("Click *Identify Bird* to run the model on your uploaded image.")


def render_prediction_result(results, elapsed_seconds):
    top_label, top_conf = results[0]

    st.markdown(
        f"""
        <div class="bv-result">
            <div class="bv-eyebrow">Prediction</div>
            <div class="bv-species">{top_label.title()}</div>
            <div class="bv-confidence">{top_conf:.1f}% confidence</div>
            <div class="bv-bar-track" style="margin-top:0.9rem;">
                <div class="bv-bar-fill" style="width:{min(top_conf, 100):.1f}%;"></div>
            </div>
            <div style="color:var(--ink-muted); font-size:0.85rem;">Model: VGG7 · Inference time: {elapsed_seconds:.2f}s</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if top_conf < LOW_CONFIDENCE_THRESHOLD:
        st.markdown(
            '<div class="bv-note" style="margin-top:0.8rem;">Low-confidence prediction — '
            "the image may be difficult for the model to classify.</div>",
            unsafe_allow_html=True,
        )

    st.markdown('<h3 style="margin-top:1.6rem;">Top predictions</h3>', unsafe_allow_html=True)
    for rank, (label, conf) in enumerate(results, start=1):
        st.markdown(
            f"""
            <div class="bv-rank-row">
                <span>{rank}. {label.title()}</span>
                <span class="bv-rank-pct">{conf:.1f}%</span>
            </div>
            <div class="bv-bar-track">
                <div class="bv-bar-fill" style="width:{min(conf, 100):.1f}%;"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_about_model():
    st.markdown('<h2 class="bv-section-title">About the model</h2>', unsafe_allow_html=True)
    cols = st.columns(5)
    cards = [
        (str(METRICS["num_classes"]), "Bird species"),
        (f"{METRICS['test_accuracy']:.0f}%", "Test accuracy"),
        (f"{METRICS['val_accuracy']:.0f}%", "Validation accuracy"),
        ("VGG7", "CNN architecture"),
        ("PyTorch", "Framework"),
    ]
    for col, (value, label) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="bv-metric">
                    <div class="bv-metric-value">{value}</div>
                    <div class="bv-metric-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.caption(
        f"Test accuracy ({METRICS['test_accuracy']:.0f}%) was measured on a held-out set of "
        f"{METRICS['test_images']} images. Validation accuracy ({METRICS['val_accuracy']:.0f}%) "
        "was measured during training on the validation split."
    )


def render_performance():
    st.markdown('<h2 class="bv-section-title">Model performance</h2>', unsafe_allow_html=True)
    cols = st.columns(4)
    perf_cards = [
        (f"{METRICS['val_accuracy']:.2f}%", "Validation accuracy"),
        (f"{METRICS['test_accuracy']:.2f}%", "Test accuracy"),
        (f"{METRICS['precision']:.2f}%", "Macro precision"),
        (f"{METRICS['recall']:.2f}%", "Macro recall"),
    ]
    for col, (value, label) in zip(cols, perf_cards):
        with col:
            st.markdown(
                f"""
                <div class="bv-metric">
                    <div class="bv-metric-value">{value}</div>
                    <div class="bv-metric-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.caption(f"Macro F1-score: {METRICS['f1']:.2f}%")


def render_error_analysis():
    st.markdown('<h2 class="bv-section-title">Model error analysis</h2>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="bv-card">
            The model correctly classified {METRICS['correct']} of {METRICS['test_images']}
            test images and misclassified {METRICS['incorrect']}. Some of the incorrect
            predictions were confused between visually similar bird classes; see the
            image below for specific examples from the test set.
        </div>
        """,
        unsafe_allow_html=True,
    )
    if os.path.exists(INCORRECT_PREDICTIONS_PATH):
        st.image(INCORRECT_PREDICTIONS_PATH, use_container_width=True)
    else:
        st.markdown(
            '<div class="bv-note">Error analysis image not found at '
            f'<code>results/incorrect_predictions.png</code>. Run '
            "<code>src/error_analysis.py</code> to generate it.</div>",
            unsafe_allow_html=True,
        )


def render_confusion_matrix():
    st.markdown('<h2 class="bv-section-title">Confusion matrix</h2>', unsafe_allow_html=True)
    st.caption("The confusion matrix shows how predictions are distributed across the 20 bird species.")
    if os.path.exists(CONFUSION_MATRIX_PATH):
        st.image(CONFUSION_MATRIX_PATH, use_container_width=True)
    else:
        st.markdown(
            '<div class="bv-note">Confusion matrix image not found at '
            f'<code>results/final_confusion_matrix.png</code>. Run '
            "<code>src/confusion_matrix.py</code> to generate it.</div>",
            unsafe_allow_html=True,
        )


def render_how_it_works():
    st.markdown('<h2 class="bv-section-title">How it works</h2>', unsafe_allow_html=True)
    steps = [
        ("01", "Upload image"),
        ("02", "Image preprocessing"),
        ("03", "VGG7 prediction"),
        ("04", "Bird species + confidence"),
    ]
    cols = st.columns(4)
    for col, (num, label) in zip(cols, steps):
        with col:
            st.markdown(
                f"""
                <div class="bv-step">
                    <div class="bv-step-num">{num}</div>
                    <div class="bv-step-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_tech_stack():
    st.markdown('<h2 class="bv-section-title">Tech stack</h2>', unsafe_allow_html=True)
    chips = ["Python", "PyTorch", "Torchvision", "CUDA", "Streamlit", "Scikit-learn", "Matplotlib"]
    st.markdown(
        "".join(f'<span class="bv-tech-chip">{chip}</span>' for chip in chips),
        unsafe_allow_html=True,
    )


def render_footer():
    github_url = os.environ.get("BIRDVISION_GITHUB_URL", "#")
    st.markdown(
        f"""
        <div class="bv-footer">
            <strong>BirdVision AI</strong><br/>
            Built with PyTorch and a custom VGG7 CNN.<br/>
            <a href="{github_url}" target="_blank">GitHub repository</a> ·
            20-species bird classifier · VGG7 architecture · Streamlit demo interface
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="BirdVision AI",
        page_icon="🦜",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_css()
    render_hero()

    model, class_names, device, model_error = load_model()
    render_upload_and_predict(model, class_names, device, model_error)

    render_about_model()
    render_performance()
    render_error_analysis()
    render_confusion_matrix()
    render_how_it_works()
    render_tech_stack()
    render_footer()


if __name__ == "__main__":
    main()
