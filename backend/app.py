import os
import sys
import time

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from supabase import create_client


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

SRC_DIR = os.path.join(PROJECT_ROOT, "src")
MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "best_vgg7_birds.pth"
)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

CONFUSION_MATRIX_PATH = os.path.join(
    RESULTS_DIR,
    "final_confusion_matrix.png"
)

INCORRECT_PREDICTIONS_PATH = os.path.join(
    RESULTS_DIR,
    "incorrect_predictions.png"
)

PER_CLASS_METRICS_PATH = os.path.join(
    RESULTS_DIR,
    "per_class_metrics.png"
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORT VGG7
# ============================================================

try:
    from src.vgg7 import VGG7
    VGG7_IMPORT_ERROR = None

except Exception as exc:
    VGG7 = None
    VGG7_IMPORT_ERROR = str(exc)


# ============================================================
# MODEL INFORMATION
# ============================================================

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

LOW_CONFIDENCE_THRESHOLD = 50.0
# ---------------------------------------------------------------------------
# Supabase prediction history
# ---------------------------------------------------------------------------

@st.cache_resource
def get_supabase():
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]

        return create_client(url, key)

    except Exception as exc:
        print(f"Supabase connection error: {exc}")
        return None


def save_prediction(image_name, predicted_class, confidence):
    try:
        supabase = get_supabase()

        if supabase is None:
            return False

        supabase.table("prediction_history").insert({
            "image_name": image_name,
            "predicted_class": predicted_class,
            "confidence": float(confidence)
        }).execute()

        return True

    except Exception as exc:
        print(f"Could not save prediction: {exc}")
        return False


def get_prediction_history():
    try:
        supabase = get_supabase()

        if supabase is None:
            return []

        response = (
            supabase
            .table("prediction_history")
            .select("*")
            .order("created_at", desc=True)
            .execute()
        )

        return response.data or []

    except Exception as exc:
        print(f"Could not load history: {exc}")
        return []


def clear_prediction_history():
    try:
        supabase = get_supabase()

        if supabase is None:
            return False

        supabase.table("prediction_history").delete().neq("id", 0).execute()

        return True

    except Exception as exc:
        print(f"Could not clear history: {exc}")
        return False

# ---------------------------------------------------------------------------
# Supabase prediction history
# ---------------------------------------------------------------------------

@st.cache_resource
def get_supabase():
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except Exception:
        return None


def save_prediction(image_name, predicted_class, confidence):
    """Save one prediction to Supabase."""
    try:
        supabase = get_supabase()

        if supabase is None:
            return False

        supabase.table("prediction_history").insert({
            "image_name": image_name,
            "predicted_class": predicted_class,
            "confidence": float(confidence),
        }).execute()

        return True

    except Exception as exc:
        print(f"Could not save prediction history: {exc}")
        return False


def get_prediction_history():
    """Load prediction history from Supabase."""
    try:
        supabase = get_supabase()

        if supabase is None:
            return []

        response = (
            supabase
            .table("prediction_history")
            .select("*")
            .order("created_at", desc=True)
            .execute()
        )

        return response.data or []

    except Exception as exc:
        print(f"Could not load prediction history: {exc}")
        return []


def clear_prediction_history():
    """Delete all prediction history."""
    try:
        supabase = get_supabase()

        if supabase is None:
            return False

        # Delete every row.
        supabase.table("prediction_history").delete().neq("id", 0).execute()

        return True

    except Exception as exc:
        print(f"Could not clear prediction history: {exc}")
        return False


# ============================================================
# SESSION HISTORY
# ============================================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if VGG7 is None:
        return (
            None,
            DEFAULT_CLASS_NAMES,
            device,
            f"Could not import VGG7: {VGG7_IMPORT_ERROR}"
        )

    if not os.path.exists(MODEL_PATH):
        return (
            None,
            DEFAULT_CLASS_NAMES,
            device,
            f"Model weights not found: {MODEL_PATH}"
        )

    try:

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=device
        )

    except Exception as exc:

        return (
            None,
            DEFAULT_CLASS_NAMES,
            device,
            f"Failed to load model: {exc}"
        )

    class_names = DEFAULT_CLASS_NAMES
    state_dict = checkpoint

    if isinstance(checkpoint, dict):

        for key in (
            "class_names",
            "classes",
            "idx_to_class"
        ):

            if key in checkpoint:

                stored = checkpoint[key]

                if isinstance(stored, dict):

                    class_names = [
                        stored[i]
                        for i in sorted(
                            stored,
                            key=lambda x: int(x)
                        )
                    ]

                else:

                    class_names = list(stored)

                break

        for key in (
            "model_state_dict",
            "state_dict",
            "model"
        ):

            if (
                key in checkpoint
                and isinstance(checkpoint[key], dict)
            ):

                state_dict = checkpoint[key]
                break

    try:

        model = VGG7(
            num_classes=len(class_names)
        )

    except TypeError:

        model = VGG7()

    try:

        model.load_state_dict(state_dict)

    except Exception as exc:

        return (
            None,
            class_names,
            device,
            f"Could not apply model weights: {exc}"
        )

    model.to(device)
    model.eval()

    return (
        model,
        class_names,
        device,
        None
    )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    transform = transforms.Compose([

        transforms.Resize(
            (224, 224)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],

            std=[
                0.229,
                0.224,
                0.225
            ]
        )
    ])

    return transform(
        image.convert("RGB")
    ).unsqueeze(0)


# ============================================================
# PREDICTION
# ============================================================

def predict_image(
    model,
    image_tensor,
    class_names,
    device,
    top_k=3
):

    image_tensor = image_tensor.to(device)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = F.softmax(
            outputs,
            dim=1
        )[0]

    k = min(
        top_k,
        len(class_names)
    )

    top_probs, top_idxs = torch.topk(
        probabilities,
        k=k
    )

    results = []

    for prob, idx in zip(
        top_probs,
        top_idxs
    ):

        results.append(
            (
                class_names[idx.item()],
                prob.item() * 100
            )
        )

    return results


# ============================================================
# CSS
# ============================================================

def inject_css():

    st.markdown(
        """
        <style>

        @import url(
        'https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&display=swap'
        );

        :root {
            --forest-deep: #163527;
            --forest: #1F5B3F;
            --forest-mid: #2E7D57;
            --cream: #F7F4EC;
            --card: #FFFFFF;
            --gold: #B08A4E;
            --ink: #1B2B22;
            --ink-muted: #5B6B60;
            --line: #E4DFCF;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: var(--ink);
        }

        .stApp {
            background: var(--cream);
        }

        h1, h2, h3 {
            font-family: 'Fraunces', serif;
            color: var(--forest-deep);
        }

        #MainMenu,
        footer,
        header {
            visibility: hidden;
        }

        .block-container {
            padding-top: 1.5rem;
            max-width: 1100px;
        }

        /* HERO */

        .bv-hero {
            background:
                linear-gradient(
                    135deg,
                    var(--forest-deep),
                    var(--forest),
                    var(--forest-mid)
                );

            border-radius: 24px;
            padding: 3rem;
            color: white;
            margin-bottom: 2rem;
        }

        .bv-hero h1 {
            color: white;
            font-size: 2.7rem;
            margin-bottom: 0.5rem;
        }

        .bv-subtitle {
            font-size: 1.15rem;
            color: #D8E5DC;
        }

        .bv-desc {
            color: #C3D3C8;
        }

        .bv-badge {
            display: inline-block;
            background: rgba(255,255,255,0.12);
            border: 1px solid rgba(255,255,255,0.25);
            padding: 0.4rem 0.9rem;
            border-radius: 999px;
            font-size: 0.82rem;
        }

        /* SECTION */

        .bv-section-title {
            margin-top: 2.5rem;
            margin-bottom: 1rem;
        }

        /* REAL CARDS */

        .result-card {
            background: white;
            border-radius: 18px;
            padding: 1.5rem;
            border: 1px solid var(--line);
            box-shadow:
                0 6px 24px rgba(22,53,39,0.07);
        }

        /* RESULT */

        .bv-result {
            background:
                linear-gradient(
                    180deg,
                    #FFFFFF,
                    #F4F8F5
                );

            border-radius: 20px;
            padding: 2rem;
            text-align: center;
            border: 1px solid var(--line);
        }

        .bv-eyebrow {
            color: var(--ink-muted);
        }

        .bv-species {
            font-family: 'Fraunces', serif;
            font-size: 2rem;
            color: var(--forest-deep);
            margin: 0.4rem 0;
        }

        .bv-confidence {
            color: var(--forest-mid);
            font-weight: 600;
        }

        .bv-bar-track {
            background: #EAE6D8;
            border-radius: 999px;
            height: 10px;
            width: 100%;
            overflow: hidden;
            margin: 0.4rem 0 1rem 0;
        }

        .bv-bar-fill {
            background:
                linear-gradient(
                    90deg,
                    var(--forest-mid),
                    var(--gold)
                );

            height: 100%;
            border-radius: 999px;
        }

        .bv-rank-row {
            display: flex;
            justify-content: space-between;
            margin-top: 0.7rem;
        }

        /* HISTORY */

        .history-card {
            background: white;
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 1rem;
            margin-bottom: 1rem;
        }

        .history-title {
            font-family: 'Fraunces', serif;
            font-size: 1.1rem;
            color: var(--forest-deep);
        }

        .history-confidence {
            color: var(--forest-mid);
            font-weight: 600;
        }

        /* METRICS */

        .metric-card {
            background: white;
            border-radius: 16px;
            padding: 1.2rem;
            text-align: center;
            border: 1px solid var(--line);
        }

        .metric-value {
            font-family: 'Fraunces', serif;
            font-size: 1.8rem;
            color: var(--forest-deep);
            font-weight: 600;
        }

        .metric-label {
            color: var(--ink-muted);
            font-size: 0.85rem;
        }

        /* BUTTON */

        .stButton > button {
            background: var(--forest-deep);
            color: white;
            border-radius: 12px;
            border: none;
            font-weight: 600;
        }

        .stButton > button:hover {
            background: var(--forest-mid);
            color: white;
        }

        </style>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# HERO
# ============================================================

def render_hero():

    st.markdown(
        """
        <div class="bv-hero">

            <div class="bv-badge">
                VGG7 • PyTorch • 20 species
            </div>

            <h1>
                BirdVision AI
            </h1>

            <p class="bv-subtitle">
                AI-powered bird species classification
                using a custom VGG7 deep learning model.
            </p>

            <p class="bv-desc">
                Upload a bird image and let our computer
                vision model identify the species.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# PREDICTION RESULT
# ============================================================

def render_prediction_result(
    results,
    elapsed_seconds
):

    top_label, top_conf = results[0]

    st.markdown(
        f"""
        <div class="bv-result">

            <div class="bv-eyebrow">
                Prediction
            </div>

            <div class="bv-species">
                {top_label.title()}
            </div>

            <div class="bv-confidence">
                {top_conf:.1f}% confidence
            </div>

            <div class="bv-bar-track">
                <div
                    class="bv-bar-fill"
                    style="width:{min(top_conf,100):.1f}%"
                ></div>
            </div>

            <div style="color:#5B6B60;">
                VGG7 • Inference time:
                {elapsed_seconds:.2f}s
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        "### Top predictions"
    )

    for rank, (label, confidence) in enumerate(
        results,
        start=1
    ):

        st.markdown(
            f"""
            <div class="bv-rank-row">

                <span>
                    {rank}. {label.title()}
                </span>

                <span>
                    {confidence:.1f}%
                </span>

            </div>

            <div class="bv-bar-track">

                <div
                    class="bv-bar-fill"
                    style="width:{min(confidence,100):.1f}%"
                ></div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# IDENTIFICATION SECTION
# ============================================================

def render_upload_and_predict(
    model,
    class_names,
    device,
    model_error
):

    st.markdown(
        '<h2 class="bv-section-title">Identify a bird</h2>',
        unsafe_allow_html=True
    )

    if model_error:

        st.error(model_error)

        return

    left, right = st.columns(
        [1, 1],
        gap="large"
    )

    # --------------------------------------------------------
    # LEFT SIDE
    # --------------------------------------------------------

    with left:

        st.markdown(
            "### Upload a bird image"
        )

        st.caption(
            "PNG, JPG or JPEG • "
            "Recommended image size: 224 × 224 or larger"
        )

        uploaded_file = st.file_uploader(
            "Upload image",
            type=[
                "png",
                "jpg",
                "jpeg"
            ],
            label_visibility="collapsed"
        )

        image = None

        if uploaded_file is not None:

            try:

                image = Image.open(
                    uploaded_file
                ).convert("RGB")

                st.image(
                    image,
                    use_container_width=True
                )

            except Exception:

                st.error(
                    "Invalid image. "
                    "Please upload a PNG or JPEG."
                )

        predict_clicked = st.button(
            "Identify Bird",
            type="primary",
            disabled=image is None,
            use_container_width=True
        )

    # --------------------------------------------------------
    # RIGHT SIDE
    # --------------------------------------------------------

    with right:

        st.markdown(
            "### Prediction"
        )

        if image is None:

            st.info(
                "Upload an image on the left "
                "to begin identification."
            )

        elif not predict_clicked:

            st.info(
                "Your image is ready. "
                "Click **Identify Bird**."
            )

        else:

            with st.spinner(
                "Running the VGG7 model..."
            ):

                start = time.time()

                tensor = preprocess_image(
                    image
                )

                results = predict_image(
                     model,
                     tensor,
                     class_names,
                     device,
                     top_k=3
)
                elapsed = time.time() - start
                top_label, top_conf = results[0]
                save_prediction(
                    uploaded_file.name,
                    top_label,
                    top_conf
                    )
                render_prediction_result(results, elapsed)

            # ------------------------------------------------
            # SAVE HISTORY
            # ------------------------------------------------

            top_label, top_conf = results[0]

            history_item = {
                "image": image.copy(),
                "filename": uploaded_file.name,
                "prediction": top_label,
                "confidence": top_conf,
                "time": time.strftime(
                    "%d %b %Y, %I:%M:%S %p"
                )
            }

            st.session_state.prediction_history.insert(
                0,
                history_item
            )


# ============================================================
# HISTORY
# ============================================================

def render_history():

    st.markdown(
        '<h2 class="bv-section-title">Prediction History</h2>',
        unsafe_allow_html=True
    )

    history = st.session_state.prediction_history

    if not history:

        st.info(
            "No images have been classified yet. "
            "Your prediction history will appear here."
        )

        return

    col1, col2 = st.columns(
        [4, 1]
    )

    with col1:

        st.write(
            f"**{len(history)} image(s) classified in this session**"
        )

    with col2:

        if st.button(
            "Clear History",
            use_container_width=True
        ):

            st.session_state.prediction_history = []

            st.rerun()

    for index, item in enumerate(history):

        with st.container():

            image_col, info_col = st.columns(
                [1, 3]
            )

            with image_col:

                st.image(
                    item["image"],
                    use_container_width=True
                )

            with info_col:

                st.markdown(
                    f"""
                    <div class="history-card">

                        <div class="history-title">
                            {item["prediction"].title()}
                        </div>

                        <p>
                            <b>File:</b>
                            {item["filename"]}
                        </p>

                        <p class="history-confidence">
                            Confidence:
                            {item["confidence"]:.1f}%
                        </p>

                        <p>
                            <b>Uploaded:</b>
                            {item["time"]}
                        </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )
                def render_history():
    st.markdown(
        '<h2 class="bv-section-title">Prediction history</h2>',
        unsafe_allow_html=True,
    )

    history = get_prediction_history()

    if not history:
        st.markdown(
            """
            <div class="bv-card">
                <strong>No predictions yet.</strong><br/>
                Your uploaded images and prediction results will appear here
                after you identify a bird.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Clear history button
    if st.button("Clear Prediction History"):
        if clear_prediction_history():
            st.success("Prediction history cleared.")
            st.rerun()
        else:
            st.error("Could not clear prediction history.")

    st.markdown(
        f"""
        <div class="bv-card">
            <strong>{len(history)}</strong> prediction(s) recorded
        </div>
        """,
        unsafe_allow_html=True,
    )

    for item in history:
        image_name = item.get("image_name", "Unknown image")
        predicted_class = item.get("predicted_class", "Unknown")
        confidence = float(item.get("confidence", 0))

        created_at = item.get("created_at", "")

        # Make timestamp easier to read
        if created_at:
            created_at = created_at.replace("T", " ")
            if "+" in created_at:
                created_at = created_at.split("+")[0]
            elif created_at.endswith("Z"):
                created_at = created_at[:-1]

        st.markdown(
            f"""
            <div class="bv-card" style="padding:1.2rem 1.5rem;">
                <div style="
                    display:flex;
                    justify-content:space-between;
                    align-items:center;
                    gap:20px;
                    flex-wrap:wrap;
                ">
                    <div>
                        <div style="
                            color:var(--ink-muted);
                            font-size:0.8rem;
                            margin-bottom:0.25rem;
                        ">
                            Uploaded image
                        </div>

                        <div style="
                            font-weight:600;
                            color:var(--ink);
                        ">
                            {image_name}
                        </div>

                        <div style="
                            margin-top:0.45rem;
                            font-family:'Fraunces',serif;
                            font-size:1.25rem;
                            color:var(--forest-deep);
                        ">
                            {predicted_class.title()}
                        </div>
                    </div>

                    <div style="text-align:right;">
                        <div style="
                            color:var(--forest-mid);
                            font-weight:700;
                            font-size:1.2rem;
                        ">
                            {confidence:.1f}%
                        </div>

                        <div style="
                            color:var(--ink-muted);
                            font-size:0.8rem;
                        ">
                            Confidence
                        </div>

                        <div style="
                            color:var(--ink-muted);
                            font-size:0.75rem;
                            margin-top:0.3rem;
                        ">
                            {created_at}
                        </div>
                    </div>
                </div>

                <div class="bv-bar-track" style="margin-top:1rem;">
                    <div
                        class="bv-bar-fill"
                        style="width:{min(confidence, 100):.1f}%;">
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# ABOUT MODEL
# ============================================================

def render_about_model():

    st.markdown(
        '<h2 class="bv-section-title">About the model</h2>',
        unsafe_allow_html=True
    )

    cols = st.columns(5)

    cards = [

        (
            str(METRICS["num_classes"]),
            "Bird species"
        ),

        (
            f'{METRICS["test_accuracy"]:.0f}%',
            "Test accuracy"
        ),

        (
            f'{METRICS["val_accuracy"]:.0f}%',
            "Validation accuracy"
        ),

        (
            "VGG7",
            "CNN architecture"
        ),

        (
            "PyTorch",
            "Framework"
        )
    ]

    for col, (value, label) in zip(
        cols,
        cards
    ):

        with col:

            st.markdown(
                f"""
                <div class="metric-card">

                    <div class="metric-value">
                        {value}
                    </div>

                    <div class="metric-label">
                        {label}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# PERFORMANCE
# ============================================================

def render_performance():

    st.markdown(
        '<h2 class="bv-section-title">Model performance</h2>',
        unsafe_allow_html=True
    )

    cols = st.columns(4)

    cards = [

        (
            f'{METRICS["val_accuracy"]:.2f}%',
            "Validation accuracy"
        ),

        (
            f'{METRICS["test_accuracy"]:.2f}%',
            "Test accuracy"
        ),

        (
            f'{METRICS["precision"]:.2f}%',
            "Macro precision"
        ),

        (
            f'{METRICS["recall"]:.2f}%',
            "Macro recall"
        )
    ]

    for col, (value, label) in zip(
        cols,
        cards
    ):

        with col:

            st.markdown(
                f"""
                <div class="metric-card">

                    <div class="metric-value">
                        {value}
                    </div>

                    <div class="metric-label">
                        {label}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    st.caption(
        f'Macro F1-score: {METRICS["f1"]:.2f}%'
    )


# ============================================================
# ERROR ANALYSIS
# ============================================================

def render_error_analysis():

    st.markdown(
        '<h2 class="bv-section-title">Model error analysis</h2>',
        unsafe_allow_html=True
    )

    st.write(
        f"The model correctly classified "
        f"{METRICS['correct']} of "
        f"{METRICS['test_images']} test images "
        f"and misclassified "
        f"{METRICS['incorrect']}."
    )

    if os.path.exists(
        INCORRECT_PREDICTIONS_PATH
    ):

        st.image(
            INCORRECT_PREDICTIONS_PATH,
            use_container_width=True
        )


# ============================================================
# CONFUSION MATRIX
# ============================================================

def render_confusion_matrix():

    st.markdown(
        '<h2 class="bv-section-title">Confusion matrix</h2>',
        unsafe_allow_html=True
    )

    st.caption(
        "Prediction distribution across the 20 bird species."
    )

    if os.path.exists(
        CONFUSION_MATRIX_PATH
    ):

        st.image(
            CONFUSION_MATRIX_PATH,
            use_container_width=True
        )


# ============================================================
# HOW IT WORKS
# ============================================================

def render_how_it_works():

    st.markdown(
        '<h2 class="bv-section-title">How it works</h2>',
        unsafe_allow_html=True
    )

    cols = st.columns(4)

    steps = [

        ("01", "Upload image"),

        ("02", "Image preprocessing"),

        ("03", "VGG7 prediction"),

        ("04", "Species + confidence")
    ]

    for col, (number, label) in zip(
        cols,
        steps
    ):

        with col:

            st.markdown(
                f"""
                <div style="
                    text-align:center;
                    padding:1rem;
                ">

                    <div style="
                        font-family:Fraunces;
                        font-size:1.5rem;
                        color:#B08A4E;
                    ">
                        {number}
                    </div>

                    <b>{label}</b>

                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# TECH STACK
# ============================================================

def render_tech_stack():

    st.markdown(
        '<h2 class="bv-section-title">Tech stack</h2>',
        unsafe_allow_html=True
    )

    st.write(
        "Python • PyTorch • Torchvision • CUDA • "
        "Streamlit • Scikit-learn • Matplotlib"
    )


# ============================================================
# FOOTER
# ============================================================

def render_footer():

    st.markdown(
        """
        <hr>

        <div style="
            text-align:center;
            color:#5B6B60;
            padding:1rem;
        ">

            <b>BirdVision AI</b><br>

            Custom VGG7 CNN for
            20-species bird classification.<br>

            Built with PyTorch.

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MAIN
# ============================================================

def main():

    st.set_page_config(
        page_title="BirdVision AI",
        page_icon="🦜",
        layout="wide"
    )

    inject_css()

    render_hero()

    model, class_names, device, model_error = load_model()

    render_upload_and_predict(
        model,
        class_names,
        device,
        model_error
    )
    

    # NEW
    render_history()

    render_about_model()

    render_performance()

    render_error_analysis()

    render_confusion_matrix()

    render_how_it_works()

    render_tech_stack()

    render_footer()


if __name__ == "__main__":
    main()