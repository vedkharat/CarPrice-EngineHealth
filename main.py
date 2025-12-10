import streamlit as st
import joblib
import numpy as np
import librosa
import io
import pandas as pd
from pathlib import Path
import base64
from pathlib import Path
import requests

# Base directory for this app
BASE_DIR = Path(__file__).resolve().parent

# Google Drive direct download URL for the price model
PRICE_MODEL_URL = "https://drive.google.com/uc?export=download&id=1cfNPQwWBjt_QXYVFE_7pUa-rahi8yVEih"

PRICE_MODEL_PATH = BASE_DIR / "car_price_catboost.pkl"
ENGINE_MODEL_PATH = BASE_DIR / "engine_health_catboost.pkl"




# 0. Page config + global CSS


st.set_page_config(
    page_title="Car Price & Engine Health Studio",
    page_icon="🚗",
    layout="wide",
)

# --- Global CSS (Tesla/Lambo style) ---
st.markdown(
    """
    <style>
    /* Global page */
    .stApp {
        background: radial-gradient(circle at top left, #141721 0, #05060a 55%, #020308 100%);
        color: #f5f5f7;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text",
                     "Roboto", "Segoe UI", sans-serif;
    }

    /* Make content a bit narrower, like Tesla site */
    .block-container {
        padding-top: 1.5rem;
        max-width: 1150px;
    }

    /* Hero section */
    .hero-wrapper {
        position: relative;
        z-index: 2;  /* sit clearly above background video */
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.75rem;
        margin: 1.25rem 0 1.5rem;
        text-align: center;

        padding: 1.1rem 2.2rem;
        border-radius: 999px;
        background: radial-gradient(circle at top,
                      rgba(0, 0, 0, 0.92) 0%,
                      rgba(0, 0, 0, 0.78) 45%,
                      rgba(0, 0, 0, 0.65) 100%);
        box-shadow: 0 0 35px rgba(0, 0, 0, 0.9);
    }

    .hero-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.35rem 0.85rem;
        border-radius: 999px;
        background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%);
        font-size: 0.78rem;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        font-weight: 600;
        color: #0b1020;
    }

    .hero-pill span:first-child {
        color: #0b1020;
    }

    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        color: #ffffff;
        text-shadow: 0 0 24px rgba(0, 0, 0, 1);
    }

    .hero-subtitle {
        font-size: 1.0rem;
        color: #f9fbff;
        max-width: 640px;
        margin: 0 auto;
        text-shadow: 0 0 18px rgba(0, 0, 0, 1);
    }

    /* Cards, info boxes etc. */
    .card {
        background: rgba(9, 15, 30, 0.88);
        border-radius: 999px;
        padding: 0.9rem 1.3rem;
        margin: 1.5rem 0 1rem;
        border: 1px solid rgba(71, 85, 105, 0.8);
    }

    .card-secondary {
        border-radius: 18px;
        padding: 1.1rem 1.4rem;
    }

    .success-box {
        background: linear-gradient(90deg, #22c55e, #16a34a);
        border-radius: 999px;
        padding: 0.55rem 1.2rem;
        font-weight: 600;
        color: #022c22;
        display: inline-block;
        margin-top: 0.6rem;
    }

    .warning-box {
        background: linear-gradient(90deg, #f97316, #ea580c);
        border-radius: 999px;
        padding: 0.55rem 1.2rem;
        font-weight: 600;
        color: #1b1006;
        display: inline-block;
        margin-top: 0.6rem;
    }

    .muted-note {
        font-size: 0.8rem;
        color: #e5b96c;
        margin-top: 0.4rem;
    }

    .upload-hint {
        font-size: 0.78rem;
        color: #cbd5f5;
        margin-bottom: 0.2rem;
    }

    /* Full-page Lamborghini-style background video */
    .bg-video {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        object-fit: cover;
        z-index: 0;
        opacity: 0.45;
        pointer-events: none;
    }

    /* Section switcher: radio -> pill tabs */
    .stRadio > div {
        display: inline-flex;
        gap: 0;
        padding: 0.18rem;
        border-radius: 999px;
        background: rgba(15, 23, 42, 0.92);
        border: 1px solid rgba(148, 163, 184, 0.7);
        backdrop-filter: blur(12px);
        margin-bottom: 0.8rem;
    }

    .stRadio [role="radio"] {
        padding: 0.3rem 0.95rem;
        border-radius: 999px;
        cursor: pointer;
        transition: background 0.18s ease-out, color 0.18s ease-out;
    }

    .stRadio [aria-checked="true"] {
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        color: #020617 !important;
        font-weight: 600;
    }

    .stRadio label {
        color: #e5e7eb !important;
        font-size: 0.86rem;
    }

    li::marker {
        color: #9ca3ff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    /* Remove the dark oval behind the main heading */
    .hero-wrapper,
    .hero-wrapper::before {
        background: transparent !important;
        box-shadow: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Hero section
st.markdown(
    """
    <div class="hero-wrapper">
      <div class="hero-pill">
        <span>AI-powered</span>
        <span>Price &amp; Engine Health Studio</span>
      </div>
      <div class="hero-title">Car Price &amp; Engine Health Studio</div>
      <p class="hero-subtitle">
        Estimate your car's resale price and check engine health from sound clips &mdash; all in your browser.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# Valid value lists for basic input validation


VALID_MANUFACTURERS = {
    "acura", "audi", "bmw", "buick", "cadillac", "chevrolet", "chevy",
    "chrysler", "dodge", "ford", "gmc", "honda", "hyundai", "infiniti",
    "jeep", "kia", "land rover", "lexus", "lincoln", "mazda", "mercedes-benz",
    "mini", "mitsubishi", "nissan", "porsche", "ram", "subaru", "tesla",
    "toyota", "volkswagen", "vw", "volvo", "jaguar", "fiat", "harley-davidson",
}

VALID_US_STATE_CODES = {
    "al","ak","az","ar","ca","co","ct","de","fl","ga","hi","id","il","in","ia",
    "ks","ky","la","me","md","ma","mi","mn","ms","mo","mt","ne","nv","nh","nj",
    "nm","ny","nc","nd","oh","ok","or","pa","ri","sc","sd","tn","tx","ut","vt",
    "va","wa","wv","wi","wy","dc"
}

def validate_car_inputs(manufacturer: str, model_name: str, state: str) -> bool:
    """
    Basic sanity checks for manufacturer / model / state.

    Returns True if everything looks OK, False otherwise
    (and shows error messages via Streamlit).
    """
    errors = []

    manu = (manufacturer or "").strip().lower()
    model = (model_name or "").strip()
    state_code = (state or "").strip().lower()

    # Manufacturer must be from our supported set
    if manu not in VALID_MANUFACTURERS:
        errors.append(
            "Manufacturer not recognized. Please use a common brand name "
            "(e.g. Toyota, Honda, Ford, BMW, etc.)."
        )

    # Very short model_names are usually nonsense
    if len(model) < 2:
        errors.append("Model name looks too short. Please enter a full model name (e.g. 'corolla').")

    # State must be a valid 2-letter US code
    if state_code not in VALID_US_STATE_CODES:
        errors.append("State must be a valid 2-letter US state code (e.g. 'ca', 'ny', 'tx').")

    if errors:
        for msg in errors:
            st.error(msg)
        return False

    return True

# Placeholder used for background video so we always have exactly one <video> tag
bg_slot = st.empty()


# ---------- Background video helpers ----------

@st.cache_data
def load_video_b64(video_name: str) -> str:
    """Read a video from assets/ and return as base64 string."""
    base_dir = Path(__file__).resolve().parent
    video_path = base_dir / "assets" / video_name
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    return base64.b64encode(video_bytes).decode("utf-8")


def render_background_video(video_name: str):
    """
    Render a fixed background video behind the whole app.

    """
    try:
        b64 = load_video_b64(video_name)
    except FileNotFoundError:
        st.warning(f"Background video '{video_name}' not found in assets/.")
        return

    video_id = f"bg-video-{video_name.replace('.', '-')}"
    bg_slot.markdown(
        f"""
        <video id="{video_id}" class="bg-video" autoplay muted playsinline>
            <source src="data:video/mp4;base64,{b64}" type="video/mp4">
        </video>
        """,
        unsafe_allow_html=True,
    )





CATEGORICAL_COLS = [
    "region", "manufacturer", "model", "fuel",
    "title_status", "transmission", "drive",
    "size", "type", "paint_color", "state",
    "age_bucket",
]

CONDITION_SCORES = {
    "new": 5,
    "like new": 4,
    "excellent": 4,
    "good": 3,
    "fair": 2,
    "salvage": 1,
}

def _file_looks_like_pickle(path: Path) -> bool:
    """
    Quick sanity check: does this file look like a Python pickle?

    Most pickles start with 0x80 (the PROTO opcode). If we don't see that,
    it's probably an HTML error page or some other bad content.
    """
    try:
        with open(path, "rb") as f:
            header = f.read(4)
        return len(header) >= 1 and header[0] == 0x80
    except Exception:
        return False


def download_if_missing(url: str, dest: Path):
    """
    Download a file from `url` to `dest` if it is missing OR looks corrupted.
    This fixes the case where an earlier run saved an HTML error page instead
    of a real .pkl, which then crashes joblib.load().
    """
    # If file exists and looks like a real pickle, keep it.
    if dest.exists() and _file_looks_like_pickle(dest):
        return

    # Otherwise, try to (re)download it.
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            tmp_path = dest.with_suffix(dest.suffix + ".tmp")
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            tmp_path.replace(dest)
    except Exception as e:
        # If something went wrong, remove any half-downloaded file
        try:
            dest.unlink(missing_ok=True)
        except Exception:
            pass

        st.error("❌ Could not download the price model bundle from Google Drive.")
        st.exception(e)
        st.stop()


# 1. Load models
@st.cache_resource
def load_models():
    # Ensure price model is present (download from Google Drive if needed)
    download_if_missing(PRICE_MODEL_URL, PRICE_MODEL_PATH)

    # Load price model bundle
    bundle = joblib.load(PRICE_MODEL_PATH)
    car_price_model = bundle["model"]
    price_feature_names = bundle["feature_names"]
    # Fallback to your existing categorical list if not stored in the bundle
    cat_cols_in_model = bundle.get("cat_cols", CATEGORICAL_COLS)

    # Load the small engine-health model from the repo
    engine_model = joblib.load(ENGINE_MODEL_PATH)

    return car_price_model, price_feature_names, engine_model


car_price_model, price_feature_names, engine_model = load_models()



# 2. Helper: extract MFCC from uploaded audio

def extract_mfcc_from_file(file, n_mfcc=40):
    y, sr = librosa.load(file, sr=None)
    y = librosa.effects.trim(y)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)
    feats = np.concatenate([mfcc_mean, mfcc_std])
    return feats.reshape(1, -1)


def build_price_features_from_form(
    year,
    odometer,
    condition,
    fuel,
    manufacturer,
    model_name,
    state,
    transmission,
    price_feature_names,
):
    cond_key = condition.lower().strip()
    cond_score = CONDITION_SCORES.get(cond_key, 3.0)

    raw = {
        "year": int(year),
        "odometer": float(odometer),
        "condition": float(cond_score),
        "fuel": fuel,
        "manufacturer": manufacturer.lower(),
        "model": model_name.lower(),
        "state": state.lower(),
        "transmission": transmission.lower(),
    }

    df = pd.DataFrame([raw])
    df = df.reindex(columns=price_feature_names)

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("unknown")

    return df


# 3. Tabs + background video


# Tesla/Lambo-style pill tabs (radio)
section = st.radio(
    "Choose section",
    ("Car details", "Engine sound"),
    index=0,
    horizontal=True,
    label_visibility="collapsed",
    key="section_radio",
)

# SUPER SIMPLE: map sections → videos
if section == "Car details":
    bg_video = "front_lights.mp4"    # taillights for price tab
else:
    bg_video = "rear_lights.mp4"   # headlights for engine tab

# This runs on every rerun, so every time you flip the radio,
# the <video> element is replaced with the other clip and autoplay starts again.
render_background_video(bg_video)

# 4. Car price section

if section == "Car details":

    st.markdown("### Car Details")

    col1, col2 = st.columns(2)

    with col1:
        year = st.number_input("Manufacture year", min_value=1980, max_value=2025, value=2015)
        odometer = st.number_input("Odometer (miles)", min_value=0, max_value=500000, value=80000)
        condition = st.selectbox("Condition", ["salvage", "fair", "good", "excellent", "like new", "new"])
        fuel = st.selectbox("Fuel", ["gas", "diesel", "hybrid", "electric", "other"])

    with col2:
        manufacturer = st.text_input("Manufacturer (e.g. toyota, honda)", "toyota")
        model_name = st.text_input("Model name (e.g. corolla)", "corolla")
        state = st.text_input("State (2-letter code, e.g. ca, ny)", "ca")
        transmission = st.selectbox("Transmission", ["automatic", "manual", "other"])

    st.markdown(
        '<div class="muted-note">📙 All price estimates are shown in US dollars (USD).</div>',
        unsafe_allow_html=True,
    )

    # --- Predict button & result ---
    if st.button("Predict price"):
        # 1) Validate the text inputs before we even build the feature dataframe
        if not validate_car_inputs(manufacturer, model_name, state):
            # Stop this Streamlit run here – do NOT call the model
            st.stop()

        try:
            X_price = build_price_features_from_form(
                year=year,
                odometer=odometer,
                condition=condition,
                fuel=fuel,
                manufacturer=manufacturer,
                model_name=model_name,
                state=state,
                transmission=transmission,
                price_feature_names=price_feature_names,
            )


            log_price_pred = car_price_model.predict(X_price)[0]
            price_pred = float(np.expm1(log_price_pred))
            price_pred = max(0.0, min(price_pred, 200000.0))

            st.markdown(
                f'<div class="success-box">Estimated car price: <strong>${price_pred:,.0f}</strong></div>',
                unsafe_allow_html=True,
            )

        except Exception as e:
            st.markdown(
                '<div class="warning-box">Price prediction failed. Please double-check the inputs or try again.</div>',
                unsafe_allow_html=True,
            )
            st.exception(e)

    st.markdown(
        """
        <div class="card card-secondary">
          <h4>ℹ️ About this price estimate</h4>
          <ul>
            <li>Model trained on <strong>2021 US Craigslist used-car listings</strong>.</li>
            <li>Prices are <strong>rough estimates in USD</strong>, not official valuations.</li>
            <li>Works best for common passenger cars (not luxury, exotic, or commercial vehicles).</li>
            <li>Condition, mileage, year and state have the biggest impact on the prediction.</li>
            <li>Always compare with <strong>local market listings</strong> before making any final decision.</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

# 5. Engine sound section

else:  # section == "Engine sound"

    st.markdown("### Engine sound")

    st.markdown(
        """
        <div class="upload-hint">
          📌 Upload a short engine sound clip (idle, starting, or while revving).
          The model will try to guess whether the sound is <strong>healthy</strong> or <strong>faulty</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_audio = st.file_uploader(
        "Upload an engine sound clip (max ~20MB, WAV/MP3/OGG)",
        type=["wav", "mp3", "ogg"],
    )

    if uploaded_audio is not None:
        st.audio(uploaded_audio)

        if st.button("Analyze engine health"):
            try:
                feats = extract_mfcc_from_file(uploaded_audio)
                prob = engine_model.predict_proba(feats)[0]
                pred = engine_model.predict(feats)[0]

                prob_healthy = float(prob[0])
                prob_faulty = float(prob[1])

                if pred == 0:
                    st.markdown(
                        f'<div class="success-box">✅ Engine predicted as <strong>HEALTHY</strong> '
                        f'(P(faulty) ≈ {prob_faulty*100:.1f}%).</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="warning-box">⚠️ Engine predicted as <strong>FAULTY</strong> '
                        f'(P(faulty) ≈ {prob_faulty*100:.1f}%).</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    f'<div class="confidence-text">Confidence — '
                    f'Healthy: {prob_healthy*100:.1f}%, Faulty: {prob_faulty*100:.1f}%.</div>',
                    unsafe_allow_html=True,
                )

            except Exception as e:
                st.markdown(
                    '<div class="warning-box">Engine analysis failed. Please try another clip or check the file format.</div>',
                    unsafe_allow_html=True,
                )
                st.exception(e)
    else:
        st.info("Upload an audio file to run the engine health model.")

    st.markdown(
        """
        <div class="card card-secondary">
          <h4>ℹ️ About this engine health estimate</h4>
          <ul>
            <li>Model trained on a <strong>small engine-sound dataset</strong>, with augmentation.</li>
            <li>Predictions may be <strong>overfitted to the training clips</strong> and might not generalise to every real-world car.</li>
            <li>Treat this tool as a <strong>demo / second opinion</strong>, not a professional diagnosis.</li>
            <li>For any real issues, always consult a <strong>qualified mechanic</strong>.</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
