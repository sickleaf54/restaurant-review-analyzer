import re
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

# -------------------------------------------------
# Paths – must match filenames in your repo
# -------------------------------------------------
MODEL_PATH = "restaurant_review_model.keras"
CSV_PATH = "Restaurant reviews.csv"  # <- must match EXACT file name


# -------------------------------------------------
# Helpers (must match training pre-processing)
# -------------------------------------------------
def clean_text(text: str) -> str:
    """Lowercase, remove non-letters, collapse spaces."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"[\n\t]+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """Load the CSV and clean the review text (no labels needed for inference)."""
    df = pd.read_csv(csv_path)

    # Drop weird numeric column if present (from your dataset)
    df = df.drop(columns=["7514"], errors="ignore")

    # Remove non-numeric ratings like "Like"
    df = df[df["Rating"] != "Like"].copy()

    # Drop rows without rating or review
    df = df.dropna(subset=["Rating", "Review"])

    # Convert rating to float just in case you need it later
    df["Rating"] = df["Rating"].astype(float)

    data = df[["Restaurant", "Reviewer", "Review", "Rating"]].copy()
    data["clean_review"] = data["Review"].apply(clean_text)
    return data


OVERALL_LABELS = {0: "Negative", 1: "Neutral", 2: "Positive"}
SERVICE_LABELS = {0: "Bad", 1: "Neutral", 2: "Good"}


def summarize_ratio(ratio: float) -> str:
    """Turn a 0–1 ratio into a qualitative label."""
    if ratio >= 0.8:
        return "High"
    elif ratio >= 0.5:
        return "Moderate"
    else:
        return "Low"


# -------------------------------------------------
# Cached loaders (so Cloud doesn't reload every run)
# -------------------------------------------------
@st.cache_resource
def load_model_cached():
    # Use tf.keras, not standalone keras
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


@st.cache_data
def load_data_cached():
    return load_and_prepare_data(CSV_PATH)


model = load_model_cached()
data = load_data_cached()


# -------------------------------------------------
# Core analysis
# -------------------------------------------------
def analyze_restaurant(restaurant_name: str):
    # Find rows for this restaurant
    mask = data["Restaurant"].str.contains(restaurant_name, case=False, na=False)
    subset = data[mask].copy()

    if subset.empty:
        st.warning(f"No reviews found for '{restaurant_name}'.")
        return

    unique_restaurants = subset["Restaurant"].unique()
    st.write("**Matched restaurant(s):**")
    for name in unique_restaurants:
        st.write(f"- {name}")

    reviews_raw = subset["Review"].astype(str).tolist()
    reviews_clean = [clean_text(t) for t in reviews_raw]

    if len(reviews_clean) == 0:
        st.warning("No usable reviews for this restaurant.")
        return

    # 🔑 Make sure we pass a tf.string tensor to the model
    inputs = tf.constant(reviews_clean, dtype=tf.string)

    preds = model.predict(inputs, batch_size=32, verbose=0)

    # Keras 3 may return a dict OR a list/tuple, handle both
    if isinstance(preds, dict):
        overall_probs = preds["overall_output"]
        service_probs = preds["service_output"]
        allergy_probs = preds["allergy_output"]
        health_probs = preds["health_output"]
        veg_probs = preds["veg_output"]
    else:
        # Fallback: assume order [overall, service, allergy, health, veg]
        overall_probs, service_probs, allergy_probs, health_probs, veg_probs = preds

    overall_pred = overall_probs.argmax(axis=-1)
    service_pred = service_probs.argmax(axis=-1)
    allergy_pred = (allergy_probs >= 0.5).astype(int).ravel()
    health_pred = (health_probs >= 0.5).astype(int).ravel()
    veg_pred = (veg_probs >= 0.5).astype(int).ravel()

    n = len(reviews_clean)

    # Overall sentiment distribution
    overall_counts = np.bincount(overall_pred, minlength=3)
    overall_ratios = overall_counts / n

    # Service sentiment distribution
    service_counts = np.bincount(service_pred, minlength=3)
    service_ratios = service_counts / n

    # Food-friendly ratios
    allergy_safe_ratio = float(allergy_pred.mean()) if n > 0 else 0.0
    health_ok_ratio = float(health_pred.mean()) if n > 0 else 0.0
    veg_friendly_ratio = float(veg_pred.mean()) if n > 0 else 0.0

    # --------- Display results ----------
    st.subheader("Overall Sentiment")
    for i in range(3):
        st.write(
            f"{OVERALL_LABELS[i]:>8}: {overall_counts[i]:3d} reviews "
            f"({overall_ratios[i] * 100:5.1f}%)"
        )
    dominant_overall = OVERALL_LABELS[int(overall_counts.argmax())]
    st.info(f"**Dominant overall sentiment:** {dominant_overall}")

    st.subheader("Service Sentiment")
    for i in range(3):
        st.write(
            f"{SERVICE_LABELS[i]:>8}: {service_counts[i]:3d} reviews "
            f"({service_ratios[i] * 100:5.1f}%)"
        )
    dominant_service = SERVICE_LABELS[int(service_counts.argmax())]
    st.info(f"**Dominant service sentiment:** {dominant_service}")

    st.subheader("Food Friendly Factors")
    st.write(
        f"- Allergy safety: **{summarize_ratio(allergy_safe_ratio)}** "
        f"({allergy_safe_ratio * 100:5.1f}% of reviews predicted safe)"
    )
    st.write(
        f"- Health standards: **{summarize_ratio(health_ok_ratio)}** "
        f"({health_ok_ratio * 100:5.1f}% of reviews with no issues)"
    )
    st.write(
        f"- Vegetarian friendly: **{summarize_ratio(veg_friendly_ratio)}** "
        f"({veg_friendly_ratio * 100:5.1f}% of reviews predicted veg-friendly)"
    )

    st.subheader("Sample Reviews & Predictions")
    show_n = min(5, n)
    for i in range(show_n):
        st.markdown(f"**Review {i + 1}:**")
        txt = reviews_raw[i]
        if len(txt) > 500:
            txt_short = txt[:500] + " ..."
        else:
            txt_short = txt

        st.write(txt_short)
        st.write(
            f"- Overall sentiment: **{OVERALL_LABELS[int(overall_pred[i])]}**\n"
            f"- Service sentiment: **{SERVICE_LABELS[int(service_pred[i])]}**\n"
            f"- Allergy safe: {'Yes' if allergy_pred[i] == 1 else '⚠️ Risk mentioned'}\n"
            f"- Health standard: {'OK' if health_pred[i] == 1 else '⚠️ Issue mentioned'}\n"
            f"- Veg friendly: {'Yes' if veg_pred[i] == 1 else 'No / poor options'}"
        )


# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
st.title("Restaurant Review Analyzer")

st.write(
    """
This app uses a multi-output deep learning model trained on restaurant reviews to estimate:

- **Overall sentiment** (positive / neutral / negative)  
- **Service sentiment** (good / neutral / bad)  
- **Food-friendly factors** (allergy safety, health standards, vegetarian friendliness)
"""
)

unique_restaurants = sorted(data["Restaurant"].dropna().unique())

st.sidebar.header("Restaurant Search")
search_text = st.sidebar.text_input(
    "Type part of a restaurant name",
    value="",
    help="Case-insensitive substring search",
)

if search_text:
    matches = [r for r in unique_restaurants if search_text.lower() in r.lower()]
else:
    matches = unique_restaurants[:50]  # show first 50 as default

if not matches:
    st.sidebar.write("No matches. Try a different search.")
    selected_restaurant = None
else:
    selected_restaurant = st.sidebar.selectbox(
        "Select a restaurant from matches",
        matches,
    )

if selected_restaurant:
    st.write(f"### Analysis for: **{selected_restaurant}**")
    if st.button("Run Analysis"):
        analyze_restaurant(selected_restaurant)
else:
    st.info(
        "Use the sidebar to search for a restaurant name from the dataset, "
        "then click **Run Analysis**."
    )
