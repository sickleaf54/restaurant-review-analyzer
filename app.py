import re
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

MODEL_PATH = "restaurant_review_model.keras"
CSV_PATH = "Restaurant reviews.csv"  # must match filename exactly

# -------------------------------------------------
# NLTK / VADER set-up
# -------------------------------------------------
try:
    sia = SentimentIntensityAnalyzer()
except LookupError:
    nltk.download("vader_lexicon")
    sia = SentimentIntensityAnalyzer()


def vader_compound(text: str) -> float:
    if not isinstance(text, str):
        text = str(text)
    return float(sia.polarity_scores(text)["compound"])


# -------------------------------------------------
# Keyword lists for "mentions"
# -------------------------------------------------
HEALTH_BAD_WORDS = [
    "food poisoning",
    "got sick",
    "made me sick",
    "felt sick",
    "vomit",
    "vomiting",
    "nausea",
    "diarrhea",
    "diarrhoea",
    "stomach ache",
    "stomach cramps",
    "upset stomach",
    "undercooked",
    "raw chicken",
    "raw meat",
    "not fully cooked",
    "smelled bad",
    "off smell",
    "rotten",
    "spoiled",
    "mold",
    "mouldy",
    "dirty kitchen",
    "unsanitary",
    "cockroach",
    "roaches",
    "rat",
    "mouse droppings",
    "hair in food",
    "found a bug",
]

ALLERGY_MENTION_WORDS = [
    "allergy",
    "allergic",
    "nut allergy",
    "tree nut",
    "peanut",
    "peanuts",
    "almond",
    "walnut",
    "cashew",
    "pistachio",
    "hazelnut",
    "milk allergy",
    "dairy allergy",
    "lactose intolerant",
    "egg allergy",
    "fish allergy",
    "shellfish allergy",
    "shrimp allergy",
    "wheat allergy",
    "gluten allergy",
    "soy allergy",
    "sesame allergy",
    "may contain nuts",
    "traces of nuts",
]

VEG_GOOD = [
    "vegetarian",
    "vegetarian options",
    "good for vegetarians",
    "lots of vegetarian options",
    "vegan",
    "vegan options",
    "vegan friendly",
    "veg options",
    "veg friendly",
    "vegetarian friendly",
    "plant based",
    "plant-based",
    "meatless options",
    "tofu",
    "paneer",
    "lentil curry",
    "chana masala",
    "falafel",
    "salad bar",
]

VEG_BAD = [
    "no vegetarian options",
    "no veg options",
    "not many veg options",
    "hardly any veg options",
    "only meat",
    "nothing vegetarian",
    "no vegan options",
    "not vegan friendly",
    "limited vegetarian choices",
    "limited veg choices",
    "no meatless options",
    "everything has meat",
    "all dishes have meat",
    "not veggie friendly",
    "poor vegetarian options",
    "bad for vegetarians",
]

VEG_MENTION_WORDS = list(set(VEG_GOOD + VEG_BAD))


# -------------------------------------------------
# Basic text clean
# -------------------------------------------------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"[\n\t]+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.drop(columns=["7514"], errors="ignore")
    df = df[df["Rating"] != "Like"].copy()
    df = df.dropna(subset=["Rating", "Review"])
    df["Rating"] = df["Rating"].astype(float)
    data = df[["Restaurant", "Reviewer", "Review", "Rating"]].copy()
    data["clean_review"] = data["Review"].apply(clean_text)
    return data


OVERALL_LABELS = {0: "Negative", 1: "Neutral", 2: "Positive"}
SERVICE_LABELS = {0: "Bad", 1: "Neutral", 2: "Good"}


def summarize_ratio(ratio: float) -> str:
    if ratio >= 0.8:
        return "High"
    elif ratio >= 0.5:
        return "Moderate"
    else:
        return "Low"


# -------------------------------------------------
# Interpretation helpers
# -------------------------------------------------
def overall_from_vader_and_model(text: str, model_probs_row: np.ndarray) -> int:
    """
    Decide final overall sentiment mostly from VADER, with model as backup.

    - Strongly positive / negative VADER overrides model.
    - In the middle range, use the model's prediction.
    """
    comp = vader_compound(text)  # -1 .. 1
    model_label = int(np.argmax(model_probs_row))

    if comp <= -0.5:
        return 0  # Negative
    if comp >= 0.5:
        return 2  # Positive

    # Mild sentiment -> trust the model
    return model_label


def interpret_health_label(text: str, p_ok: float) -> str:
    """
    Health:
      - If no health keywords -> 'None mentioned'
      - If low p_ok and a bad keyword -> '⚠️ Issue mentioned'
      - If high p_ok -> 'OK'
      - Else -> 'None mentioned'
    """
    t = text.lower()
    has_keyword = any(w in t for w in HEALTH_BAD_WORDS)

    if not has_keyword:
        return "None mentioned"

    if p_ok >= 0.7:
        return "OK"

    if p_ok < 0.4 and has_keyword:
        return "⚠️ Issue mentioned"

    return "None mentioned"


def interpret_allergy_label(text: str, p_safe: float) -> str:
    """
    Allergy:
      - If no allergy-related words -> 'None mentioned'
      - Else -> use model to decide Safe / Risk
    """
    t = text.lower()
    has_mention = any(w in t for w in ALLERGY_MENTION_WORDS)

    if not has_mention:
        return "None mentioned"

    if p_safe >= 0.5:
        return "Safe"
    else:
        return "⚠️ Risk mentioned"


def interpret_veg_label(text: str, p_veg_friendly: float) -> str:
    """
    Veg:
      - If no veg-related words -> 'None mentioned'
      - Else -> model + keywords for Veg-friendly vs Not
    """
    t = text.lower()
    has_mention = any(w in t for w in VEG_MENTION_WORDS)

    if not has_mention:
        return "None mentioned"

    # If text itself sounds negative about veg
    if any(w in t for w in VEG_BAD):
        return "Not veg-friendly"

    # If text sounds positive OR model is confident
    if any(w in t for w in VEG_GOOD) or p_veg_friendly >= 0.5:
        return "Veg-friendly"

    return "None mentioned"


# -------------------------------------------------
# Cached model & data
# -------------------------------------------------
@st.cache_resource
def load_model_cached():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_data_cached():
    return load_and_prepare_data(CSV_PATH)


model = load_model_cached()
data = load_data_cached()


# -------------------------------------------------
# Core analysis
# -------------------------------------------------
def analyze_restaurant(restaurant_name: str):
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

    inputs = tf.constant(reviews_clean, dtype=tf.string)
    preds = model.predict(inputs, batch_size=32, verbose=0)

    if isinstance(preds, dict):
        overall_probs = preds["overall_output"]
        service_probs = preds["service_output"]
        allergy_probs = preds["allergy_output"]
        health_probs = preds["health_output"]
        veg_probs = preds["veg_output"]
    else:
        overall_probs, service_probs, allergy_probs, health_probs, veg_probs = preds

    n = len(reviews_clean)

    # ---- Overall sentiment using VADER + model ----
    overall_final = []
    for i in range(n):
        label = overall_from_vader_and_model(reviews_raw[i], overall_probs[i])
        overall_final.append(label)
    overall_final = np.array(overall_final, dtype=np.int32)

    overall_counts = np.bincount(overall_final, minlength=3)
    overall_ratios = overall_counts / n

    # ---- Service from model directly ----
    service_pred = service_probs.argmax(axis=-1)
    service_counts = np.bincount(service_pred, minlength=3)
    service_ratios = service_counts / n

    # ---- Allergy / Health / Veg with "None mentioned" ----
    allergy_labels = []
    health_labels = []
    veg_labels = []

    for i in range(n):
        p_allergy = float(allergy_probs[i][0])
        p_health_ok = float(health_probs[i][0])
        p_veg = float(veg_probs[i][0])

        allergy_labels.append(interpret_allergy_label(reviews_raw[i], p_allergy))
        health_labels.append(interpret_health_label(reviews_raw[i], p_health_ok))
        veg_labels.append(interpret_veg_label(reviews_raw[i], p_veg))

    # Aggregate only over reviews that actually mention the factor
    def ratio_from_labels(labels, positive_values):
        mentioned = [lbl for lbl in labels if lbl != "None mentioned"]
        if not mentioned:
            return 0.0, 0, 0
        pos_count = sum(1 for lbl in mentioned if lbl in positive_values)
        total = len(mentioned)
        return pos_count / total, pos_count, total

    allergy_safe_ratio, allergy_safe_count, allergy_total_mentioned = ratio_from_labels(
        allergy_labels, positive_values={"Safe"}
    )
    health_ok_ratio, health_ok_count, health_total_mentioned = ratio_from_labels(
        health_labels, positive_values={"OK"}
    )
    veg_friendly_ratio, veg_friendly_count, veg_total_mentioned = ratio_from_labels(
        veg_labels, positive_values={"Veg-friendly"}
    )

    # --------- Display results ----------
    st.subheader("Overall Sentiment (VADER + Model)")
    for i in range(3):
        st.write(
            f"{OVERALL_LABELS[i]:>8}: {overall_counts[i]:3d} reviews "
            f"({overall_ratios[i] * 100:5.1f}%)"
        )
    dominant_overall = OVERALL_LABELS[int(overall_counts.argmax())]
    st.info(f"**Dominant overall sentiment:** {dominant_overall}")

    st.subheader("Service Sentiment (Model)")
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
        f"({allergy_safe_ratio * 100:5.1f}% of "
        f"{allergy_total_mentioned} review(s) that mention allergies are predicted safe; "
        f"{allergy_total_mentioned} of {n} reviews mention allergies)"
    )

    st.write(
        f"- Health standards: **{summarize_ratio(health_ok_ratio)}** "
        f"({health_ok_ratio * 100:5.1f}% of "
        f"{health_total_mentioned} review(s) that mention health issues are OK; "
        f"{health_total_mentioned} of {n} reviews mention health issues)"
    )

    st.write(
        f"- Vegetarian friendly: **{summarize_ratio(veg_friendly_ratio)}** "
        f"({veg_friendly_ratio * 100:5.1f}% of "
        f"{veg_total_mentioned} review(s) that mention veg options are veg-friendly; "
        f"{veg_total_mentioned} of {n} reviews mention veg options)"
    )

    st.subheader("Sample Reviews & Predictions")
    show_n = min(5, n)
    for i in range(show_n):
        st.markdown(f"**Review {i + 1}:**")
        txt = reviews_raw[i]
        txt_short = txt if len(txt) <= 500 else txt[:500] + " ..."
        st.write(txt_short)

        overall_label = OVERALL_LABELS[int(overall_final[i])]
        service_label = SERVICE_LABELS[int(service_pred[i])]
        allergy_label = allergy_labels[i]
        health_label = health_labels[i]
        veg_label = veg_labels[i]

        st.write(
            f"- Overall sentiment: **{overall_label}**\n"
            f"- Service sentiment: **{service_label}**\n"
            f"- Allergy: {allergy_label}\n"
            f"- Health: {health_label}\n"
            f"- Veg options: {veg_label}"
        )


# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
st.title("Restaurant Review Analyzer")

st.write(
    """
This app uses a multi-output deep learning model plus NLTK VADER to estimate:

- **Overall sentiment** (using VADER + model)  
- **Service sentiment** (model)  
- **Food-friendly factors** (allergies, health, vegetarian friendliness)

When a factor (like allergies) is **not mentioned** in a review,
it is shown as **“None mentioned”** and is **not counted** in that factor's score.
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
    matches = unique_restaurants[:50]

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
