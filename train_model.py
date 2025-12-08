"""
Train a multi-output restaurant review model and save it.

Uses:
- NLTK VADER + rating to create better overall sentiment labels
- Keyword-based labels for service, allergy, health, vegetarian-friendliness

Outputs:
- restaurant_review_model.keras
"""

import re
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

import tensorflow as tf
from tensorflow.keras import layers, Model

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# =========================================================
# 1. CONFIG
# =========================================================

CSV_PATH = "Restaurant reviews.csv"  # must match your file name

RANDOM_SEED = 42
TEST_SIZE = 0.2
MAX_TOKENS = 20000
MAX_SEQUENCE_LENGTH = 200
EMBEDDING_DIM = 128
BATCH_SIZE = 64
EPOCHS = 5

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# =========================================================
# 2. NLTK / VADER SETUP
# =========================================================

try:
    sia = SentimentIntensityAnalyzer()
except LookupError:
    nltk.download("vader_lexicon")
    sia = SentimentIntensityAnalyzer()


def vader_compound(text: str) -> float:
    if not isinstance(text, str):
        text = str(text)
    return float(sia.polarity_scores(text)["compound"])


# =========================================================
# 3. KEYWORD LISTS
# =========================================================

ALLERGY_BAD_WORDS = [
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
    "cross contamination",
    "cross-contamination",
    "may contain nuts",
    "traces of nuts",
    "allergic reaction",
    "broke out in hives",
    "anaphylactic",
]

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

POS_SERVICE_WORDS = [
    "friendly",
    "very friendly",
    "attentive",
    "very attentive",
    "polite",
    "professional",
    "welcoming",
    "great service",
    "excellent service",
    "amazing service",
    "fast service",
    "quick service",
    "prompt service",
    "prompt",
    "checked on us often",
    "refilled our drinks",
    "helpful",
    "accommodating",
    "kind staff",
    "courteous",
    "smiling staff",
    "made us feel welcome",
    "went above and beyond",
    "super nice",
]

NEG_SERVICE_WORDS = [
    "rude",
    "very rude",
    "unfriendly",
    "ignored us",
    "slow service",
    "very slow",
    "took forever",
    "bad service",
    "terrible service",
    "horrible service",
    "unprofessional",
    "never checked on us",
    "did not care",
    "didn't care",
    "forgot our order",
    "messed up our order",
    "brought the wrong dish",
    "cold food",
    "served late",
    "dirty table",
    "filthy",
    "disgusting",
    "staff yelling",
    "arguing with customers",
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
    "beyond burger",
    "impossible burger",
    "salad bar",
    "plenty of veggies",
    "great vegetarian menu",
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
    "no salad",
    "everything has meat",
    "all dishes have meat",
    "bacon on everything",
    "not veggie friendly",
    "poor vegetarian options",
    "bad for vegetarians",
    "unfriendly to vegans",
    "no plant based options",
    "lack of vegetarian choices",
]


# =========================================================
# 4. TEXT & LABEL HELPERS
# =========================================================

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"[\n\t]+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def auto_label_overall(rating: float, text: str) -> int:
    """
    Better overall sentiment using BOTH rating and VADER.

    Classes:
      0 = negative
      1 = neutral
      2 = positive
    """
    rating = float(rating)
    comp = vader_compound(text)  # -1..1

    # Strong text overrides weird ratings
    if comp <= -0.5:
        return 0
    if comp >= 0.5:
        return 2

    # Otherwise use rating as a strong prior
    if rating <= 2:
        # but if text is mildly positive, allow positive
        if comp >= 0.2:
            return 2
        return 0
    if rating >= 4:
        # but if text is mildly negative, allow negative
        if comp <= -0.2:
            return 0
        return 2

    # rating == 3 or something weird -> use softer thresholds
    if comp <= -0.2:
        return 0
    if comp >= 0.2:
        return 2
    return 1


def auto_label_service_sentiment(text: str) -> int:
    """
    Service sentiment:
      0 = bad
      1 = neutral / unclear
      2 = good
    """
    t = str(text).lower()
    if any(w in t for w in NEG_SERVICE_WORDS):
        return 0
    if any(w in t for w in POS_SERVICE_WORDS):
        return 2
    return 1


def auto_label_allergy(text: str) -> int:
    t = str(text).lower()
    if any(w in t for w in ALLERGY_BAD_WORDS):
        return 0
    return 1


def auto_label_health(text: str) -> int:
    t = str(text).lower()
    if any(w in t for w in HEALTH_BAD_WORDS):
        return 0
    return 1


def auto_label_veg(text: str) -> int:
    t = str(text).lower()
    if any(w in t for w in VEG_BAD):
        return 0
    if any(w in t for w in VEG_GOOD):
        return 1
    return 1  # default to "OK" if nothing obvious


# =========================================================
# 5. LOAD & PREP DATA
# =========================================================

def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Drop weird numeric column if present
    df = df.drop(columns=["7514"], errors="ignore")

    # Remove non-numeric rating rows like "Like"
    df = df[df["Rating"] != "Like"].copy()

    # Drop rows with missing rating or review
    df = df.dropna(subset=["Rating", "Review"])

    df["Rating"] = df["Rating"].astype(float)

    data = df[["Restaurant", "Reviewer", "Review", "Rating"]].copy()

    # Labels
    data["overall_sentiment"] = data.apply(
        lambda row: auto_label_overall(row["Rating"], row["Review"]),
        axis=1,
    )
    data["service_sentiment"] = data["Review"].apply(auto_label_service_sentiment)
    data["allergy_safe"] = data["Review"].apply(auto_label_allergy)
    data["health_standard_ok"] = data["Review"].apply(auto_label_health)
    data["vegetarian_friendly"] = data["Review"].apply(auto_label_veg)

    data["clean_review"] = data["Review"].apply(clean_text)

    return data


# =========================================================
# 6. BUILD MODEL
# =========================================================

def build_model(train_texts: np.ndarray) -> Model:
    vectorizer = layers.TextVectorization(
        max_tokens=MAX_TOKENS,
        output_mode="int",
        output_sequence_length=MAX_SEQUENCE_LENGTH,
        name="text_vectorizer",
    )
    vectorizer.adapt(train_texts)

    text_input = layers.Input(shape=(), dtype=tf.string, name="text")
    x = vectorizer(text_input)
    x = layers.Embedding(
        input_dim=len(vectorizer.get_vocabulary()),
        output_dim=EMBEDDING_DIM,
        mask_zero=True,
        name="embedding",
    )(x)
    x = layers.Bidirectional(layers.LSTM(96, return_sequences=True), name="bilstm_1")(x)
    x = layers.Bidirectional(layers.LSTM(64), name="bilstm_2")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu", name="shared_dense")(x)

    overall_output = layers.Dense(3, activation="softmax", name="overall_output")(x)
    service_output = layers.Dense(3, activation="softmax", name="service_output")(x)
    allergy_output = layers.Dense(1, activation="sigmoid", name="allergy_output")(x)
    health_output = layers.Dense(1, activation="sigmoid", name="health_output")(x)
    veg_output = layers.Dense(1, activation="sigmoid", name="veg_output")(x)

    model = Model(
        inputs=text_input,
        outputs={
            "overall_output": overall_output,
            "service_output": service_output,
            "allergy_output": allergy_output,
            "health_output": health_output,
            "veg_output": veg_output,
        },
        name="restaurant_review_multi_output_model",
    )

    model.compile(
        optimizer="adam",
        loss={
            "overall_output": "sparse_categorical_crossentropy",
            "service_output": "sparse_categorical_crossentropy",
            "allergy_output": "binary_crossentropy",
            "health_output": "binary_crossentropy",
            "veg_output": "binary_crossentropy",
        },
        metrics={
            "overall_output": ["accuracy"],
            "service_output": ["accuracy"],
            "allergy_output": ["accuracy"],
            "health_output": ["accuracy"],
            "veg_output": ["accuracy"],
        },
    )

    return model


# =========================================================
# 7. TRAIN MODEL + EVALUATE
# =========================================================

def train_model(data: pd.DataFrame):
    X = data["clean_review"].values

    y_overall = data["overall_sentiment"].astype("int32").values
    y_service = data["service_sentiment"].astype("int32").values
    y_allergy = data["allergy_safe"].astype("float32").values
    y_health = data["health_standard_ok"].astype("float32").values
    y_veg = data["vegetarian_friendly"].astype("float32").values

    (
        X_train,
        X_val,
        y_overall_train,
        y_overall_val,
        y_service_train,
        y_service_val,
        y_allergy_train,
        y_allergy_val,
        y_health_train,
        y_health_val,
        y_veg_train,
        y_veg_val,
    ) = train_test_split(
        X,
        y_overall,
        y_service,
        y_allergy,
        y_health,
        y_veg,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_overall,
    )

    model = build_model(X_train)

    history = model.fit(
        X_train,
        {
            "overall_output": y_overall_train,
            "service_output": y_service_train,
            "allergy_output": y_allergy_train,
            "health_output": y_health_train,
            "veg_output": y_veg_train,
        },
        validation_data=(
            X_val,
            {
                "overall_output": y_overall_val,
                "service_output": y_service_val,
                "allergy_output": y_allergy_val,
                "health_output": y_health_val,
                "veg_output": y_veg_val,
            },
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
    )

    # ==============================
    #  EVALUATION ON HELD-OUT 20%
    # ==============================
    print("\n=== EVALUATION ON HELD-OUT SET (20%) ===")

    # Predict on X_val
    val_preds = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0)

    # Handle dict vs list/tuple outputs
    if isinstance(val_preds, dict):
        overall_probs = val_preds["overall_output"]
        service_probs = val_preds["service_output"]
        allergy_probs = val_preds["allergy_output"]
        health_probs = val_preds["health_output"]
        veg_probs = val_preds["veg_output"]
    else:
        overall_probs, service_probs, allergy_probs, health_probs, veg_probs = val_preds

    # -------- Overall sentiment (3-class) --------
    y_overall_pred = overall_probs.argmax(axis=-1).astype("int32")

    print("\n--- Overall sentiment ---")
    print("Accuracy:", accuracy_score(y_overall_val, y_overall_pred))
    print("\nClassification report (overall):")
    print(classification_report(y_overall_val, y_overall_pred))
    print("Confusion matrix (overall):")
    print(confusion_matrix(y_overall_val, y_overall_pred))

    # -------- Service sentiment (3-class) --------
    y_service_pred = service_probs.argmax(axis=-1).astype("int32")

    print("\n--- Service sentiment ---")
    print("Accuracy:", accuracy_score(y_service_val, y_service_pred))
    print("\nClassification report (service):")
    print(classification_report(y_service_val, y_service_pred))
    print("Confusion matrix (service):")
    print(confusion_matrix(y_service_val, y_service_pred))

    # -------- Allergy safety (binary) --------
    y_allergy_pred = (allergy_probs.flatten() >= 0.5).astype("int32")

    print("\n--- Allergy safety (binary) ---")
    print("Accuracy:", accuracy_score(y_allergy_val.astype("int32"), y_allergy_pred))
    print("\nClassification report (allergy):")
    print(classification_report(y_allergy_val.astype("int32"), y_allergy_pred))
    print("Confusion matrix (allergy):")
    print(confusion_matrix(y_allergy_val.astype("int32"), y_allergy_pred))

    # -------- Health standards (binary) --------
    y_health_pred = (health_probs.flatten() >= 0.5).astype("int32")

    print("\n--- Health standards (binary) ---")
    print("Accuracy:", accuracy_score(y_health_val.astype("int32"), y_health_pred))
    print("\nClassification report (health):")
    print(classification_report(y_health_val.astype("int32"), y_health_pred))
    print("Confusion matrix (health):")
    print(confusion_matrix(y_health_val.astype("int32"), y_health_pred))

    # -------- Veg-friendly (binary) --------
    y_veg_pred = (veg_probs.flatten() >= 0.5).astype("int32")

    print("\n--- Vegetarian-friendly (binary) ---")
    print("Accuracy:", accuracy_score(y_veg_val.astype("int32"), y_veg_pred))
    print("\nClassification report (veg-friendly):")
    print(classification_report(y_veg_val.astype("int32"), y_veg_pred))
    print("Confusion matrix (veg-friendly):")
    print(confusion_matrix(y_veg_val.astype("int32"), y_veg_pred))

    return model, history


# =========================================================
# 8. MAIN
# =========================================================

if __name__ == "__main__":
    print("Loading and preparing data...")
    data = load_and_prepare_data(CSV_PATH)
    print("Data shape:", data.shape)

    print("\nTraining multi-output model (with NLTK VADER labels)...")
    model, history = train_model(data)
    print("\nTraining complete.")

    MODEL_PATH = "restaurant_review_model.keras"
    model.save(MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")
