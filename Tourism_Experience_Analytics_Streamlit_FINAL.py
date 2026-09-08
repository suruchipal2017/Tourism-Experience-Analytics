
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Tourism Experience Analytics",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at 10% 10%, #dbeafe 0%, transparent 30%),
        radial-gradient(circle at 90% 10%, #ede9fe 0%, transparent 30%),
        linear-gradient(135deg, #f8fbff 0%, #eef2ff 50%, #faf5ff 100%);
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}
.hero {
    background: linear-gradient(135deg, #172554, #1d4ed8, #6d28d9);
    padding: 35px 40px;
    border-radius: 25px;
    margin-bottom: 25px;
    box-shadow: 0 15px 40px rgba(30, 64, 175, 0.25);
}
.hero h1 {
    color: white;
    font-size: 42px;
    margin-bottom: 8px;
    font-weight: 800;
}
.hero p {
    color: #dbeafe;
    font-size: 18px;
    margin-bottom: 0;
}
.section-title {
    font-size: 27px;
    font-weight: 800;
    color: #172554;
    margin-top: 25px;
    margin-bottom: 15px;
}
.input-card, .result-card, .info-card, .recommend-card {
    background: rgba(255,255,255,0.94);
    border-radius: 20px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 10px 30px rgba(15,23,42,0.08);
}
.input-card { padding: 25px; }
.result-card { padding: 25px; min-height: 145px; }
.result-label { color: #64748b; font-size: 15px; font-weight: 600; }
.result-value { color: #172554; font-size: 30px; font-weight: 800; margin-top: 8px; }
.result-sub { color: #475569; font-size: 14px; margin-top: 7px; }
.info-card { padding: 20px; height: 100%; }
.info-number { font-size: 27px; font-weight: 800; color: #1d4ed8; }
.info-label { color: #64748b; font-size: 14px; }
.recommend-card { padding: 20px; margin: 12px 0; }
.rank {
    font-size: 25px;
    font-weight: 800;
    color: #2563eb;
    min-width: 55px;
    text-align: center;
}
.attraction-name { font-size: 21px; font-weight: 750; color: #172554; }
.score {
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    color: white;
    border-radius: 14px;
    padding: 10px 15px;
    min-width: 85px;
    text-align: center;
    font-weight: 700;
}
.badge {
    display: inline-block;
    padding: 5px 10px;
    border-radius: 20px;
    background: #eff6ff;
    color: #1d4ed8;
    font-size: 12px;
    font-weight: 700;
    margin-top: 8px;
}
.stButton > button {
    width: 100%;
    border-radius: 14px;
    padding: 12px 20px;
    font-size: 17px;
    font-weight: 750;
    border: none;
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    color: white;
}
.footer {
    text-align: center;
    color: #64748b;
    padding: 30px 10px 10px;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# PATHS
# ============================================================

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

CLASSIFICATION_MODEL_PATH = os.path.join(
    BASE_PATH, "tourism_random_forest_200_maxcompressed_v2.joblib"
)

CLASSIFICATION_METADATA_PATH = os.path.join(
    BASE_PATH, "tourism_visitmode_preprocessing.joblib"
)

REGRESSION_MODEL_PATH = os.path.join(
    BASE_PATH, "tourism_gradient_boosting_regression_model.joblib"
)

TRANSACTION_PATH = os.path.join(BASE_PATH, "Transaction (1).xlsx")
ITEM_PATH = os.path.join(BASE_PATH, "Item.xlsx")
MODE_PATH = os.path.join(BASE_PATH, "Mode (1).xlsx")
TYPE_PATH = os.path.join(BASE_PATH, "Type.xlsx")
CITY_PATH = os.path.join(BASE_PATH, "City.xlsx")
COUNTRY_PATH = os.path.join(BASE_PATH, "Country.xlsx")
REGION_PATH = os.path.join(BASE_PATH, "Region.xlsx")

# ============================================================
# LOAD MODELS AND DATA
# ============================================================

@st.cache_resource
def load_models():
    classification_model = joblib.load(CLASSIFICATION_MODEL_PATH)
    regression_model = joblib.load(REGRESSION_MODEL_PATH)
    classification_metadata = joblib.load(CLASSIFICATION_METADATA_PATH)

    return (
        classification_model,
        regression_model,
        classification_metadata
    )


@st.cache_data
def load_data():
    transaction = pd.read_excel(TRANSACTION_PATH)
    item = pd.read_excel(ITEM_PATH)
    mode = pd.read_excel(MODE_PATH)
    attraction_type = pd.read_excel(TYPE_PATH)
    city = pd.read_excel(CITY_PATH)
    country = pd.read_excel(COUNTRY_PATH)
    region = pd.read_excel(REGION_PATH)

    return (
        transaction,
        item,
        mode,
        attraction_type,
        city,
        country,
        region
    )


try:
    (
        classification_model,
        regression_model,
        classification_metadata
    ) = load_models()

    (
        transaction,
        item,
        mode,
        attraction_type,
        city,
        country,
        region
    ) = load_data()

except Exception as e:
    st.error(f"Application files could not be loaded: {e}")
    st.stop()


# ============================================================
# MAPPINGS
# ============================================================

mode_map = {
    1: "Business",
    2: "Couples",
    3: "Family",
    4: "Friends",
    5: "Solo"
}

mode_reverse = {
    value: key for key, value in mode_map.items()
}

type_map = {
    2: "Ancient Ruins",
    10: "Ballets",
    13: "Beaches",
    19: "Caverns & Caves",
    34: "Flea & Street Markets",
    44: "Historic Sites",
    45: "History Museums",
    61: "National Parks",
    63: "Nature & Wildlife Areas",
    64: "Neighborhoods",
    72: "Points of Interest & Landmarks",
    76: "Religious Sites",
    82: "Spas",
    84: "Speciality Museums",
    91: "Volcanos",
    92: "Water Parks",
    93: "Waterfalls"
}


# ============================================================
# PREPARE DATA
# ============================================================

transaction.columns = transaction.columns.astype(str).str.strip()
item.columns = item.columns.astype(str).str.strip()
city.columns = city.columns.astype(str).str.strip()
country.columns = country.columns.astype(str).str.strip()
region.columns = region.columns.astype(str).str.strip()

for col in [
    "UserId",
    "AttractionId",
    "VisitYear",
    "VisitMonth",
    "VisitMode",
    "Rating"
]:
    if col in transaction.columns:
        transaction[col] = pd.to_numeric(
            transaction[col], errors="coerce"
        )

for col in ["AttractionId", "AttractionTypeId", "AttractionCityId"]:
    if col in item.columns:
        item[col] = pd.to_numeric(item[col], errors="coerce")

city["CityId"] = pd.to_numeric(city["CityId"], errors="coerce")
city["CountryId"] = pd.to_numeric(city["CountryId"], errors="coerce")
country["CountryId"] = pd.to_numeric(country["CountryId"], errors="coerce")
country["RegionId"] = pd.to_numeric(country["RegionId"], errors="coerce")
region["RegionId"] = pd.to_numeric(region["RegionId"], errors="coerce")
region["ContinentId"] = pd.to_numeric(region["ContinentId"], errors="coerce")

city["CityName"] = city["CityName"].fillna("Unknown City")

item_city = item.merge(
    city[["CityId", "CityName", "CountryId"]],
    left_on="AttractionCityId",
    right_on="CityId",
    how="left"
)

item_city["CityName"] = item_city["CityName"].fillna("Unknown Location")

item_city = item_city.merge(
    country[["CountryId", "RegionId"]],
    on="CountryId",
    how="left",
    suffixes=("", "_Country")
)

item_city = item_city.merge(
    region[["RegionId", "Region"]],
    on="RegionId",
    how="left"
)

item_city["Region"] = item_city["Region"].fillna("Unknown Region")


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">
<h1>🌍 Tourism Experience Analytics</h1>
<p>
AI-powered Visit Mode Prediction, Attraction Rating Prediction
and Personalized Tourism Recommendations
</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🌟 Tourism AI")
    st.markdown("---")
    st.write(f"📊 Transactions: **{len(transaction):,}**")
    st.write(f"👤 Users: **{transaction['UserId'].nunique():,}**")
    st.write(f"🏝️ Attractions: **{item['AttractionId'].nunique():,}**")
    st.markdown("---")
    st.write("🌲 Random Forest — VisitMode")
    st.write("📈 Gradient Boosting — Rating")
    st.write("🤝 Hybrid Recommendation Engine")


# ============================================================
# ANALYTICS DASHBOARD — REQUIRED BY PROJECT DOCUMENT
# ============================================================

st.markdown(
    '<div class="section-title">📊 Tourism Analytics Dashboard</div>',
    unsafe_allow_html=True
)

dash1, dash2, dash3, dash4 = st.columns(4)

with dash1:
    st.markdown(
        f"""
<div class="info-card">
<div class="info-number">{len(transaction):,}</div>
<div class="info-label">Historical Transactions</div>
</div>
""",
        unsafe_allow_html=True
    )

with dash2:
    st.markdown(
        f"""
<div class="info-card">
<div class="info-number">{transaction["UserId"].nunique():,}</div>
<div class="info-label">Users</div>
</div>
""",
        unsafe_allow_html=True
    )

with dash3:
    st.markdown(
        f"""
<div class="info-card">
<div class="info-number">{item["AttractionId"].nunique():,}</div>
<div class="info-label">Attractions</div>
</div>
""",
        unsafe_allow_html=True
    )

with dash4:
    st.markdown(
        f"""
<div class="info-card">
<div class="info-number">⭐ {transaction["Rating"].mean():.2f}</div>
<div class="info-label">Average Rating</div>
</div>
""",
        unsafe_allow_html=True
    )

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    popular_dashboard = (
        transaction.groupby("AttractionId")
        .size()
        .sort_values(ascending=False)
        .head(10)
        .rename("Visits")
        .to_frame()
        .merge(
            item[["AttractionId", "Attraction"]],
            left_index=True,
            right_on="AttractionId",
            how="left"
        )
        .set_index("Attraction")["Visits"]
    )

    st.markdown("**🏝️ Popular Attractions**")
    st.bar_chart(popular_dashboard)

with chart_col2:
    user_region_dashboard = (
        transaction[["UserId"]]
        .drop_duplicates()
        .merge(
            pd.read_excel(COUNTRY_PATH)[["CountryId", "RegionId"]],
            left_on="UserId",
            right_on="CountryId",
            how="left"
        )
        if False else None
    )

    # User segments are based on historical interaction count.
    user_counts = (
        transaction.groupby("UserId")
        .size()
        .rename("Interactions")
    )

    user_segments = pd.cut(
        user_counts,
        bins=[0, 1, 3, 5, np.inf],
        labels=[
            "1 Interaction",
            "2-3 Interactions",
            "4-5 Interactions",
            "6+ Interactions"
        ]
    ).value_counts().sort_index()

    st.markdown("**👥 User Segments**")
    st.bar_chart(user_segments)

# Top regions based on attraction visit activity.
region_dashboard = (
    transaction[["AttractionId"]]
    .merge(
        item_city[["AttractionId", "Region"]],
        on="AttractionId",
        how="left"
    )
    .groupby("Region")
    .size()
    .sort_values(ascending=False)
    .head(10)
)

st.markdown("**🌎 Top Regions by Attraction Visits**")
st.bar_chart(region_dashboard)

st.markdown("---")


# ============================================================
# INPUT
# ============================================================

st.markdown(
    '<div class="section-title">🧳 Plan Your Tourism Experience</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="input-card">',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    user_id = st.number_input(
        "👤 User ID",
        min_value=1,
        value=1,
        step=1
    )

with col2:
    visit_year = st.number_input(
        "📅 Visit Year",
        min_value=2013,
        max_value=2030,
        value=2022,
        step=1
    )

with col3:
    visit_month = st.selectbox(
        "🗓️ Visit Month",
        list(range(1, 13)),
        index=6
    )

location_names = sorted(
    item_city["CityName"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

selected_location = st.selectbox(
    "📍 Location",
    location_names
)

location_attractions = sorted(
    item_city[
        item_city["CityName"] == selected_location
    ]["Attraction"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

if not location_attractions:
    location_attractions = sorted(
        item["Attraction"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

selected_attraction = st.selectbox(
    "🏝️ Attraction",
    location_attractions
)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

analyze = st.button(
    "🚀 Analyze My Tourism Experience"
)


# ============================================================
# ANALYSIS
# ============================================================

if analyze:
    attraction_row = item[
        item["Attraction"].astype(str) == selected_attraction
    ]

    if attraction_row.empty:
        st.error("Selected attraction was not found.")
        st.stop()

    attraction_row = attraction_row.iloc[0]

    attraction_id = int(attraction_row["AttractionId"])
    attraction_type_id = int(attraction_row["AttractionTypeId"])
    selected_city_id = int(attraction_row["AttractionCityId"])

    # Model was trained with historical information up to 2019.
    # Use only that information for deployment consistency.
    model_history_pool = transaction[
        transaction["VisitYear"] <= 2019
    ].copy()

    # If the requested period is within the training period,
    # keep only information before the selected year/month.
    if visit_year <= 2019:
        model_history_pool = model_history_pool[
            (model_history_pool["VisitYear"] < visit_year)
            |
            (
                (model_history_pool["VisitYear"] == visit_year)
                &
                (model_history_pool["VisitMonth"] < visit_month)
            )
        ]

    user_history = model_history_pool[
        model_history_pool["UserId"] == int(user_id)
    ].copy()

    is_new_user = user_history.empty

    # --------------------------------------------------------
    # CLASSIFICATION INPUT
    # Preferred Visit Mode is intentionally NOT an input here.
    # The classifier predicts the likely visit mode from user,
    # attraction, visit-time and historical features.
    # --------------------------------------------------------

    class_feature_names = classification_metadata["feature_names"]
    scale_features = classification_metadata["scale_features"]
    scaler = classification_metadata["scaler"]

    class_input = pd.DataFrame(
        0.0,
        index=[0],
        columns=class_feature_names
    )

    class_input.loc[0, "VisitYear"] = visit_year
    class_input.loc[0, "VisitMonth"] = visit_month

    quarter = ((visit_month - 1) // 3) + 1

    if "VisitQuarter" in class_input.columns:
        class_input.loc[0, "VisitQuarter"] = quarter

    if visit_month in [12, 1, 2]:
        season = "Winter"
    elif visit_month in [3, 4, 5]:
        season = "Spring"
    elif visit_month in [6, 7, 8]:
        season = "Summer"
    else:
        season = "Autumn"

    season_column = f"Season_{season}"

    if season_column in class_input.columns:
        class_input.loc[0, season_column] = 1

    type_column = f"AttractionType_{attraction_type_id}"

    if type_column in class_input.columns:
        class_input.loc[0, type_column] = 1

    # User history
    user_values = {
        "TotalInteractions_TrainUser": len(user_history),
        "UniqueAttractions_TrainUser": user_history["AttractionId"].nunique(),
        "UniqueVisitModes_User_TrainUser": user_history["VisitMode"].nunique(),
        "UniqueVisitYears_TrainUser": user_history["VisitYear"].nunique(),
        "UniqueVisitMonths_User_TrainUser": user_history["VisitMonth"].nunique()
    }

    for col, value in user_values.items():
        if col in class_input.columns:
            class_input.loc[0, col] = value

    # Attraction history
    attraction_history = model_history_pool[
        model_history_pool["AttractionId"] == attraction_id
    ]

    attraction_values = {
        "TotalInteractions_Attraction_TrainAttraction":
            len(attraction_history),
        "UniqueUsers_TrainAttraction":
            attraction_history["UserId"].nunique(),
        "UniqueVisitModes_Attraction_TrainAttraction":
            attraction_history["VisitMode"].nunique(),
        "UniqueVisitYears_Attraction_TrainAttraction":
            attraction_history["VisitYear"].nunique(),
        "UniqueVisitMonths_Attraction_TrainAttraction":
            attraction_history["VisitMonth"].nunique()
    }

    for col, value in attraction_values.items():
        if col in class_input.columns:
            class_input.loc[0, col] = value

    # Pair history
    pair_history = model_history_pool[
        (model_history_pool["UserId"] == int(user_id))
        &
        (model_history_pool["AttractionId"] == attraction_id)
    ]

    pair_interactions = len(pair_history)

    if pair_interactions > 0:
        first_pair_year = int(pair_history["VisitYear"].min())
        last_pair_year = int(pair_history["VisitYear"].max())
        pair_months = int(pair_history["VisitMonth"].nunique())
        pair_modes = int(pair_history["VisitMode"].nunique())
    else:
        first_pair_year = 0
        last_pair_year = 0
        pair_months = 0
        pair_modes = 0

    pair_values = {
        "InteractionCount_TrainPair": pair_interactions,
        "FirstVisitYear_TrainPair": first_pair_year,
        "LastVisitYear_TrainPair": last_pair_year,
        "UniqueVisitMonths_TrainPair": pair_months,
        "UniqueVisitModes_TrainPair": pair_modes
    }

    for col, value in pair_values.items():
        if col in class_input.columns:
            class_input.loc[0, col] = value

    class_input = class_input[class_feature_names]

    class_input_scaled = class_input.copy()

    available_scale_features = [
        col for col in scale_features
        if col in class_input_scaled.columns
    ]

    if available_scale_features:
        class_input_scaled[available_scale_features] = (
            scaler.transform(
                class_input[available_scale_features]
            )
        )

    # --------------------------------------------------------
    # VISIT MODE PREDICTION
    # --------------------------------------------------------

    predicted_visit_mode_id = int(
        classification_model.predict(
            class_input_scaled
        )[0]
    )

    predicted_visit_mode = mode_map.get(
        predicted_visit_mode_id,
        str(predicted_visit_mode_id)
    )

    if hasattr(classification_model, "predict_proba"):
        mode_probabilities = classification_model.predict_proba(
            class_input_scaled
        )[0]
        mode_confidence = float(mode_probabilities.max())
    else:
        mode_probabilities = None
        mode_confidence = None

    # --------------------------------------------------------
    # REGRESSION INPUT
    # --------------------------------------------------------

    reg_feature_names = list(
        regression_model.feature_names_in_
    )

    reg_input = pd.DataFrame(
        0.0,
        index=[0],
        columns=reg_feature_names
    )

    reg_input.loc[0, "VisitYear"] = visit_year
    reg_input.loc[0, "VisitMonth"] = visit_month
    reg_input.loc[0, "VisitQuarter"] = quarter

    visit_mode_column = f"VisitMode_{predicted_visit_mode_id}"
    if visit_mode_column in reg_input.columns:
        reg_input.loc[0, visit_mode_column] = 1

    if season_column in reg_input.columns:
        reg_input.loc[0, season_column] = 1

    if type_column in reg_input.columns:
        reg_input.loc[0, type_column] = 1

    # Prior-only historical features
    for col, value in user_values.items():
        if col in reg_input.columns:
            reg_input.loc[0, col] = value

    for col, value in attraction_values.items():
        if col in reg_input.columns:
            reg_input.loc[0, col] = value

    for col, value in pair_values.items():
        if col in reg_input.columns:
            reg_input.loc[0, col] = value

    reg_input = reg_input[reg_feature_names]

    predicted_rating = float(
        regression_model.predict(reg_input)[0]
    )

    predicted_rating = float(
        np.clip(predicted_rating, 1, 5)
    )

    # --------------------------------------------------------
    # RESULT CARDS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">✨ AI Prediction Results</div>',
        unsafe_allow_html=True
    )

    r1, r2, r3 = st.columns(3)

    confidence_text = (
        f"{mode_confidence * 100:.1f}%"
        if mode_confidence is not None
        else "N/A"
    )

    # Use native Streamlit components here instead of HTML so the
    # prediction values are rendered reliably in every Streamlit/Colab setup.
    with r1:
        with st.container(border=True):
            st.markdown("### 🧠 Predicted Visit Mode")
            st.markdown(f"## {predicted_visit_mode}")
            st.caption(classification_name)

    with r2:
        with st.container(border=True):
            st.markdown("### 📈 Predicted Attraction Rating")
            st.markdown(f"## ⭐ {predicted_rating:.2f}/5")
            st.caption("Gradient Boosting regression")

    with r3:
        with st.container(border=True):
            st.markdown("### 🎯 Visit Mode Confidence")
            st.markdown(f"## {confidence_text}")
            st.caption("Highest class probability")

    # --------------------------------------------------------
    # USER PROFILE
    # --------------------------------------------------------

    if is_new_user:
        st.info(
            "✨ New user detected. Recommendations use the selected "
            "location, predicted visit mode, attraction quality and "
            "historical popularity."
        )
    else:
        st.success(
            f"👤 Returning user detected. Prior interactions used: "
            f"{len(user_history)}"
        )

        st.markdown(
            '<div class="section-title">👤 Your Tourism Profile</div>',
            unsafe_allow_html=True
        )

        p1, p2, p3, p4 = st.columns(4)

        with p1:
            st.markdown(
                f"""
<div class="info-card">
<div class="info-number">{len(user_history)}</div>
<div class="info-label">Prior Visits</div>
</div>
""",
                unsafe_allow_html=True
            )

        with p2:
            user_avg_rating = float(
                user_history["Rating"].mean()
            )

            st.markdown(
                f"""
<div class="info-card">
<div class="info-number">⭐ {user_avg_rating:.2f}</div>
<div class="info-label">Average Rating</div>
</div>
""",
                unsafe_allow_html=True
            )

        with p3:
            unique_attractions = (
                user_history["AttractionId"].nunique()
            )

            st.markdown(
                f"""
<div class="info-card">
<div class="info-number">{unique_attractions}</div>
<div class="info-label">Attractions Visited</div>
</div>
""",
                unsafe_allow_html=True
            )

        with p4:
            favorite_mode_id = user_history["VisitMode"].mode()

            favorite_mode = (
                mode_map.get(
                    int(favorite_mode_id.iloc[0]),
                    "Unknown"
                )
                if not favorite_mode_id.empty
                else "Unknown"
            )

            st.markdown(
                f"""
<div class="info-card">
<div class="info-number">{favorite_mode}</div>
<div class="info-label">Frequent Visit Mode</div>
</div>
""",
                unsafe_allow_html=True
            )

    # --------------------------------------------------------
    # HYBRID PERSONALIZED RECOMMENDATIONS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">🏝️ Personalized Attraction Recommendations</div>',
        unsafe_allow_html=True
    )

    train_data = transaction[
        transaction["VisitYear"] <= 2019
    ].copy()

    attraction_stats = (
        train_data
        .groupby("AttractionId")
        .agg(
            AverageRating=("Rating", "mean"),
            TotalInteractions=("Rating", "count"),
            UniqueUsers=("UserId", "nunique")
        )
        .reset_index()
    )

    attraction_stats = attraction_stats.merge(
        item[
            [
                "AttractionId",
                "Attraction",
                "AttractionCityId",
                "AttractionTypeId"
            ]
        ],
        on="AttractionId",
        how="left"
    )

    max_interactions = max(
        attraction_stats["TotalInteractions"].max(),
        1
    )

    attraction_stats["PopularityScore"] = (
        attraction_stats["TotalInteractions"]
        / max_interactions
    )

    attraction_stats["QualityScore"] = (
        attraction_stats["AverageRating"] / 5.0
    )

    # User history used for personalization
    visited_ids = set(
        user_history["AttractionId"]
        .dropna()
        .astype(int)
    )

    recommendations = attraction_stats[
        ~attraction_stats["AttractionId"].isin(visited_ids)
    ].copy()

    # Collaborative score using item-item similarity
    user_item = (
        train_data
        .pivot_table(
            index="UserId",
            columns="AttractionId",
            values="Rating",
            aggfunc="mean",
            fill_value=0
        )
    )

    from sklearn.metrics.pairwise import cosine_similarity

    item_ids = user_item.columns.tolist()

    if len(item_ids) > 0:
        similarity = cosine_similarity(user_item.T)
        similarity_df = pd.DataFrame(
            similarity,
            index=item_ids,
            columns=item_ids
        )
    else:
        similarity_df = pd.DataFrame()

    recommendations["CollaborativeScore"] = 0.0

    if not user_history.empty and not similarity_df.empty:

        history_ratings = (
            user_history
            .groupby("AttractionId")["Rating"]
            .mean()
            .to_dict()
        )

        profile_items = [
            int(aid)
            for aid in history_ratings
            if aid in similarity_df.index
        ]

        if profile_items:

            collab_scores = []

            for candidate_id in recommendations["AttractionId"]:

                score_sum = 0.0
                weight_sum = 0.0

                for seen_id in profile_items:

                    similarity_value = float(
                        similarity_df.loc[
                            seen_id,
                            candidate_id
                        ]
                    )

                    rating_weight = max(
                        float(
                            history_ratings[seen_id]
                        ) / 5.0,
                        0.2
                    )

                    score_sum += (
                        similarity_value
                        * rating_weight
                    )

                    weight_sum += rating_weight

                collab_scores.append(
                    score_sum / weight_sum
                    if weight_sum > 0
                    else 0.0
                )

            recommendations["CollaborativeScore"] = (
                collab_scores
            )

    # Content score: selected location + selected attraction type
    recommendations["ContentScore"] = (
        0.60
        * (
            recommendations["AttractionCityId"]
            == selected_city_id
        ).astype(float)
        +
        0.40
        * (
            recommendations["AttractionTypeId"]
            == attraction_type_id
        ).astype(float)
    )

    # Predicted visit-mode affinity
    mode_counts = (
        train_data[
            train_data["VisitMode"] == predicted_visit_mode_id
        ]
        .groupby("AttractionId")
        .size()
        .rename("ModeVisits")
        .reset_index()
    )

    recommendations = recommendations.merge(
        mode_counts,
        on="AttractionId",
        how="left"
    )

    recommendations["ModeVisits"] = (
        recommendations["ModeVisits"].fillna(0)
    )

    max_mode_visits = max(
        recommendations["ModeVisits"].max(),
        1
    )

    recommendations["ModeAffinity"] = (
        recommendations["ModeVisits"]
        / max_mode_visits
    )

    # Hybrid final score
    if is_new_user:
        recommendations["PersonalizedScore"] = (
            0.30 * recommendations["ContentScore"]
            + 0.30 * recommendations["ModeAffinity"]
            + 0.25 * recommendations["QualityScore"]
            + 0.15 * recommendations["PopularityScore"]
        )

        method_text = (
            "Cold-start content + VisitMode + "
            "quality/popularity"
        )

    else:
        recommendations["PersonalizedScore"] = (
            0.40 * recommendations["CollaborativeScore"]
            + 0.20 * recommendations["ContentScore"]
            + 0.15 * recommendations["ModeAffinity"]
            + 0.15 * recommendations["QualityScore"]
            + 0.10 * recommendations["PopularityScore"]
        )

        method_text = (
            "Hybrid collaborative + content + "
            "VisitMode + quality/popularity"
        )

    recommendations = (
        recommendations
        .sort_values(
            [
                "PersonalizedScore",
                "QualityScore",
                "PopularityScore"
            ],
            ascending=False
        )
        .head(5)
        .reset_index(drop=True)
    )

    recommendations["Rank"] = (
        recommendations.index + 1
    )

    st.info(
        f"✨ Recommendation method: {method_text}. "
        "Previously visited attractions are excluded."
    )

    if recommendations.empty:
        st.warning(
            "No new attractions are available for recommendation."
        )
    else:

        for _, rec in recommendations.iterrows():

            attraction_name = str(rec["Attraction"])
            avg_rating = float(rec["AverageRating"])
            interactions = int(rec["TotalInteractions"])
            score = float(rec["PersonalizedScore"])

            attraction_type = type_map.get(
                int(rec["AttractionTypeId"])
                if not pd.isna(rec["AttractionTypeId"])
                else 0,
                "Tourist Attraction"
            )

            rank = int(rec["Rank"])

            card_html = (
                '<div class="recommend-card">'
                '<div style="display:flex;align-items:center;gap:18px;">'
                f'<div class="rank">#{rank}</div>'
                '<div style="flex:1;">'
                f'<div class="attraction-name">🏝️ {attraction_name}</div>'
                '<div style="margin-top:7px;color:#64748b;">'
                f'⭐ Average Rating: <b>{avg_rating:.2f}/5</b>'
                '&nbsp;&nbsp; | &nbsp;&nbsp;'
                f'👥 Historical Visits: <b>{interactions:,}</b>'
                '</div>'
                f'<div class="badge">🏷️ {attraction_type}</div>'
                '</div>'
                f'<div class="score">Score<br>{score:.3f}</div>'
                '</div>'
                '</div>'
            )

            st.markdown(
                card_html,
                unsafe_allow_html=True
            )

    # --------------------------------------------------------
    # PROBABILITY CHART
    # --------------------------------------------------------

    if mode_probabilities is not None:

        st.markdown(
            '<div class="section-title">📊 Visit Mode Probability</div>',
            unsafe_allow_html=True
        )

        probability_df = pd.DataFrame(
            {
                "Visit Mode": [
                    mode_map.get(
                        int(cls),
                        str(cls)
                    )
                    for cls in classification_model.classes_
                ],
                "Probability": mode_probabilities
            }
        ).set_index("Visit Mode")

        st.bar_chart(probability_df)

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">🔎 Analysis Summary</div>',
        unsafe_allow_html=True
    )

    s1, s2, s3, s4 = st.columns(4)

    with s1:
        st.markdown(
            f"""
<div class="info-card">
<div class="info-number">{visit_year}</div>
<div class="info-label">Visit Year</div>
</div>
""",
            unsafe_allow_html=True
        )

    with s2:
        st.markdown(
            f"""
<div class="info-card">
<div class="info-number">{selected_location}</div>
<div class="info-label">Location</div>
</div>
""",
            unsafe_allow_html=True
        )

    with s3:
        st.markdown(
            f"""
<div class="info-card">
<div class="info-number">{predicted_visit_mode}</div>
<div class="info-label">Predicted Visit Mode</div>
</div>
""",
            unsafe_allow_html=True
        )

    with s4:
        st.markdown(
            f"""
<div class="info-card">
<div class="info-number">🏝️</div>
<div class="info-label">{selected_attraction}</div>
</div>
""",
            unsafe_allow_html=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div class="footer">
🌍 Tourism Experience Analytics |
Classification • Prediction • Recommendation
<br>
Built with Python, Machine Learning and Streamlit
</div>
""",
    unsafe_allow_html=True
)
