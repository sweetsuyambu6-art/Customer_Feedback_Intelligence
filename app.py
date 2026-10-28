import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).parent
SAMPLE_PATH = BASE_DIR / "data" / "sample_feedback.csv"

st.set_page_config(page_title="Customer Feedback Intelligence", page_icon="💬", layout="wide")

@st.cache_data
def load_sample():
    return pd.read_csv(SAMPLE_PATH)

@st.cache_resource
def train_demo_model():
    data = load_sample()
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=12000, stop_words="english")),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])
    model.fit(data["feedback"].astype(str), data["sentiment"].astype(str))
    return model

def normalize_text(value):
    if pd.isna(value):
        return ""
    value = str(value).strip()
    value = re.sub(r"\s+", " ", value)
    return value

def find_column(columns, candidates):
    lower_map = {str(c).strip().lower(): c for c in columns}
    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]
    return None

st.title("💬 Customer Feedback Intelligence")
st.caption("Turn customer reviews into sentiment insights and actionable themes.")

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload customer feedback CSV", type=["csv"])
    st.markdown("**Required:** a text column containing customer feedback.")
    st.markdown("Optional columns: rating, product, date, customer_id.")
    use_sample = st.checkbox("Use included demo dataset", value=uploaded is None)
    st.divider()
    st.caption("Demo model: TF-IDF + Logistic Regression trained on the included illustrative sample. Validate with a larger, representative labeled dataset before business use.")

try:
    model = train_demo_model()
except Exception as exc:
    st.error(f"Could not load the demo model: {exc}")
    st.stop()

if uploaded is not None and not use_sample:
    try:
        df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        st.stop()
else:
    df = load_sample().copy()

if df.empty:
    st.warning("The selected dataset is empty.")
    st.stop()

text_col = find_column(df.columns, ["feedback", "review", "text", "comment", "customer_feedback", "review_text"])
if text_col is None:
    st.error("Could not identify the feedback text column. Rename it to one of: feedback, review, text, comment, customer_feedback, review_text.")
    st.write("Columns found:", list(df.columns))
    st.stop()

df[text_col] = df[text_col].map(normalize_text)
df = df[df[text_col].ne("")].copy()
if df.empty:
    st.warning("No non-empty feedback rows were found.")
    st.stop()

with st.spinner("Analyzing feedback..."):
    df["predicted_sentiment"] = model.predict(df[text_col])
    try:
        probs = model.predict_proba(df[text_col])
        df["confidence"] = probs.max(axis=1).round(3)
    except Exception:
        df["confidence"] = None

rating_col = find_column(df.columns, ["rating", "stars", "score", "customer_rating"])
product_col = find_column(df.columns, ["product", "product_name", "item", "category"])
date_col = find_column(df.columns, ["date", "created_at", "review_date", "timestamp"])

total = len(df)
positive = int((df["predicted_sentiment"] == "Positive").sum())
negative = int((df["predicted_sentiment"] == "Negative").sum())
neutral = int((df["predicted_sentiment"] == "Neutral").sum())
negative_pct = (negative / total * 100) if total else 0

m1, m2, m3, m4 = st.columns(4)
m1.metric("Feedback analyzed", f"{total:,}")
m2.metric("Positive", f"{positive:,}", f"{positive/total*100:.1f}%" if total else "0%")
m3.metric("Negative", f"{negative:,}", f"{negative_pct:.1f}% of feedback" if total else "0%")
m4.metric("Neutral", f"{neutral:,}", f"{neutral/total*100:.1f}%" if total else "0%")

left, right = st.columns(2)
with left:
    sentiment_counts = df["predicted_sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"], fill_value=0).rename_axis("Sentiment").reset_index(name="Count")
    st.subheader("Sentiment distribution")
    st.plotly_chart(px.pie(sentiment_counts, names="Sentiment", values="Count", hole=0.48), use_container_width=True)
with right:
    st.subheader("Sentiment by product")
    if product_col:
        product_summary = df.groupby([product_col, "predicted_sentiment"]).size().reset_index(name="Count")
        st.plotly_chart(px.bar(product_summary, x=product_col, y="Count", color="predicted_sentiment", barmode="stack"), use_container_width=True)
    else:
        st.info("Add a `product` column to compare sentiment by product.")

if date_col:
    parsed_dates = pd.to_datetime(df[date_col], errors="coerce")
    trend = df.assign(_date=parsed_dates).dropna(subset=["_date"])
    if not trend.empty:
        trend["_month"] = trend["_date"].dt.to_period("M").astype(str)
        trend_counts = trend.groupby(["_month", "predicted_sentiment"]).size().reset_index(name="Count")
        st.subheader("Feedback sentiment over time")
        st.plotly_chart(px.line(trend_counts, x="_month", y="Count", color="predicted_sentiment", markers=True), use_container_width=True)

if rating_col:
    df[rating_col] = pd.to_numeric(df[rating_col], errors="coerce")
    rating_summary = df.dropna(subset=[rating_col]).groupby("predicted_sentiment")[rating_col].mean().reset_index()
    if not rating_summary.empty:
        st.subheader("Average rating by predicted sentiment")
        st.plotly_chart(px.bar(rating_summary, x="predicted_sentiment", y=rating_col, color="predicted_sentiment"), use_container_width=True)

st.subheader("Common terms in negative feedback")
negative_text = " ".join(df.loc[df["predicted_sentiment"] == "Negative", text_col].astype(str)).lower()
tokens = re.findall(r"\b[a-z][a-z']{2,}\b", negative_text)
stopwords = {"the", "and", "for", "was", "with", "this", "that", "but", "not", "are", "have", "had", "has", "very", "too", "from", "they", "you", "your", "its", "our", "their", "product", "service"}
freq = pd.Series([t for t in tokens if t not in stopwords]).value_counts().head(15)
if not freq.empty:
    terms = freq.rename_axis("Term").reset_index(name="Mentions")
    st.plotly_chart(px.bar(terms.sort_values("Mentions"), x="Mentions", y="Term", orientation="h"), use_container_width=True)
else:
    st.info("No negative feedback terms found in the current dataset.")

st.subheader("Explore analyzed feedback")
sentiment_filter = st.multiselect("Filter by sentiment", ["Positive", "Neutral", "Negative"], default=["Positive", "Neutral", "Negative"])
filtered = df[df["predicted_sentiment"].isin(sentiment_filter)].copy()
display_cols = [text_col, "predicted_sentiment", "confidence"] + [c for c in [rating_col, product_col, date_col] if c and c not in [text_col]]
st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)

csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button("Download analyzed feedback (CSV)", data=csv_bytes, file_name="analyzed_customer_feedback.csv", mime="text/csv")
st.caption("Important: sentiment predictions are model estimates, not ground truth. Review sample predictions and evaluate performance on labeled data before using them for operational decisions.")
