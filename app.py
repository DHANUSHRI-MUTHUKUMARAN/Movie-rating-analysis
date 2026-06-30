import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="🎬 Movie Rating Analysis",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Rating Analysis Dashboard")
st.markdown(
    """
This dashboard analyzes the MovieLens dataset using:

- 📊 Exploratory Data Analysis
- 🎭 Genre Analysis
- 😊 Sentiment Analysis
- 🤖 Movie Recommendation System
"""
)

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    movies = pd.read_csv("data/processed/movies_enriched.csv")
    ratings = pd.read_csv("data/raw/ratings.csv")
    sentiment = pd.read_csv("data/processed/sentiment_results.csv")
    return movies, ratings, sentiment

movies, ratings, sentiment = load_data()

# -----------------------------
# Load Model
# -----------------------------
model_path = "models/svd_recommender.pkl"

if os.path.exists(model_path):
    svd = joblib.load(model_path)
else:
    svd = None
    st.warning("Recommendation model not found.")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Filters")

genres = sorted(
    set(
        g
        for genre_list in movies["genres"].dropna()
        for g in genre_list.split("|")
    )
)

selected_genre = st.sidebar.selectbox(
    "Select Genre",
    genres
)

min_rating = st.sidebar.slider(
    "Minimum Rating",
    0.5,
    5.0,
    3.5,
    0.5
)

# -----------------------------
# KPI Cards
# -----------------------------
st.header("📊 Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Movies",
    movies["movieId"].nunique()
)

c2.metric(
    "Users",
    ratings["userId"].nunique()
)

c3.metric(
    "Ratings",
    len(ratings)
)

c4.metric(
    "Average Rating",
    round(ratings["rating"].mean(), 2)
)

st.divider()

# -----------------------------
# Movie Browser
# -----------------------------
st.header("🎬 Movie Browser")

filtered_movies = movies[
    (movies["avg_rating"] >= min_rating)
    &
    (movies["genres"].str.contains(selected_genre))
]

st.dataframe(
    filtered_movies[
        [
            "title",
            "genres",
            "avg_rating",
            "num_ratings"
        ]
    ].drop_duplicates(),
    use_container_width=True
)

# -----------------------------
# Rating Distribution
# -----------------------------
st.header("⭐ Rating Distribution")

fig = px.histogram(
    ratings,
    x="rating",
    nbins=10,
    title="Rating Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------
# Genre Distribution
# -----------------------------
st.header("🎭 Movies by Genre")

genre_df = movies.copy()

genre_df["genres"] = genre_df["genres"].str.split("|")

genre_df = genre_df.explode("genres")

genre_count = (
    genre_df.groupby("genres")
    .size()
    .reset_index(name="Movies")
)

fig = px.bar(
    genre_count,
    x="genres",
    y="Movies",
    color="Movies",
    title="Number of Movies per Genre"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------
# Sentiment Analysis
# -----------------------------
st.header("😊 Sentiment Analysis")

fig = px.scatter(
    sentiment,
    x="avg_sentiment",
    y="avg_rating",
    hover_name="title",
    title="Sentiment vs Average Rating"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------
# Recommendation Function
# -----------------------------
def recommend_movies(user_id, n=10):

    watched = ratings[
        ratings["userId"] == user_id
    ]["movieId"].tolist()

    unseen = movies[
        ~movies["movieId"].isin(watched)
    ]["movieId"]

    predictions = []

    for movie in unseen:

        prediction = svd.predict(
            user_id,
            movie
        )

        predictions.append(
            (
                movie,
                prediction.est
            )
        )

    predictions = sorted(
        predictions,
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = pd.DataFrame(
        predictions[:n],
        columns=[
            "movieId",
            "Predicted Rating"
        ]
    )

    recommendations = recommendations.merge(
        movies,
        on="movieId"
    )

    return recommendations[
        [
            "title",
            "genres",
            "Predicted Rating"
        ]
    ]

# -----------------------------
# Recommendation Section
# -----------------------------
st.header("🤖 Personalized Movie Recommendation")

if svd is not None:

    max_user = int(ratings["userId"].max())

    user_id = st.number_input(
        "Enter User ID",
        min_value=1,
        max_value=max_user,
        value=1
    )

    if st.button("Recommend Movies"):

        recommendations = recommend_movies(user_id)

        st.success(
            f"Top 10 recommendations for User {user_id}"
        )

        st.dataframe(
            recommendations,
            use_container_width=True
        )

else:

    st.info(
        "Train the SVD model first to enable recommendations."
    )

# -----------------------------
# Footer
# -----------------------------
st.divider()

st.markdown(
    """
---
### Developed By

**Movie Rating Analysis Project**

Technologies Used

- Python
- Pandas
- Plotly
- Streamlit
- Scikit-Learn
- Surprise (SVD)
- TextBlob
- NLTK
"""
)