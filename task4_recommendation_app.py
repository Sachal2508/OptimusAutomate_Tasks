"""
Task 4: Movie Recommendation System
CodeAlpha AI Internship
================================================
Dataset  : MovieLens-style (70 movies, 500 users, ~14,600 ratings)
Methods  :
  1. Collaborative Filtering — User-User Pearson Correlation
  2. Content-Based Filtering — Genre TF-IDF + Cosine Similarity
  3. Hybrid                  — Weighted blend of both
Metrics  : Cosine similarity, Pearson correlation
UI       : Streamlit with top-N recommendations + explanations
Run with : streamlit run recommendation_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import pearsonr
import os

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
)

st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }
.stApp { background-color: #0d1117; color: #e6edf3; }
.stSelectbox > div, .stSlider > div { color: #e6edf3; }
.rec-card {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 10px; padding: 1rem; margin-bottom: 0.6rem;
}
.rec-card:hover { border-color: #58a6ff; }
.genre-badge {
    background: #21262d; border: 1px solid #30363d;
    border-radius: 12px; padding: 2px 8px;
    font-size: 0.75rem; color: #8b949e;
    display: inline-block; margin: 2px;
}
.score-bar-bg { background:#21262d; border-radius:4px; height:8px; margin-top:4px; }
.score-bar    { background:linear-gradient(90deg,#58a6ff,#3fb950); border-radius:4px; height:8px; }
</style>
""", unsafe_allow_html=True)

# ── Load data ────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(__file__)


def resolve_movielens_path():
    # Prefer src/data/, then repo-level data/ml-latest-small, then data/ml-latest-small
    candidates = [
        os.path.join(SCRIPT_DIR, "data"),
        os.path.join(os.path.dirname(SCRIPT_DIR), "data", "ml-latest-small"),
        os.path.join(os.path.dirname(SCRIPT_DIR), "data", "ml-small"),
        os.path.join(os.path.dirname(SCRIPT_DIR), "data"),
    ]
    for path in candidates:
        movies_path = os.path.join(path, "movies.csv")
        ratings_path = os.path.join(path, "ratings.csv")
        if os.path.isfile(movies_path) and os.path.isfile(ratings_path):
            return path
    return None


def download_movielens(target_dir):
    import requests, zipfile, io
    url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
    r = requests.get(url, stream=True)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(os.path.dirname(target_dir) or '.')
    # extracted folder is ml-latest-small in the parent data folder
    extracted = os.path.join(os.path.dirname(target_dir), 'ml-latest-small')
    if os.path.isdir(extracted) and not os.path.isdir(target_dir):
        # move/rename to requested target_dir
        try:
            os.replace(extracted, target_dir)
        except Exception:
            pass


DATA_DIR = resolve_movielens_path()

@st.cache_data
def load_data():
    if DATA_DIR is None:
        raise FileNotFoundError("MovieLens data not found")
    movies = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))
    ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
    return movies, ratings

@st.cache_data
def build_matrices(movies, ratings):
    # User-item matrix (users × movies)
    user_item = ratings.pivot_table(
        index="userId", columns="movieId", values="rating"
    ).fillna(0)

    # Content matrix: TF-IDF on genres
    tfidf    = TfidfVectorizer(token_pattern=r"[^|]+")
    genre_mat = tfidf.fit_transform(movies["genres"].fillna(""))
    content_sim = cosine_similarity(genre_mat)
    content_sim_df = pd.DataFrame(
        content_sim, index=movies["movieId"], columns=movies["movieId"]
    )
    return user_item, content_sim_df

if DATA_DIR is None:
    st.error("MovieLens dataset not found.")
    st.markdown("The app expects `movies.csv` and `ratings.csv`. You can download the small MovieLens dataset below.")
    if st.button("Download MovieLens (small)"):
        with st.spinner("Downloading dataset..."):
            target = os.path.join(os.path.dirname(SCRIPT_DIR), 'data')
            os.makedirs(target, exist_ok=True)
            download_movielens(target)
            st.success("Download finished. Please rerun the app or refresh the page.")
    st.stop()

movies, ratings = load_data()
user_item, content_sim_df = build_matrices(movies, ratings)

# ── Helper: movie info ───────────────────────────────────────────
def movie_info(movie_id):
    row = movies[movies["movieId"] == movie_id].iloc[0]
    return row["title"], row["genres"]

def avg_rating(movie_id):
    r = ratings[ratings["movieId"] == movie_id]["rating"]
    return r.mean() if len(r) > 0 else 0.0

def num_ratings(movie_id):
    return len(ratings[ratings["movieId"] == movie_id])

# ── Collaborative Filtering (User-User Pearson) ──────────────────
def collaborative_recommend(user_id: int, top_n: int = 10):
    if user_id not in user_item.index:
        return []

    user_vec = user_item.loc[user_id]
    already_rated = set(user_vec[user_vec > 0].index)

    # Pearson correlation with all other users
    similarities = {}
    for other_id in user_item.index:
        if other_id == user_id:
            continue
        other_vec = user_item.loc[other_id]
        # Only compare on commonly rated movies
        common = (user_vec > 0) & (other_vec > 0)
        if common.sum() < 3:
            continue
        try:
            corr, _ = pearsonr(user_vec[common], other_vec[common])
            if not np.isnan(corr):
                similarities[other_id] = corr
        except Exception:
            continue

    if not similarities:
        return []

    # Weighted average of similar users' ratings
    top_similar = sorted(similarities.items(), key=lambda x: -x[1])[:30]
    scores = {}
    for other_id, sim in top_similar:
        if sim <= 0:
            continue
        other_ratings = user_item.loc[other_id]
        for mid, r in other_ratings.items():
            if r > 0 and mid not in already_rated:
                scores[mid] = scores.get(mid, 0) + sim * r

    # Normalize
    total_sim = sum(abs(s) for _, s in top_similar if s > 0) or 1
    scored = [(mid, sc / total_sim) for mid, sc in scores.items()]
    scored.sort(key=lambda x: -x[1])
    return scored[:top_n]

# ── Content-Based Filtering (Genre Cosine Similarity) ───────────
def content_recommend(movie_id: int, top_n: int = 10):
    if movie_id not in content_sim_df.index:
        return []
    sims = content_sim_df[movie_id].drop(movie_id).sort_values(ascending=False)
    return [(mid, score) for mid, score in sims.head(top_n).items()]

# ── Hybrid Recommender ───────────────────────────────────────────
def hybrid_recommend(user_id: int, liked_movie_id: int, top_n: int = 10, alpha: float = 0.5):
    collab = dict(collaborative_recommend(user_id, top_n=30))
    content = dict(content_recommend(liked_movie_id, top_n=30))

    all_movies = set(collab) | set(content)
    if user_id in user_item.index:
        already_rated = set(user_item.loc[user_id][user_item.loc[user_id] > 0].index)
    else:
        already_rated = set()

    scores = {}
    for mid in all_movies:
        if mid in already_rated:
            continue
        c_score  = collab.get(mid, 0)
        cb_score = content.get(mid, 0)
        # Normalise collaborative scores to [0,1]
        max_c = max(collab.values()) if collab else 1
        c_norm = c_score / max_c if max_c else 0
        scores[mid] = alpha * c_norm + (1 - alpha) * cb_score

    return sorted(scores.items(), key=lambda x: -x[1])[:top_n]

# ── Render recommendation cards ──────────────────────────────────
def render_cards(recs, method_label, score_label="Score"):
    if not recs:
        st.warning("Not enough data to generate recommendations. Try a different user or movie.")
        return
    for rank, (mid, score) in enumerate(recs, 1):
        title, genres = movie_info(mid)
        avg_r  = avg_rating(mid)
        n_r    = num_ratings(mid)
        badges = " ".join(f'<span class="genre-badge">{g}</span>' for g in genres.split("|"))
        bar_w  = min(int(score * 100), 100) if score <= 1 else min(int((score/5)*100), 100)
        stars  = "⭐" * round(avg_r)

        st.markdown(f"""
<div class="rec-card">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <span style="font-size:1.05rem;font-weight:700;color:#e6edf3;">#{rank} &nbsp; {title}</span>
    <span style="color:#3fb950;font-weight:600;font-size:0.95rem;">{score_label}: {score:.3f}</span>
  </div>
  <div style="margin:6px 0;">{badges}</div>
  <div style="color:#8b949e;font-size:0.82rem;">
    {stars} &nbsp; Avg rating: <b style="color:#e3b341;">{avg_r:.1f}/5</b>
    &nbsp;·&nbsp; {n_r} ratings
  </div>
  <div class="score-bar-bg"><div class="score-bar" style="width:{bar_w}%;"></div></div>
</div>
""", unsafe_allow_html=True)

# ── UI ───────────────────────────────────────────────────────────
st.markdown("<h1 style='text-align:center;color:#58a6ff;'>🎬 Movie Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#8b949e;'>Collaborative Filtering · Content-Based · Hybrid · Cosine Similarity · Pearson Correlation</p>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar controls
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    method = st.selectbox("Recommendation Method", [
        "🤝 Collaborative Filtering (User-User)",
        "🎭 Content-Based (Genre Similarity)",
        "🔀 Hybrid (Collaborative + Content)",
    ])

    top_n = st.slider("Top N Recommendations", 3, 15, 8)

    st.markdown("---")
    st.markdown("### 📊 Dataset Stats")
    st.metric("Total Movies", len(movies))
    st.metric("Total Users", ratings["userId"].nunique())
    st.metric("Total Ratings", len(ratings))
    st.metric("Avg Rating", f"{ratings['rating'].mean():.2f} / 5")

    st.markdown("---")
    st.markdown("### ℹ️ How it works")
    st.markdown("""
**Collaborative Filtering**
Finds users with similar taste (Pearson correlation) and recommends what they liked.

**Content-Based**
Recommends movies with similar genres using TF-IDF + Cosine Similarity.

**Hybrid**
Weighted blend (α·collaborative + (1-α)·content).
    """)

# ── Method: Collaborative ────────────────────────────────────────
if "Collaborative" in method:
    st.markdown("### 🤝 Collaborative Filtering — User-User Pearson Correlation")

    col1, col2 = st.columns([2, 1])
    with col1:
        user_id = st.selectbox("Select a User ID", sorted(ratings["userId"].unique()), index=0)
    with col2:
        if user_id in user_item.index:
            rated_count = (user_item.loc[user_id] > 0).sum()
            st.metric("Movies Rated by User", int(rated_count))

    if st.button("🎯 Get Recommendations", use_container_width=True):
        with st.spinner("Computing user similarities..."):
            recs = collaborative_recommend(user_id, top_n)

        # Show user's top-rated movies
        user_ratings = ratings[ratings["userId"] == user_id].sort_values("rating", ascending=False)
        user_movies  = user_ratings.head(5).merge(movies, on="movieId")

        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.markdown("**👤 User's Top Rated Movies**")
            for _, row in user_movies.iterrows():
                st.markdown(f"- **{row['title']}** — ⭐ {row['rating']}")

        with col_b:
            st.markdown(f"**🎬 Top {top_n} Recommended Movies**")
            render_cards(recs, "Collaborative Filtering", "Pearson Score")

# ── Method: Content-Based ────────────────────────────────────────
elif "Content-Based" in method:
    st.markdown("### 🎭 Content-Based Filtering — Genre Cosine Similarity")

    movie_titles = movies.set_index("movieId")["title"].to_dict()
    selected_movie = st.selectbox(
        "Select a movie you like:",
        options=movies["movieId"].tolist(),
        format_func=lambda x: movie_titles[x],
    )

    if st.button("🎯 Find Similar Movies", use_container_width=True):
        title, genres = movie_info(selected_movie)
        st.markdown(f"**Because you liked:** {title} · *{genres}*")

        recs = content_recommend(selected_movie, top_n)
        render_cards(recs, "Content-Based", "Genre Similarity")

        # Similarity matrix heatmap (top 15 movies)
        st.markdown("---")
        st.markdown("#### 🔥 Genre Similarity Heatmap (Top 15 Movies)")
        top15_ids = [selected_movie] + [mid for mid, _ in recs[:14]]
        top15_titles = [movie_titles[mid][:20] + "…" if len(movie_titles[mid]) > 20
                        else movie_titles[mid] for mid in top15_ids]
        sim_subset = content_sim_df.loc[top15_ids, top15_ids].values

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#161b22')
        sns.heatmap(sim_subset, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=top15_titles, yticklabels=top15_titles,
                    ax=ax, linewidths=0.5, linecolor='#0d1117', vmin=0, vmax=1)
        ax.set_title("Genre Cosine Similarity Matrix", color="white", fontsize=13, fontweight="bold")
        ax.tick_params(colors="#8b949e", labelsize=8)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ── Method: Hybrid ───────────────────────────────────────────────
else:
    st.markdown("### 🔀 Hybrid Recommendation System")

    col1, col2, col3 = st.columns(3)
    with col1:
        user_id = st.selectbox("Select User ID", sorted(ratings["userId"].unique()), index=0)
    with col2:
        movie_titles = movies.set_index("movieId")["title"].to_dict()
        liked_movie  = st.selectbox(
            "Select a Movie You Like",
            options=movies["movieId"].tolist(),
            format_func=lambda x: movie_titles[x],
        )
    with col3:
        alpha = st.slider("α — Collaborative Weight", 0.0, 1.0, 0.5, 0.1,
                          help="0 = full content-based, 1 = full collaborative")

    if st.button("🎯 Get Hybrid Recommendations", use_container_width=True):
        with st.spinner("Blending collaborative + content signals..."):
            recs = hybrid_recommend(user_id, liked_movie, top_n, alpha)

        col_info, col_recs = st.columns([1, 2])
        with col_info:
            st.markdown("**⚖️ Blend Weights**")
            st.markdown(f"- Collaborative: **{alpha:.0%}**")
            st.markdown(f"- Content-Based: **{1-alpha:.0%}**")

            user_ratings = ratings[ratings["userId"] == user_id].sort_values("rating", ascending=False)
            user_movies  = user_ratings.head(5).merge(movies, on="movieId")
            st.markdown(f"**👤 User {user_id}'s Top Rated**")
            for _, row in user_movies.iterrows():
                st.markdown(f"- {row['title']} ⭐{row['rating']}")

        with col_recs:
            st.markdown(f"**🎬 Top {top_n} Hybrid Recommendations**")
            render_cards(recs, "Hybrid", "Hybrid Score")

# ── Rating distribution chart ────────────────────────────────────
st.markdown("---")
st.markdown("#### 📊 Dataset Overview")

col_a, col_b = st.columns(2)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with col_a:
    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#161b22')
    counts = ratings["rating"].value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.values, color='#58a6ff', edgecolor='#0d1117')
    ax.set_title("Rating Distribution", color='white', fontweight='bold')
    ax.set_xlabel("Rating", color='#8b949e'); ax.set_ylabel("Count", color='#8b949e')
    ax.tick_params(colors='#8b949e')
    for sp in ax.spines.values(): sp.set_edgecolor('#30363d')
    plt.tight_layout(); st.pyplot(fig); plt.close()

with col_b:
    top_movies = (ratings.groupby("movieId")["rating"]
                  .agg(["mean","count"])
                  .query("count >= 50")
                  .sort_values("mean", ascending=False)
                  .head(10)
                  .merge(movies[["movieId","title"]], on="movieId"))

    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#161b22')
    short_titles = [t[:18]+"…" if len(t) > 18 else t for t in top_movies["title"]]
    ax.barh(short_titles[::-1], top_movies["mean"].values[::-1],
            color='#3fb950', edgecolor='#0d1117')
    ax.set_title("Top Rated Movies (≥50 ratings)", color='white', fontweight='bold')
    ax.set_xlabel("Avg Rating", color='#8b949e')
    ax.tick_params(colors='#8b949e', labelsize=8)
    ax.set_xlim(3, 5)
    for sp in ax.spines.values(): sp.set_edgecolor('#30363d')
    plt.tight_layout(); st.pyplot(fig); plt.close()
