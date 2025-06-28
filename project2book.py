
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# ------------------- Load and Preprocess Data -------------------
@st.cache(allow_output_mutation=True)
def load_data():
    df_basic = pd.read_csv("C:/Users/Mithrra Sree/Downloads/Audible_Catlog.csv")
    df_adv = pd.read_csv("C:/Users/Mithrra Sree/Downloads/Audible_Catlog_Advanced_Features.csv")

    # Merge datasets
    df = pd.merge(df_basic, df_adv, on=["Book Name", "Author"], how="inner")
    df.drop(columns=["Rating_y", "Number of Reviews_y", "Price_y"], inplace=True)
    df.rename(columns={
        "Rating_x": "Rating",
        "Number of Reviews_x": "Number of Reviews",
        "Price_x": "Price"
    }, inplace=True)

    # Clean
    df.dropna(subset=["Book Name", "Author", "Description"], inplace=True)
    df["Rating"].fillna(df["Rating"].mean(), inplace=True)
    df["Number of Reviews"].fillna(0, inplace=True)
    df["Price"].fillna(df["Price"].median(), inplace=True)
    df.drop_duplicates(inplace=True)
    df["Description"] = df["Description"].fillna("")
    df.reset_index(drop=True, inplace=True)
    return df

df_merged = load_data()

# ------------------- NLP + Clustering -------------------
@st.cache(allow_output_mutation=True)
def prepare_model(df):
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df["Description"])

    kmeans = KMeans(n_clusters=5, random_state=42)
    df["Cluster"] = kmeans.fit_predict(tfidf_matrix)

    return tfidf, tfidf_matrix, df

tfidf, tfidf_matrix, df_merged = prepare_model(df_merged)

book_indices = pd.Series(df_merged.index, index=df_merged['Book Name']).drop_duplicates()

# ------------------- Recommendation Functions -------------------
def content_based_recommend(book_name, top_n=5):
    idx = book_indices.get(book_name, None)
    if idx is None:
        return ["Book not found."]
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_indices = sim_scores.argsort()[::-1][1:top_n+1]
    return df_merged['Book Name'].iloc[sim_indices].tolist()

def cluster_based_recommend(book_name, top_n=5):
    idx = book_indices.get(book_name, None)
    if idx is None:
        return ["Book not found."]
    book_cluster = df_merged.loc[idx, 'Cluster']
    cluster_books = df_merged[df_merged['Cluster'] == book_cluster]
    recs = cluster_books[cluster_books['Book Name'] != book_name]['Book Name'].head(top_n)
    return recs.tolist()

def hybrid_recommend(book_name, top_n=5):
    idx = book_indices.get(book_name, None)
    if idx is None:
        return ["Book not found."]
    
    book_cluster = df_merged.loc[idx, 'Cluster']
    cluster_df = df_merged[df_merged['Cluster'] == book_cluster].copy().reset_index(drop=True)
    cluster_tfidf = tfidf.transform(cluster_df["Description"])
    
    book_desc = df_merged.loc[idx, "Description"]
    book_vec = tfidf.transform([book_desc])
    
    sim_scores = cosine_similarity(book_vec, cluster_tfidf).flatten()
    sorted_indices = sim_scores.argsort()[::-1][1:top_n+1]
    
    return cluster_df.iloc[sorted_indices]['Book Name'].tolist()

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="Audible Insights", layout="centered")
st.title("Audible Insights: Intelligent Book Recommendations ")

# Book selection
book_list = df_merged['Book Name'].sort_values().unique().tolist()
selected_book = st.selectbox("Select a book you like:", book_list)

# Recommendation type
rec_type = st.radio("Choose a recommendation method:", ["Content-Based", "Cluster-Based", "Hybrid"])

# Trigger
if st.button("Get Recommendations"):
    if rec_type == "Content-Based":
        results = content_based_recommend(selected_book)
    elif rec_type == "Cluster-Based":
        results = cluster_based_recommend(selected_book)
    else:
        results = hybrid_recommend(selected_book)

    st.subheader("Recommended Books:")
    for i, book in enumerate(results, 1):
        st.markdown(f"**{i}.** {book}")
