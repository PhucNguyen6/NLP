import os
import csv
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEAN_DATA = os.path.join(BASE_DIR, "clean_data", "clean_comment.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(path: str = None) -> pd.DataFrame:
    """Load cleaned comment data from CSV file."""
    df = pd.read_csv(path, encoding="utf-8")
    df = df.dropna(subset=["comment"])
    df = df[df["comment"].str.strip() != ""]
    df = df.reset_index(drop=True)
    
    print(f"Loaded {len(df)} cleaned comments from {path}")
    return df

def build_vectorizer(corpus: pd.Series) -> TfidfVectorizer:
    """Build TF-IDF vectorizer from corpus of comments."""
    vectorizer =  TfidfVectorizer(
        ngram_range= (1, 2),  # Unigrams and bigrams
        max_features= 10000,  # Limit to top 10k features
        min_df= 2,           # Ignore terms in fewer than 5 documents
        max_df= 0.85,         # Ignore terms in more than 80%
        sublinear_tf= True,   # Use sublinear TF scaling
        analyzer= "word",     # Analyze at word level
        token_pattern= r"\b\w+\b",  # Token pattern to include single characters
    )
    return vectorizer

def fit_transform(vectorizer: TfidfVectorizer, corpus: pd.Series) -> np.ndarray:
    """Fit TF-IDF vectorizer to corpus and return document-term matrix."""
    tfidf_matrix = vectorizer.fit_transform(corpus)
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Number of features: {len(vectorizer.get_feature_names_out())}")
    return tfidf_matrix

def save_outputs(vectorizer: TfidfVectorizer, tfidf_matrix: np.ndarray, output_dir: str = None):
    feature_names = vectorizer.get_feature_names_out()
    idf_scores    = vectorizer.idf_

    # 1. Ma trận TF-IDF
    dense_matrix = tfidf_matrix.toarray()          # chuyển sparse → dense numpy array
    matrix_df    = pd.DataFrame(dense_matrix, columns=feature_names)
    matrix_df.insert(0, "video_id", df["video_id"].values)
    matrix_path  = os.path.join(OUTPUT_DIR, "tfidf_matrix.csv")
    matrix_df.to_csv(matrix_path, index=False, encoding="utf-8")
    print(f"\n[save] Đã lưu ma trận TF-IDF → {matrix_path}")

    # 2. Vocabulary + IDF
    vocab_df   = pd.DataFrame({"term": feature_names, "idf": idf_scores})
    vocab_df   = vocab_df.sort_values("idf", ascending=False).reset_index(drop=True)
    vocab_path = os.path.join(OUTPUT_DIR, "vocabulary.csv")
    vocab_df.to_csv(vocab_path, index=False, encoding="utf-8")
    print(f"[save] Đã lưu vocabulary      → {vocab_path}")

    # 3. Top terms mỗi video
    rows = []
    for vid in df["video_id"].unique():
        mask       = df["video_id"] == vid
        sub_matrix = tfidf_matrix[mask.values]
        mean_scores = np.asarray(sub_matrix.mean(axis=0)).flatten()
        top_idx     = np.argsort(mean_scores)[::-1][:10]
        for rank, idx in enumerate(top_idx, 1):
            rows.append({
                "video_id" : vid,
                "rank"     : rank,
                "term"     : feature_names[idx],
                "avg_tfidf": round(mean_scores[idx], 6),
            })

    top_terms_df   = pd.DataFrame(rows)
    top_terms_path = os.path.join(OUTPUT_DIR, "top_terms_per_video.csv")
    top_terms_df.to_csv(top_terms_path, index=False, encoding="utf-8")
    print(f"[save] Đã lưu top terms       → {top_terms_path}")
    
if(__name__ == "__main__"):
    df = load_data(CLEAN_DATA)
    corpus = df["comment"].tolist()
    vectorizer = build_vectorizer(df["comment"])
    tfidf_matrix = fit_transform(vectorizer, df["comment"])
    save_outputs(vectorizer, tfidf_matrix, OUTPUT_DIR)