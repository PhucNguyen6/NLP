import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report, silhouette_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

DEFAULT_INPUT = os.path.join(
    os.path.dirname(__file__), "labeled_data", "labeled_comment.csv"
)

plots_dir = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(plots_dir, exist_ok=True)

results_dir = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(results_dir, exist_ok=True)

df = pd.read_csv(DEFAULT_INPUT)

texts = df["comment"].astype(str).tolist()
labels = df["sentiment_label"].tolist()

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(labels)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(
    max_features=500, # 500 features
    ngram_range=(1,2) # unigrams and bigrams
)

X_tfidf = tfidf.fit_transform(texts).toarray()

# Doc2Vec Vectorization
tagged_data = [
    TaggedDocument(words=text.split(), tags=[str(i)])
    for i, text in enumerate(texts)
]

doc2vec_model = Doc2Vec(
    vector_size=100, # 100D vector
    window=5, # 5 words context
    min_count=1, # include all words
    workers=4, # 4 CPU cores
    epochs=40
)

doc2vec_model.build_vocab(tagged_data)
doc2vec_model.train(tagged_data, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)

X_doc2vec = [doc2vec_model.infer_vector(text.split()) for text in texts]

# Combine TF-IDF and Doc2Vec
X_combined = np.hstack((X_tfidf, X_doc2vec))

# Train 70%, còn lại 30%
X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
    X_combined, y_encoded, np.arange(len(texts)), test_size=0.3, random_state=42
)

# Chia tiếp 30% → val 10%, test 20% 
X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
    X_temp, y_temp, idx_temp, test_size=2/3, random_state=42
)

# KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
train_clusters = kmeans.fit_predict(X_train)

model_path = os.path.join(plots_dir, "kmeans_model.pkl")
joblib.dump(kmeans, model_path)

print(f"Saved KMeans model to: {model_path}")

# Map cluster labels to true sentiment labels
cluster_map = {}

for cluster_id in np.unique(train_clusters):
    idx = np.where(train_clusters == cluster_id)[0]
    true_labels = y_train[idx]
    
    most_common = np.bincount(true_labels).argmax()
    cluster_map[cluster_id] = most_common

print("Cluster mapping:", cluster_map)

# Predict on test set
test_clusters = kmeans.predict(X_test)
y_pred = np.array([cluster_map[c] for c in test_clusters])

# Evaluation
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save metrics to CSV
report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)

report_df = pd.DataFrame(report).transpose()

metrics_path = os.path.join(results_dir, "metrics.csv")
report_df.to_csv(metrics_path)

print(f"Saved metrics to: {metrics_path}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\n=== Confusion Matrix ===")
print(cm)

# Silhouette Score
sil_score = silhouette_score(X_train, train_clusters)
print("\nSilhouette Score:", sil_score)

# Save output to CSV
output_df = pd.DataFrame({
    "text": [texts[i] for i in idx_test],
    "true_label": le.inverse_transform(y_test),
    "predicted_label": le.inverse_transform(y_pred)
})

output_path = os.path.join(results_dir, "cluster_results.csv")
output_df.to_csv(output_path, index=False)

print(f"\nSaved results to: {output_path}")

# Plot confusion matrix
plt.figure(figsize=(6,5))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()

tick_marks = np.arange(len(le.classes_))
plt.xticks(tick_marks, le.classes_)
plt.yticks(tick_marks, le.classes_)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()

cm_path = os.path.join(plots_dir, "confusion_matrix.png")
plt.savefig(cm_path)
plt.close()

print(f"Saved confusion matrix to: {cm_path}")

# Cluster visualization using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train)

plt.figure()
for cluster_id in np.unique(train_clusters):
    idx = train_clusters == cluster_id
    plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=f"Cluster {cluster_id}")

plt.title("Clustering Visualization (PCA)")
plt.legend()

cluster_plot_path = os.path.join(plots_dir, "cluster_plot.png")
plt.savefig(cluster_plot_path)
plt.close()

print(f"Saved cluster plot to: {cluster_plot_path}")

# True label distribution in PCA space
plt.figure()
for label_id in np.unique(y_train):
    idx = y_train == label_id
    plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=le.classes_[label_id])

plt.title("True Label Distribution (PCA)")
plt.legend()

gt_plot_path = os.path.join(plots_dir, "true_label_distribution.png")
plt.savefig(gt_plot_path)
plt.close()

print(f"Saved True Label Distribution plot to: {gt_plot_path}")