# -*- coding: utf-8 -*-
"""
EXPERIMENT: SVM Training on Full Dataset with GPU Acceleration
Train/Val/Test split: 7/2/1 (70% train, 20% val, 10% test)
Using cuML for GPU-accelerated SVM training
"""

import pandas as pd
import numpy as np
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import joblib
import time
import cuml
from cuml.svm import SVC as cuSVC
from feature_encoder import FeatureEncoder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, "results")
os.makedirs(results_dir, exist_ok=True)

# ===========================
# Setup
# ===========================
print("=" * 80)
print("GPU-ACCELERATED SVM: Full Dataset Training (7/2/1 split)")
print("=" * 80)

from sentiment_define import analyze_sentiment

clean_csv = os.path.join(script_dir, "clean_data", "clean_comment.csv")
df = pd.read_csv(clean_csv)

print(f"\nLoading {len(df)} comments from cleaned data...")

# Analyze sentiment for ALL data
sentiment_results = []
logging.info(f"Analyzing sentiment for {len(df)} comments...")
for idx, row in df.iterrows():
    result = analyze_sentiment(row['comment'], language=row['lang'])
    sentiment_results.append(result)
    if (idx + 1) % 5000 == 0:
        logging.info(f"  Processed {idx + 1}/{len(df)}")

df['sentiment_label'] = [r['label'] for r in sentiment_results]

texts = df["comment"].astype(str).tolist()
labels = df["sentiment_label"].tolist()

print(f"\nFull dataset: {len(df)} samples")
print(f"Distribution: {df['sentiment_label'].value_counts().to_dict()}")

# ===========================
# Feature Extraction
# ===========================
print("\n" + "=" * 80)
print("FEATURE EXTRACTION")
print("=" * 80)

logging.info("Starting feature extraction with TF-IDF and Doc2Vec...")
encoder = FeatureEncoder(
    tfidf_max_features=500,
    tfidf_ngram_range=(1, 2),
    doc2vec_size=100,
    doc2vec_window=5,
    doc2vec_epochs=40
)

X_combined, y_encoded = encoder.fit_transform(texts, labels=labels)
le = encoder.label_encoder
logging.info(f"Feature extraction completed. Total features: {X_combined.shape[1]}")

# ===========================
# Train/Val/Test Split (7/2/1)
# ===========================
print("\n" + "=" * 80)
print("DATA SPLITTING (7/2/1 ratio)")
print("=" * 80)

# First split: 70% train+val, 10% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X_combined, y_encoded, test_size=0.1, random_state=42, stratify=y_encoded
)

# Second split: 70% train, 20% val (from the 90% temp data)
# 70/90 ≈ 0.777... to get final 70% train
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=2/9, random_state=42, stratify=y_temp
)

print(f"Train set: {len(X_train)} samples ({len(X_train)/len(X_combined)*100:.1f}%)")
print(f"Val set:   {len(X_val)} samples ({len(X_val)/len(X_combined)*100:.1f}%)")
print(f"Test set:  {len(X_test)} samples ({len(X_test)/len(X_combined)*100:.1f}%)")

# Check class distribution in each split
print(f"\nClass distribution:")
for split_name, y_split in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
    unique, counts = np.unique(y_split, return_counts=True)
    print(f"  {split_name}: {dict(zip(le.classes_[unique], counts))}")

# ===========================
# Model: SVM on GPU
# ===========================
print("\n" + "=" * 80)
print("TRAINING SVM ON GPU (CUDA)")
print("=" * 80)

try:
    logging.info("Initializing GPU SVM (cuML)...")
    start_time = time.time()
    
    # Convert to GPU arrays (cuML expects GPU arrays)
    logging.info("Scaling features on GPU...")
    X_train_gpu = cuml.preprocessing.StandardScaler().fit_transform(X_train)
    X_val_gpu = cuml.preprocessing.StandardScaler().fit_transform(X_val)
    X_test_gpu = cuml.preprocessing.StandardScaler().fit_transform(X_test)
    
    # Train SVM on GPU
    logging.info("Training SVM on GPU...")
    svm_model = cuSVC(kernel='rbf', C=1.0, gamma='scale')
    svm_model.fit(X_train_gpu, y_train)
    
    training_time = time.time() - start_time
    logging.info(f"✓ GPU Training completed in {training_time:.2f} seconds")
    print(f"✓ GPU Training completed in {training_time:.2f} seconds")
    
    # Predictions on validation and test sets
    logging.info("Making predictions on validation set...")
    y_pred_val = svm_model.predict(X_val_gpu)
    
    logging.info("Making predictions on test set...")
    y_pred_test = svm_model.predict(X_test_gpu)
    
except Exception as e:
    logging.warning(f"GPU training failed: {e}")
    logging.info("Falling back to CPU SVM training...")
    print(f"⚠ GPU training failed: {e}")
    print("Falling back to CPU SVM training...")
    
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    
    start_time = time.time()
    
    # Scale features
    logging.info("Scaling features on CPU...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Train SVM on CPU
    logging.info("Training SVM on CPU...")
    svm_model = SVC(kernel='rbf', C=1.0, probability=True)
    svm_model.fit(X_train_scaled, y_train)
    
    training_time = time.time() - start_time
    logging.info(f"✓ CPU Training completed in {training_time:.2f} seconds")
    print(f"✓ CPU Training completed in {training_time:.2f} seconds")
    
    # Predictions
    logging.info("Making predictions...")
    y_pred_val = svm_model.predict(X_val_scaled)
    y_pred_test = svm_model.predict(X_test_scaled)

# ===========================
# Validation Metrics
# ===========================
print("\n" + "=" * 80)
print("VALIDATION SET METRICS")
print("=" * 80)

logging.info("Computing validation metrics...")
val_accuracy = accuracy_score(y_val, y_pred_val)
val_precision = precision_score(y_val, y_pred_val, average='macro')
val_recall = recall_score(y_val, y_pred_val, average='macro')
val_f1 = f1_score(y_val, y_pred_val, average='macro')

val_metrics = {
    'Accuracy': val_accuracy,
    'Precision': val_precision,
    'Recall': val_recall,
    'F1-Score': val_f1
}

print("\n>>> VALIDATION METRICS <<<")
for key, value in val_metrics.items():
    print(f"  {key}: {value:.4f}")

# ===========================
# Test Metrics
# ===========================
print("\n" + "=" * 80)
print("TEST SET METRICS")
print("=" * 80)

logging.info("Computing test metrics...")
test_accuracy = accuracy_score(y_test, y_pred_test)
test_precision = precision_score(y_test, y_pred_test, average='macro')
test_recall = recall_score(y_test, y_pred_test, average='macro')
test_f1 = f1_score(y_test, y_pred_test, average='macro')

test_metrics = {
    'Accuracy': test_accuracy,
    'Precision': test_precision,
    'Recall': test_recall,
    'F1-Score': test_f1
}

print("\n>>> TEST METRICS <<<")
for key, value in test_metrics.items():
    print(f"  {key}: {value:.4f}")

# ===========================
# Confusion Matrix (Test Set)
# ===========================
print("\n" + "=" * 80)
print("CONFUSION MATRIX - TEST SET")
print("=" * 80)

cm_test = confusion_matrix(y_test, y_pred_test)
print("\nPredicted →  Criticism  Neutral  Praise")
print("Actual ↓")
for i, true_label in enumerate(le.classes_):
    print(f"{true_label:>8}: {cm_test[i]}")

# ===========================
# Save Results
# ===========================
logging.info("Saving results...")
svm_path = os.path.join(results_dir, "svm_full_data_gpu_model.pkl")
joblib.dump(svm_model, svm_path)
logging.info(f"Model saved to: {svm_path}")
print(f"\n✓ Model saved to: {svm_path}")

# Save metrics to CSV
metrics_df = pd.DataFrame({
    'Set': ['Validation', 'Test'],
    'Accuracy': [val_metrics['Accuracy'], test_metrics['Accuracy']],
    'Precision': [val_metrics['Precision'], test_metrics['Precision']],
    'Recall': [val_metrics['Recall'], test_metrics['Recall']],
    'F1-Score': [val_metrics['F1-Score'], test_metrics['F1-Score']]
})

metrics_csv = os.path.join(results_dir, "svm_gpu_full_data_metrics.csv")
metrics_df.to_csv(metrics_csv, index=False)
logging.info(f"Metrics saved to: {metrics_csv}")
print(f"✓ Metrics saved to: {metrics_csv}")

# Save metrics to TXT file
metrics_txt_path = os.path.join(results_dir, "svm_gpu_full_data_results.txt")

with open(metrics_txt_path, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("GPU-ACCELERATED SVM - FULL DATASET TRAINING RESULTS\n")
    f.write("Train/Val/Test Split: 7/2/1\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("DATASET INFORMATION:\n")
    f.write(f"  Total samples: {len(df)}\n")
    f.write(f"  Training samples: {len(X_train)} (70%)\n")
    f.write(f"  Validation samples: {len(X_val)} (20%)\n")
    f.write(f"  Test samples: {len(X_test)} (10%)\n")
    f.write(f"  Features: {X_combined.shape[1]} (500 TF-IDF + 100 Doc2Vec)\n")
    f.write(f"  Training time: {training_time:.2f} seconds\n\n")
    
    f.write("=" * 80 + "\n")
    f.write("VALIDATION SET RESULTS\n")
    f.write("=" * 80 + "\n")
    f.write("KEY METRICS:\n")
    for key, value in val_metrics.items():
        f.write(f"  {key}: {value:.4f}\n")
    f.write("\n")
    
    f.write("=" * 80 + "\n")
    f.write("TEST SET RESULTS\n")
    f.write("=" * 80 + "\n")
    f.write("KEY METRICS:\n")
    for key, value in test_metrics.items():
        f.write(f"  {key}: {value:.4f}\n")
    f.write("\n")
    
    f.write("CONFUSION MATRIX (TEST SET):\n")
    f.write("            Predicted → Criticism  Neutral  Praise\n")
    f.write("Actual ↓\n")
    for i, true_label in enumerate(le.classes_):
        f.write(f"{true_label:>12}: {cm_test[i]}\n")
    f.write("\n")

logging.info(f"Results saved to: {metrics_txt_path}")
print(f"✓ Results saved to: {metrics_txt_path}")

# ===========================
# Final Summary
# ===========================
print("\n" + "=" * 80)
print("✓ EXPERIMENT COMPLETED")
print("=" * 80)

print(f"""
📊 FINAL RESULTS:
{'='*50}

VALIDATION SET:
  Accuracy:    {val_accuracy:.4f}
  Precision:   {val_metrics['Precision']:.4f}
  Recall:      {val_metrics['Recall']:.4f}
  F1-Score:    {val_metrics['F1-Score']:.4f}

TEST SET:
  Accuracy:    {test_accuracy:.4f}
  Precision:   {test_metrics['Precision']:.4f}
  Recall:      {test_metrics['Recall']:.4f}
  F1-Score:    {test_metrics['F1-Score']:.4f}

TRAINING INFO:
  Total time: {training_time:.2f} seconds
  Dataset size: {len(df)} samples
  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}

Output Files:
• Model: {svm_path}
• Metrics CSV: {metrics_csv}
• Results TXT: {metrics_txt_path}
""")

logging.info("All tasks completed successfully!")
