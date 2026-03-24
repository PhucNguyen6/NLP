# -*- coding: utf-8 -*-
"""
Feature Encoding Module
Handles TF-IDF, Doc2Vec encoding and label encoding for sentiment classification
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


class FeatureEncoder:
    """
    Encodes text data into combined TF-IDF + Doc2Vec features
    """
    
    def __init__(self, tfidf_max_features=500, tfidf_ngram_range=(1, 2),
                 doc2vec_size=100, doc2vec_window=5, doc2vec_epochs=40):
        """
        Initialize feature encoder
        
        Args:
            tfidf_max_features: Maximum number of TF-IDF features
            tfidf_ngram_range: N-gram range for TF-IDF
            doc2vec_size: Vector size for Doc2Vec
            doc2vec_window: Window size for Doc2Vec
            doc2vec_epochs: Training epochs for Doc2Vec
        """
        self.tfidf_max_features = tfidf_max_features
        self.tfidf_ngram_range = tfidf_ngram_range
        self.doc2vec_size = doc2vec_size
        self.doc2vec_window = doc2vec_window
        self.doc2vec_epochs = doc2vec_epochs
        
        self.tfidf = None
        self.doc2vec_model = None
        self.label_encoder = None
    
    def encode_labels(self, labels):
        """
        Encode class labels to numeric values
        
        Args:
            labels: List of class labels
            
        Returns:
            Encoded labels (numpy array), label encoder, classes
        """
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(labels)
        
        print(f"Classes: {self.label_encoder.classes_}")
        print(f"Label distribution: {np.unique(y_encoded, return_counts=True)}")
        
        return y_encoded
    
    def fit_tfidf(self, texts):
        """
        Fit TF-IDF vectorizer on texts
        
        Args:
            texts: List of text documents
            
        Returns:
            TF-IDF feature matrix (numpy array)
        """
        print("  Computing TF-IDF (500 features, 1-2 grams)...")
        self.tfidf = TfidfVectorizer(
            max_features=self.tfidf_max_features,
            ngram_range=self.tfidf_ngram_range
        )
        X_tfidf = self.tfidf.fit_transform(texts).toarray()
        print(f"    TF-IDF shape: {X_tfidf.shape}")
        
        return X_tfidf
    
    def transform_tfidf(self, texts):
        """
        Transform texts using fitted TF-IDF vectorizer
        
        Args:
            texts: List of text documents
            
        Returns:
            TF-IDF feature matrix (numpy array)
        """
        if self.tfidf is None:
            raise ValueError("TF-IDF not fitted yet. Call fit_tfidf first.")
        
        return self.tfidf.transform(texts).toarray()
    
    def fit_doc2vec(self, texts):
        """
        Train Doc2Vec model on texts
        
        Args:
            texts: List of text documents
            
        Returns:
            Doc2Vec features (numpy array)
        """
        print("  Training Doc2Vec (100D vectors, 40 epochs)...")
        
        # Create tagged documents
        tagged_data = [
            TaggedDocument(words=text.split(), tags=[str(i)])
            for i, text in enumerate(texts)
        ]
        
        # Train Doc2Vec
        self.doc2vec_model = Doc2Vec(
            vector_size=self.doc2vec_size,
            window=self.doc2vec_window,
            min_count=1,
            workers=4,
            epochs=self.doc2vec_epochs
        )
        self.doc2vec_model.build_vocab(tagged_data)
        self.doc2vec_model.train(
            tagged_data,
            total_examples=self.doc2vec_model.corpus_count,
            epochs=self.doc2vec_model.epochs
        )
        
        # Infer vectors
        X_doc2vec = np.array([
            self.doc2vec_model.infer_vector(text.split())
            for text in texts
        ])
        print(f"    Doc2Vec shape: {X_doc2vec.shape}")
        
        return X_doc2vec
    
    def transform_doc2vec(self, texts):
        """
        Transform texts using fitted Doc2Vec model
        
        Args:
            texts: List of text documents
            
        Returns:
            Doc2Vec features (numpy array)
        """
        if self.doc2vec_model is None:
            raise ValueError("Doc2Vec not fitted yet. Call fit_doc2vec first.")
        
        return np.array([
            self.doc2vec_model.infer_vector(text.split())
            for text in texts
        ])
    
    def fit_transform(self, texts, labels=None):
        """
        Fit both TF-IDF and Doc2Vec on texts, optionally encode labels
        
        Args:
            texts: List of text documents
            labels: (Optional) List of class labels
            
        Returns:
            Combined features (numpy array), encoded labels (if provided)
        """
        # Encode labels if provided
        y_encoded = None
        if labels is not None:
            y_encoded = self.encode_labels(labels)
        
        # Fit and transform features
        X_tfidf = self.fit_tfidf(texts)
        X_doc2vec = self.fit_doc2vec(texts)
        
        # Combine features
        X_combined = np.hstack((X_tfidf, X_doc2vec))
        print(f"  Combined features shape: {X_combined.shape}")
        
        return (X_combined, y_encoded) if y_encoded is not None else X_combined
    
    def transform(self, texts):
        """
        Transform texts using fitted encoders
        
        Args:
            texts: List of text documents
            
        Returns:
            Combined features (numpy array)
        """
        X_tfidf = self.transform_tfidf(texts)
        X_doc2vec = self.transform_doc2vec(texts)
        
        return np.hstack((X_tfidf, X_doc2vec))
