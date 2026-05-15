# -*- coding: utf-8 -*-
"""
Pipeline theo kiến trúc hệ thống:
Preprocessing -> Sentiment Module + RAG Pipeline -> LLM Service -> Output.
"""

import json
import logging
import pickle
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import joblib

from config import EMBEDDING_CONFIG, LOGGING_CONFIG, MODELS_DIR
from embeddings_manager import get_embeddings_manager
from llm_client import get_llm_client
from preprocessing_comment import preprocess_comment
from rag_retriever import get_rag_retriever
from dictionary_expander import get_dictionary_expander

logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["level"]),
    format=LOGGING_CONFIG["format"],
)
logger = logging.getLogger(__name__)

SVM_MODEL_FILES = {
    "tfidf": "svm_model_tf_idf.pkl",
    "doc2vec": "svm_model_doc2vec.pkl",
    "xlmroberta": "svm_model_xlm_roberta.pkl",
}

VI_LABEL_MAP = {
    "praise": "khen",
    "positive": "khen",
    "neutral": "trung lap",
    "criticism": "che",
    "negative": "che",
    "khen": "khen",
    "che": "che",
    "trung lap": "trung lap",
}


def to_vi_label(label: Optional[str]) -> Optional[str]:
    if label is None:
        return None
    return VI_LABEL_MAP.get(str(label).strip().lower(), str(label))


class PreprocessingLayer:
    def run(self, text: str) -> Dict:
        cleaned, detected_lang = preprocess_comment(text)
        return {
            "raw_text": text,
            "cleaned_text": cleaned if cleaned else text.strip(),
            "language": detected_lang or "unknown",
        }


class SentimentModule:
    def __init__(self):
        self.models = self._load_models()

    def _load_models(self) -> Dict[str, Dict]:
        loaded = {}
        for key, filename in SVM_MODEL_FILES.items():
            model_path = Path(MODELS_DIR) / filename
            if not model_path.exists():
                logger.warning("Missing SVM model file: %s", model_path)
                continue
            try:
                package = joblib.load(model_path)
            except Exception:
                # Keep backward compatibility with plain pickle artifacts.
                with open(model_path, "rb") as f:
                    package = pickle.load(f)
            if isinstance(package, dict) and "model" in package:
                loaded[key] = package
            else:
                loaded[key] = {"model": package, "scaler": None, "label_encoder": None}
        return loaded

    def predict_all(self, embedding_map: Dict[str, np.ndarray]) -> Dict[str, Dict]:
        results = {}
        for model_name, package in self.models.items():
            vector = embedding_map.get(model_name)
            if vector is None:
                continue
            model = package.get("model")
            scaler = package.get("scaler")
            label_encoder = package.get("label_encoder")
            try:
                features = vector.reshape(1, -1)
                if scaler is not None:
                    features = scaler.transform(features)
                pred_raw = model.predict(features)[0]
                label = (
                    label_encoder.inverse_transform([pred_raw])[0]
                    if label_encoder is not None
                    else str(pred_raw)
                )
                label = to_vi_label(label)
                confidence = None
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(features)[0]
                    confidence = float(np.max(proba))
                results[model_name] = {"label": label, "confidence": confidence}
            except Exception as ex:
                logger.warning("Predict failed for %s: %s", model_name, ex)
        return results


PRIMARY_ENCODERS = ("tfidf", "doc2vec", "xlmroberta")


class RAGPipeline:
    def __init__(self):
        self.retriever = get_rag_retriever()

    def run(self, query: str, top_k: int = 5, model: str = "xlmroberta") -> Dict:
        m = model if model in PRIMARY_ENCODERS else "xlmroberta"
        docs = self.retriever.retrieve_context(query=query, top_k=top_k, model=m)
        distribution = Counter([to_vi_label(d.get("sentiment_label", "neutral")) for d in docs])
        return {
            "documents": docs,
            "distribution": dict(distribution),
            "rag_majority_label": distribution.most_common(1)[0][0] if distribution else "trung lap",
        }


class LLMService:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.client = get_llm_client() if enabled else None

    def explain(self, comment: str, sentiment_label: str, rag_docs: List[Dict]) -> Optional[str]:
        if not self.enabled or self.client is None:
            return None
        context = "\n".join(
            f"- [{to_vi_label(doc.get('sentiment_label', 'unknown'))}] {doc.get('text_content', '')[:160]}"
            for doc in rag_docs[:3]
        )
        return self.client.generate_sentiment_explanation(
            query=comment,
            context=context,
            sentiment_label=sentiment_label,
        )


class SentimentArchitecturePipeline:
    def __init__(
        self,
        use_llm: bool = True,
        auto_expand_dict: bool = True,
        primary_encoder: str = "xlmroberta",
    ):
        self.preprocessing = PreprocessingLayer()
        self.embedding = get_embeddings_manager()
        self.sentiment = SentimentModule()
        self.rag = RAGPipeline()
        self.llm = LLMService(enabled=use_llm)
        self.dict_expander = get_dictionary_expander() if auto_expand_dict else None
        self.primary_encoder = primary_encoder if primary_encoder in PRIMARY_ENCODERS else "xlmroberta"

    def _build_embedding_map(self, cleaned_text: str) -> Dict[str, np.ndarray]:
        emb = self.embedding.embed_sentence(cleaned_text, models=EMBEDDING_CONFIG["models"])
        return {k: np.asarray(v, dtype=np.float32) for k, v in emb.items()}

    def analyze_comment(
        self,
        comment: str,
        return_explanation: bool = True,
        top_k: int = 5,
        language: str = "auto",
        primary_encoder: Optional[str] = None,
    ) -> Dict:
        started = time.perf_counter()
        preprocessed = self.preprocessing.run(comment)
        enc = (primary_encoder or self.primary_encoder).lower()
        if enc not in PRIMARY_ENCODERS:
            enc = "xlmroberta"

        embedding_map = self._build_embedding_map(preprocessed["cleaned_text"])
        sentiment_by_encoder = self.sentiment.predict_all(embedding_map)
        rag_result = self.rag.run(preprocessed["cleaned_text"], top_k=top_k, model=enc)

        # Dictionary expansion happens before generating the final LLM explanation.
        if self.dict_expander is not None:
            requested_lang = language if language != "auto" else preprocessed["language"]
            try:
                self.dict_expander.expand_from_text(comment, language=requested_lang or "vi")
            except Exception as ex:
                # Dictionary expansion should never break inference.
                logger.warning("Dictionary expansion failed (ignored): %s", ex)

        final_label = rag_result["rag_majority_label"]
        if enc in sentiment_by_encoder:
            final_label = sentiment_by_encoder[enc]["label"]
        elif "xlmroberta" in sentiment_by_encoder:
            final_label = sentiment_by_encoder["xlmroberta"]["label"]

        explanation = None
        if return_explanation:
            explanation = self.llm.explain(comment, final_label, rag_result["documents"])

        return {
            "timestamp": datetime.now().isoformat(),
            "input": comment,
            "preprocessing": preprocessed,
            "sentiment_module": {
                "predictions": sentiment_by_encoder,
                "final_label": final_label,
            },
            "rag_pipeline": {
                "distribution": rag_result["distribution"],
                "top_k": top_k,
                "documents": [
                    {
                        "text": d.get("text_content", "")[:180],
                        "sentiment": to_vi_label(d.get("sentiment_label", "unknown")),
                        "similarity": float(d.get("similarity", 0.0)),
                    }
                    for d in rag_result["documents"]
                ],
            },
            "llm_explanation": explanation,
            "metadata": {
                "requested_language": language,
                "primary_encoder": enc,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        }

    def compare_encoders(self, comments: List[str]) -> Dict:
        output = []
        totals = {name: 0.0 for name in EMBEDDING_CONFIG["models"]}
        for text in comments:
            pre = self.preprocessing.run(text)
            run_item = {"input": text, "cleaned": pre["cleaned_text"], "predictions": {}}
            for encoder in EMBEDDING_CONFIG["models"]:
                started = time.perf_counter()
                vector = self.embedding.embed_sentence(pre["cleaned_text"], models=[encoder]).get(encoder)
                elapsed_ms = (time.perf_counter() - started) * 1000
                totals[encoder] += elapsed_ms
                if vector is None:
                    run_item["predictions"][encoder] = {"label": None, "encode_ms": round(elapsed_ms, 2)}
                    continue
                pred = self.sentiment.predict_all({encoder: np.asarray(vector, dtype=np.float32)}).get(
                    encoder, {}
                )
                pred["encode_ms"] = round(elapsed_ms, 2)
                run_item["predictions"][encoder] = pred
            output.append(run_item)
        n = len(comments) or 1
        return {
            "samples": output,
            "avg_encode_ms": {k: round(v / n, 2) for k, v in totals.items()},
        }

    def batch_analyze(
        self,
        comments: List[str],
        return_explanations: bool = True,
        top_k: int = 5,
        primary_encoder: Optional[str] = None,
    ) -> List[Dict]:
        return [
            self.analyze_comment(
                c,
                return_explanation=return_explanations,
                top_k=top_k,
                primary_encoder=primary_encoder,
            )
            for c in comments
        ]

    def export_results(self, results: List[Dict], output_format: str = "json") -> str:
        if output_format == "json":
            return json.dumps(results, ensure_ascii=False, indent=2)
        raise ValueError("Only json is currently supported in compact pipeline")

    def interactive_session(self):
        print("Interactive mode (type 'exit' to stop)")
        while True:
            text = input(">>> ").strip()
            if text.lower() == "exit":
                break
            res = self.analyze_comment(text, return_explanation=self.llm.enabled)
            print(json.dumps(res["sentiment_module"], ensure_ascii=False, indent=2))

    def get_statistics(self) -> Dict:
        return {
            "available_encoders": EMBEDDING_CONFIG["models"],
            "loaded_svm_models": list(self.sentiment.models.keys()),
            "llm_enabled": self.llm.enabled,
            "primary_encoder": self.primary_encoder,
        }


_pipeline_store: Dict[tuple, SentimentArchitecturePipeline] = {}


def get_pipeline(
    use_llm: bool = True,
    auto_expand_dict: bool = True,
    primary_encoder: str = "xlmroberta",
) -> SentimentArchitecturePipeline:
    pe = primary_encoder if primary_encoder in PRIMARY_ENCODERS else "xlmroberta"
    key = (use_llm, auto_expand_dict, pe)
    if key not in _pipeline_store:
        _pipeline_store[key] = SentimentArchitecturePipeline(
            use_llm=use_llm,
            auto_expand_dict=auto_expand_dict,
            primary_encoder=pe,
        )
    return _pipeline_store[key]
