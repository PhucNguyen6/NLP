# -*- coding: utf-8 -*-
"""
Điểm vào duy nhất cho dự án NLP.

  python -m sentence_list.cli --help
  python -m sentence_list.cli data crawl
  python -m sentence_list.cli data clean
  python -m sentence_list.cli dict build
  python -m sentence_list.cli train all --device cuda --skip-learning-curve
  python -m sentence_list.cli db setup
  python -m sentence_list.cli db fill
  python -m sentence_list.cli db eval
  python -m sentence_list.cli predict "Bình luận hay quá!"
  python -m sentence_list.cli ui
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
# Cho phép import phẳng (config, experiments, …) khi chạy: python -m sentence_list.cli
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))


def _train_args(p):
    p.add_argument(
        "sub",
        nargs="?",
        choices=["encode", "train", "all", "gpu-check"],
        default="all",
        help="all | encode | train | gpu-check",
    )
    p.add_argument("--method", choices=["tfidf", "doc2vec", "xlmroberta", "all"], default="all")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default=None)
    p.add_argument("--label-mode", choices=["auto", "teacher", "lexicon", "percentile", "hybrid", "cluster"], default=None)
    p.add_argument("--skip-learning-curve", action="store_true")
    p.add_argument("--learning-curve-jobs", type=int, default=None)


def cmd_data(args):
    from data import run_crawl, run_preprocess

    if args.action == "crawl":
        run_crawl()
    elif args.action == "clean":
        run_preprocess()
    else:
        raise SystemExit(f"Lệnh data không hợp lệ: {args.action}")


def cmd_dict(args):
    from dictionary import build_dictionary, export_dictionary_chunks

    if args.action == "build":
        if args.skip_crawl:
            info = export_dictionary_chunks()
        else:
            info = build_dictionary(export_chunks=True)
        n = info.get("total_chunks")
        if n is None and isinstance(info.get("metadata"), dict):
            n = info["metadata"].get("total_chunks")
        print(f"✓ Chunk: {n if n is not None else info}")
    else:
        raise SystemExit(f"Lệnh dict không hợp lệ: {args.action}")


def cmd_train(args):
    from config import torch_cuda_diagnostics
    from experiments import run_encode, run_train

    sub = args.sub or "all"
    if sub == "gpu-check":
        info = torch_cuda_diagnostics()
        for k, v in info.items():
            print(f"{k}: {v}")
        return
    if sub in ("encode", "all"):
        method = "all" if sub == "all" else args.method
        run_encode(method, device=args.device, label_mode=args.label_mode)
    if sub in ("train", "all"):
        run_train(
            skip_learning_curve=args.skip_learning_curve,
            learning_curve_jobs=args.learning_curve_jobs,
        )


def cmd_db(args):
    from db_ops import run_populate_dictionary, run_populate_embeddings, run_rag_eval, run_setup

    if args.action == "setup":
        run_setup(skip_confirm=args.yes)
    elif args.action == "fill":
        run_populate_embeddings(force=args.force)
    elif args.action == "fill-dict":
        run_populate_dictionary()
    elif args.action == "eval":
        run_rag_eval(
            max_queries=args.max_queries,
            top_k=args.top_k,
            threshold=args.threshold,
        )
    else:
        raise SystemExit(f"Lệnh db không hợp lệ: {args.action}")


def cmd_predict(args):
    from inference_pipeline import get_pipeline
    from utils import create_sentiment_report, print_summary_statistics

    pipeline = get_pipeline(use_llm=not args.no_llm, primary_encoder=args.encoder)
    if args.interactive:
        pipeline.interactive_session()
        return
    if args.batch:
        with open(args.batch, encoding="utf-8") as f:
            comments = [ln.strip() for ln in f if ln.strip()]
        results = pipeline.batch_analyze(comments, return_explanations=not args.no_llm, top_k=args.top_k)
        out = Path(args.output)
        out.write_text(pipeline.export_results(results), encoding="utf-8")
        print(f"Đã lưu {out}")
        print_summary_statistics(results)
        return
    if args.text:
        r = pipeline.analyze_comment(args.text, return_explanation=not args.no_llm, top_k=args.top_k)
        print(create_sentiment_report(r))
        return
    raise SystemExit('Cần câu bình luận, --batch, hoặc --interactive')


def cmd_ui(_args):
    app = BASE / "streamlit_app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app)], cwd=str(BASE.parent))


def main(argv=None):
    from config import setup_project_logging

    setup_project_logging()
    parser = argparse.ArgumentParser(
        prog="nlp",
        description="NLP Sentiment + RAG — một lệnh cho toàn bộ pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_data = sub.add_parser("data", help="Crawl & tiền xử lý bình luận")
    p_data.add_argument("action", choices=["crawl", "clean"])
    p_data.set_defaults(func=cmd_data)

    p_dict = sub.add_parser("dict", help="Từ điển CSV → JSON chunk")
    p_dict.add_argument("action", choices=["build"], default="build", nargs="?")
    p_dict.add_argument("--skip-crawl", action="store_true", help="Chỉ chunk từ CSV có sẵn")
    p_dict.set_defaults(func=cmd_dict)

    p_train = sub.add_parser("train", help="Encode + train SVM (experiments.py)")
    _train_args(p_train)
    p_train.set_defaults(func=cmd_train)

    p_db = sub.add_parser("db", help="PostgreSQL & đánh giá RAG")
    p_db.add_argument("action", choices=["setup", "fill", "fill-dict", "eval"])
    p_db.add_argument("-y", "--yes", action="store_true", help="setup không hỏi xác nhận")
    p_db.add_argument("--force", action="store_true", help="nạp lại embedding")
    p_db.add_argument("--max-queries", type=int, default=150)
    p_db.add_argument("--top-k", type=int, default=None)
    p_db.add_argument("--threshold", type=float, default=None)
    p_db.set_defaults(func=cmd_db)

    p_pred = sub.add_parser("predict", help="Phân tích bình luận (CLI)")
    p_pred.add_argument("text", nargs="?", help="Một câu bình luận")
    p_pred.add_argument("--batch", "-b", type=str)
    p_pred.add_argument("--output", "-o", default="results.json")
    p_pred.add_argument("--interactive", "-i", action="store_true")
    p_pred.add_argument("--top-k", type=int, default=5)
    p_pred.add_argument("--no-llm", action="store_true")
    p_pred.add_argument("--encoder", default="xlmroberta", choices=["xlmroberta", "doc2vec", "tfidf"])
    p_pred.set_defaults(func=cmd_predict)

    p_ui = sub.add_parser("ui", help="Streamlit chat")
    p_ui.set_defaults(func=cmd_ui)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
