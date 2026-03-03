import os
import csv
from collections import Counter
from typing import Set

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CLEAN = os.path.join(
    os.path.dirname(__file__), "clean_data", "clean_comment.csv"
)
DEFAULT_WORDLIST = os.path.join(
    BASE_DIR, "word_list", "Viet74K.txt"
)

# Load danh sách từ
def load_vocab(path: str) -> Set[str]:
    """Load word list file and return set of lowercased tokens."""
    vocab = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            w = line.strip().lower()
            if w:
                vocab.add(w)
    return vocab

# Phát hiện từ lóng và từ không có trong từ điển
def detect_slangs(clean_path: str = None, wordlist_path: str = None) -> Counter:
    """Scan cleaned comments and return counter of terms not in vocabulary.

    Terms are split on whitespace; punctuation should already be removed by
    preprocessing. Returned keys are lowercased.
    """
    if clean_path is None:
        clean_path = DEFAULT_CLEAN
    if wordlist_path is None:
        wordlist_path = DEFAULT_WORDLIST

    if not os.path.exists(clean_path):
        raise FileNotFoundError(f"clean data not found: {clean_path}")
    if not os.path.exists(wordlist_path):
        raise FileNotFoundError(f"word list not found: {wordlist_path}")

    vocab = load_vocab(wordlist_path)
    unknowns = Counter()

    with open(clean_path, newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            comment = row.get("comment", "")
            for tok in comment.split():
                t = tok.lower()
                if t.isdigit():
                    continue
                if t not in vocab:
                    unknowns[t] += 1
    return unknowns

c = detect_slangs()
if not c:
    print("Không tìm thấy từ lóng hoặc từ không có trong từ điển.")

out_dir = os.path.join(os.path.dirname(__file__), "detect")
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "slangs.csv")
with open(out_file, "w", newline="", encoding="utf-8") as fout:
    writer = csv.writer(fout)
    writer.writerow(["term", "count"])
    for term, cnt in c.most_common():
        writer.writerow([term, cnt])
print(f"Đã lưu các từ vào {out_file}")
