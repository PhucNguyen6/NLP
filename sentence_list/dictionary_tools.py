import os
import pandas as pd


RAW_DICT_DIR = "raw_dict"
CLEAN_DICT_DIR = "clean_dict"


def ensure_dirs():
    os.makedirs(RAW_DICT_DIR, exist_ok=True)
    os.makedirs(CLEAN_DICT_DIR, exist_ok=True)


def crawl_vi_dict(output_path: str = f"{RAW_DICT_DIR}/vi_dict.csv"):
    ensure_dirs()
    df = pd.read_csv("hf://datasets/tsdocode/vietnamese-dictionary/vi_dictionary.csv")
    df.to_csv(output_path, index=False)
    return output_path


def crawl_en_dict(output_path: str = f"{RAW_DICT_DIR}/en_dict.csv"):
    ensure_dirs()
    df = pd.read_csv("hf://datasets/mrrtmob/english-khmer-dictionary/dictionary.csv")
    keep_cols = [c for c in ["word_en", "definition_en", "part_of_speech"] if c in df.columns]
    if keep_cols:
        df = df[keep_cols]
        df = df.rename(columns={"word_en": "word", "part_of_speech": "pos"})
    else:
        drop_cols = [c for c in ["word_km", "definition_km", "example_en", "example_km"] if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)
    df.to_csv(output_path, index=False)
    return output_path


def clean_vi_dict(input_path: str = f"{RAW_DICT_DIR}/vi_dict.csv", output_path: str = f"{CLEAN_DICT_DIR}/vi_dict.csv"):
    ensure_dirs()
    df = pd.read_csv(input_path)
    rename_map = {"meaning": "definition"}
    for src, dst in rename_map.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})
    required = [c for c in ["word", "definition"] if c in df.columns]
    df = df.dropna(subset=required)
    for col in required:
        df[col] = df[col].astype(str).str.strip()
    df = df[(df["word"] != "") & (df["definition"] != "")]
    df = df.drop_duplicates(subset=["word", "definition"], keep="first")
    df.to_csv(output_path, index=False)
    return output_path


def clean_en_dict(input_path: str = f"{RAW_DICT_DIR}/en_dict.csv", output_path: str = f"{CLEAN_DICT_DIR}/en_dict.csv"):
    ensure_dirs()
    df = pd.read_csv(input_path)
    if "definition_en" in df.columns and "definition" not in df.columns:
        df = df.rename(columns={"definition_en": "definition"})
    if "word_en" in df.columns and "word" not in df.columns:
        df = df.rename(columns={"word_en": "word"})
    required = [c for c in ["word", "definition"] if c in df.columns]
    df = df.dropna(subset=required)
    for col in required:
        df[col] = df[col].astype(str).str.strip()
    df = df[(df["word"] != "") & (df["definition"] != "")]
    df = df.drop_duplicates(subset=["word", "definition"], keep="first")
    df.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    vi_raw = crawl_vi_dict()
    en_raw = crawl_en_dict()
    vi_clean = clean_vi_dict(vi_raw)
    en_clean = clean_en_dict(en_raw)
    print(f"Saved cleaned dictionaries: {vi_clean}, {en_clean}")
