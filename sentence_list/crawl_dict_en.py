import pandas as pd
import os

# English dictionary 176K
df_en = pd.read_csv("hf://datasets/mrrtmob/english-khmer-dictionary/dictionary.csv")
df_en.drop(columns=["word_km"], inplace=True)
df_en.drop(columns=["definition_km"], inplace=True)
df_en.drop(columns=["example_en"], inplace=True)
df_en.drop(columns=["example_km"], inplace=True)


os.makedirs("raw_dict", exist_ok=True)
df_en.to_csv("raw_dict/en_dict.csv", index=False)
