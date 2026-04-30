import pandas as pd
import os

# Vietnamese dictionary 36K
df_vi = pd.read_csv("hf://datasets/tsdocode/vietnamese-dictionary/vi_dictionary.csv")


os.makedirs("raw_dict", exist_ok=True)
df_vi.to_csv("raw_dict/vi_dict.csv", index=False)

