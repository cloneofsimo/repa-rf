# /home/ubuntu/pd12m.int8/dataset/pd12m/metadata/pd12m.001.parquet from 000 to 124
import glob

import pandas as pd

parquet_files = glob.glob("../dataset/pd12m/metadata/pd12m.*.parquet")

df = pd.concat([pd.read_parquet(file) for file in parquet_files])
print(df.head())

df.to_csv("../dataset/pd12m.csv", index=False)
