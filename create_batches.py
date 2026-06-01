import pandas as pd

df = pd.read_csv("data/raw/train.csv")
n = len(df)
third = n // 3

df.iloc[:third].to_csv("data/batches/batch_1.csv", index=False)
df.iloc[third:2*third].to_csv("data/batches/batch_2.csv", index=False)
df.iloc[2*third:].to_csv("data/batches/batch_3.csv", index=False)