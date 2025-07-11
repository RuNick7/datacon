import pandas as pd
import numpy as np

SRC_CSV = "data_desc_and_fp.csv"   # исходный файл
THRESH = 0.70

df = pd.read_csv(SRC_CSV)

df = df.dropna(axis=0, how="any")
print(len(df.columns))

cols_constant = [col for col in df.columns if df[col].nunique(dropna=False) == 1]
df = df.drop(columns=cols_constant)
print(len(df.columns))

num_cols = df.select_dtypes(include=[np.number]).columns
print(f"Числовых признаков: {len(num_cols)}")

corr = df[num_cols].corr().abs()                 # |r|
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
# upper — верхний треугольник без диагонали (k=1)

to_drop = [col for col in upper.columns if any(upper[col] > THRESH)]
print(f"Удаляем {len(to_drop)} скоррелированных столбцов (r > {THRESH}):")
for c in to_drop:
    print(" •", c)

df_reduced = df.drop(columns=to_drop)
print(len(df_reduced.columns))