from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, MACCSkeys
from rdkit.Chem import rdFingerprintGenerator as rfp

INPUT_CSV  = Path("ic50_clean.csv")          # очищенный датасет
CLEAN_OUT  = Path("ic50_2d_desc_fps_clean.csv") # после фильтрации
CORR_THRES = 0.70       # |r| порог корреляции
MAKE_FP    = True        # True → рассчитываем Morgan + MACCS

RDLogger.DisableLog("rdApp.*")  # скрываем варнинги RDKit
warnings.filterwarnings("ignore", category=RuntimeWarning)  # NaN при корр.

"""2-D дескрипторы"""

FAMILIES = (
    "BCUT2D", "PEOE_VSA", "SMR_VSA", "SlogP_VSA",
    "EState_VSA", "VSA_EState", "fr_",
)

CORE = [
    "MolWt", "ExactMolWt", "HeavyAtomCount", "HeavyAtomMolWt", "NumValenceElectrons",
    "NumHeteroatoms", "NHOHCount", "NOCount", "NumAmideBonds", "RingCount",
    "NumSpiroAtoms", "NumBridgeheadAtoms", "NumAliphaticRings", "NumAromaticRings",
    "NumSaturatedRings", "NumAliphaticHeterocycles", "NumAliphaticCarbocycles",
    "NumAromaticHeterocycles", "NumAromaticCarbocycles", "NumSaturatedHeterocycles",
    "NumSaturatedCarbocycles", "MolLogP", "MolMR", "TPSA", "LabuteASA",
    "FractionCSP3", "NumRotatableBonds", "NumHAcceptors", "NumHDonors",
    "MaxPartialCharge", "MinPartialCharge", "MaxAbsPartialCharge",
    "MinAbsPartialCharge", "BertzCT", "BalabanJ", "Ipc", "HallKierAlpha",
    "Chi0", "Chi1",
] + [f"Chi{i}n" for i in range(5)] + [f"Chi{i}v" for i in range(5)] + [f"Kappa{i}" for i in (1, 2, 3)]

EXTRA = [n for n, _ in Descriptors._descList if n.startswith(FAMILIES)]
EXTRA += ["QED", "FpDensityMorgan1", "FpDensityMorgan2", "FpDensityMorgan3"]

DESC_NAMES = [n for n in dict.fromkeys(CORE + EXTRA) if hasattr(Descriptors, n)]
DESC_FUNCS  = {n: getattr(Descriptors, n) for n in DESC_NAMES}
print(f"→ рассчитываем {len(DESC_NAMES)} 2‑D дескрипторов")

"""Генераторы отпечатков"""

FP_COLS: list[str] = []
if MAKE_FP:
    morgan_gen = rfp.GetMorganGenerator(radius=2, fpSize=2048)
    FP_COLS = ["FP_Morgan2048", "FP_MACCS166"]

"""Исходный CSV"""

df_src = pd.read_csv(INPUT_CSV)

"""Вычисление признаков для одной молекулы"""

def calc_row(smiles: str, ic50: float):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  # пропуск
    desc = [fn(mol) for fn in DESC_FUNCS.values()]
    fps  = []
    if MAKE_FP:
        fps = [
            morgan_gen.GetFingerprint(mol).ToBitString(),
            MACCSkeys.GenMACCSKeys(mol).ToBitString(),
        ]
    return [smiles, ic50] + desc + fps

"""Применяем ко всем записям"""

rows = []
for smi, val in zip(df_src["Smiles"], df_src["Standard Value (nM)"]):
    out = calc_row(smi, val)
    if out:
        rows.append(out)

columns = ["Smiles", "IC50_nM"] + DESC_NAMES + FP_COLS

df_full = pd.DataFrame(rows, columns=columns)

"""Очистка: NaN → zero-var → corr-filter"""

meta_cols = ["Smiles", "IC50_nM"]
cat_cols  = FP_COLS  # бинарные отпечатки не трогаем

desc_cols = [c for c in df_full.columns if c not in meta_cols + cat_cols]
num_df = df_full[desc_cols].dropna(axis=1, how="any")        # убираем NaN‑столбцы
num_df = num_df.loc[:, num_df.var() > 0.0]                   # нулевая дисперсия

corr = num_df.corr().abs()
upper = corr.where(np.triu(np.ones_like(corr), k=1).astype(bool))

drop_corr = [c for c in upper.columns if (upper[c] > CORR_THRES).any()]
num_df = num_df.drop(columns=drop_corr)
print(f"  дескрипторы: {len(desc_cols)} → {num_df.shape[1]} после очистки")

"""Сохранение датасета"""

df_clean = pd.concat([df_full[meta_cols + cat_cols], num_df], axis=1)
df_clean.to_csv(CLEAN_OUT, index=False)
print(f"Cохранено: {CLEAN_OUT}  ({df_clean.shape[1]-len(meta_cols)-len(cat_cols)} признаков)")