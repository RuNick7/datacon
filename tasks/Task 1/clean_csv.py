from __future__ import annotations

from pathlib import Path

import pandas as pd
from rdkit import Chem

# ───────────────────────── Параметры ────────────────────────────
INPUT_PATH = Path("input_file.csv")
OUTPUT_PATH = Path("ic50_clean.csv")

REQUIRED_COLS = {
    "Smiles",
    "Standard Value",
    "Standard Units",
    "Standard Relation",
}

NANOMOLAR_TAGS = {"nm", "nM"}

# ───────────────────────── Вспомогательные ──────────────────────

def valid_smiles(smiles: str | float | None) -> bool:
    """True, если RDKit успешно парсит строку SMILES."""
    if not isinstance(smiles, str) or not smiles:
        return False
    try:
        return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return False

# ───────────────────────── Основная логика ──────────────────────

def clean_ic50(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", quotechar='"', low_memory=False)

    # 1. Проверяем наличие обязательных столбцов
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Пропущена колонка: {', '.join(sorted(missing))}")

    df = df.loc[:, list(REQUIRED_COLS)].copy()

    # 2. Приводим IC50 к числу
    df["Standard Value"] = pd.to_numeric(df["Standard Value"], errors="coerce")
    df = df.dropna(subset=["Standard Value"])

    # 3. Фильтруем единицы (оставляем только нМ)
    df["Standard Units"] = df["Standard Units"].astype(str).str.replace(" ", "").str.lower()
    df = df[df["Standard Units"].isin({u.lower() for u in NANOMOLAR_TAGS})]

    # 4. Фильтр по точному равенству
    df["Standard Relation"] = (
        df["Standard Relation"].astype(str)
        .str.replace(r"[\"']", "", regex=True)
        .str.strip()
    )
    df = df[df["Standard Relation"] == "="]

    # 5. Удаляем дубликаты по SMILES
    df = df.drop_duplicates(subset="Smiles", keep="first")

    # 6. Валидируем SMILES
    df = df[df["Smiles"].map(valid_smiles)]

    # 7. Переименовываем числовую колонку
    df = df.rename(columns={"Standard Value": "Standard Value (nM)"})

    return df[["Smiles", "Standard Value (nM)"]]


if __name__ == "__main__":
    cleaned = clean_ic50(INPUT_PATH)
    cleaned.to_csv(OUTPUT_PATH, index=False)
    print(f"Сохранён файл: {OUTPUT_PATH} ({len(cleaned)} строк)")
