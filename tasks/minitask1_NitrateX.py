from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict

import pandas as pd
from rdkit import Chem

# ───────────────────────── CONFIG
DEFAULT_INPUT = Path("input_file.csv")
DEFAULT_OUTPUT = Path("ic50_clean.csv")

RAW_COLUMNS = [
    "Molecule ChEMBL ID",
    "Smiles",
    "Standard Type",
    "Standard Relation",
    "Standard Value",
    "Standard Units",
]

UNIT_FACTOR: Dict[str, float] = {
    "nm": 1.0,
    "nmol": 1.0,
    "nm/l": 1.0,
    "nmol/l": 1.0,
    "nmol·l-1": 1.0,
    "nmol/litre": 1.0,
    "nmolar": 1.0,
    "nm": 1.0,
    "nm": 1.0,

    "nm": 1.0,  # duplicate to cover variations
    "nM": 1.0,

    "µm": 1e3,
    "μm": 1e3,
    "um": 1e3,
    "uM": 1e3,
    "microm": 1e3,

    "pm": 1e-3,
    "pM": 1e-3,

    "mm": 1e6,
    "mM": 1e6,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("clean_ic50")

def _unit_factor(unit: str | float | None) -> float | None:
    if not isinstance(unit, str):
        return None
    return UNIT_FACTOR.get(unit.strip().replace(" ", "").lower())


def _valid_smiles(smi) -> bool:
    if not isinstance(smi, str) or not smi:
        return False
    try:
        return Chem.MolFromSmiles(smi) is not None
    except Exception:
        return False

def clean_file(src: Path, dst: Path) -> None:
    logger.info("Reading %s", src)
    df = pd.read_csv(src, sep=";", quotechar="\"", low_memory=False)

    # ensure required columns exist
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        logger.error("Missing columns in input: %s", ", ".join(missing))
        sys.exit(1)

    df = df[RAW_COLUMNS].copy()

    df["Standard Value"] = pd.to_numeric(df["Standard Value"], errors="coerce")
    df = df[df["Standard Value"].notna()]

    df["_factor"] = df["Standard Units"].apply(_unit_factor)
    df = df[df["_factor"].notna()]
    df["Standard Value (nM)"] = df["Standard Value"] * df["_factor"]

    df = df[df["Smiles"].apply(_valid_smiles)]

    before = len(df)
    df = df.drop_duplicates(subset=["Molecule ChEMBL ID"], keep="first")
    removed = before - len(df)
    if removed:
        logger.info("Removed %d duplicate Molecule IDs", removed)

    df_out = df[[
        "Molecule ChEMBL ID",
        "Smiles",
        "Standard Relation",
        "Standard Value (nM)",
    ]]

    df_out.to_csv(dst, index=False, encoding="utf-8")
    logger.info("Saved %d clean rows → %s", len(df_out), dst)


if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv:
        input_path = DEFAULT_INPUT
        output_path = DEFAULT_OUTPUT
    elif len(argv) == 1:
        input_path = Path(argv[0])
        output_path = DEFAULT_OUTPUT
    else:
        input_path = Path(argv[0])
        output_path = Path(argv[1])

    clean_file(input_path, output_path)
