import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import rdFingerprintGenerator as rfpg

gen_ecfp = rfpg.GetMorganGenerator(radius=2, fpSize=2048)

df = pd.read_csv("dataset_valid.csv", sep=',',)


def calc_fps(smiles):
    mol = Chem.MolFromSmiles(smiles)

    maccs_fp = MACCSkeys.GenMACCSKeys(mol)
    ecfp_fp  = gen_ecfp.GetFingerprint(mol)

    return pd.Series([maccs_fp.ToBitString(), ecfp_fp.ToBitString()])

df[["Maccs", "ECFP"]] = df["Smiles"].apply(calc_fps)

df.to_csv("data_with_fp.csv", index=False)
print(f"Done → data_with_fp")