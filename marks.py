import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from tqdm import tqdm
from rdkit.Chem import MACCSkeys
from rdkit.Chem import rdFingerprintGenerator as rfpg

SRC_CSV = "dataset_valid.csv"
DST_CSV = "data_desc_and_fp.csv"
SMI_COL = "Smiles"

df = pd.read_csv(SRC_CSV)
desc_names = [name for name, _ in Descriptors._descList]
calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)


def calc_all_desc(smiles):
    mol = Chem.MolFromSmiles(smiles)
    try:
        return list(calc.CalcDescriptors(mol))
    except Exception:
        return [np.nan] * len(desc_names)


tqdm.pandas()
df_desc = df[SMI_COL].progress_apply(calc_all_desc)
df_desc = pd.DataFrame(df_desc.tolist(), columns=desc_names)
df_final = pd.concat([df.reset_index(drop=True), df_desc], axis=1)

def calc_fps(smiles):
    mol = Chem.MolFromSmiles(smiles)

    maccs_fp = MACCSkeys.GenMACCSKeys(mol)
    ecfp_fp  = gen_ecfp.GetFingerprint(mol)

    return pd.Series([maccs_fp.ToBitString(), ecfp_fp.ToBitString()])


gen_ecfp = rfpg.GetMorganGenerator(radius=2, fpSize=2048)
df_final[["Maccs", "ECFP"]] = df_final["Smiles"].progress_apply(calc_fps)

df_final.to_csv(DST_CSV, index=False)
print(f"Done → {DST_CSV}  ({len(desc_names)} descriptors)")
print(len(df_final.columns))
#ale