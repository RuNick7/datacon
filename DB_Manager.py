# from DB_Manager import Manager
# и используете по названиям функций
# valid_smiles - принимает csv распакованый через pandas, возвращает csv с только валидными smiles
# calc_all_desc - принимает csv распакованый через pandas, возвращает csv с добавленными дескрипторами
# calc_fps - принимает csv распакованый через pandas, возвращает csv с добавленными фингерпринтами
# merge(in, out) - принимает пть к папке, где лежат csv, всех их объединяет и потом сохраняет в файл out в той же папке
class Manager:
    @staticmethod
    def valid_smiles(df):
        from rdkit import Chem
        def is_valid_smiles(smiles):
            try:
                return Chem.MolFromSmiles(smiles) is not None
            except:
                return False
        # Фильтрация
        return df[df['SMILES'].apply(is_valid_smiles)].copy()
    @staticmethod
    def calc_desc(df):
        import pandas as pd
        import numpy as np
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        from rdkit.ML.Descriptors import MoleculeDescriptors
        from tqdm import tqdm

        R = 0.70

        desc_names = [name for name, _ in Descriptors._descList]
        calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)

        def calc_all_desc(smiles):
            mol = Chem.MolFromSmiles(smiles)
            try:
                return list(calc.CalcDescriptors(mol))
            except Exception:
                return [np.nan] * len(desc_names)

        tqdm.pandas()
        df_desc = df["Smiles"].progress_apply(calc_all_desc)
        df_desc = pd.DataFrame(df_desc.tolist(), columns=desc_names)
        df = pd.concat([df.reset_index(drop=True), df_desc], axis=1)

        df = df.dropna(axis=0, how="any")

        cols_constant = [col for col in df.columns if df[col].nunique(dropna=False) == 1]
        df = df.drop(columns=cols_constant)

        num_cols = df.select_dtypes(include=[np.number]).columns

        corr = df[num_cols].corr().abs()  # |r|
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        to_drop = [col for col in upper.columns if any(upper[col] > R)]
        df_reduced = df.drop(columns=to_drop)
        return df_reduced
    @staticmethod
    def calc_fps(df):
        import pandas as pd
        import numpy as np
        from rdkit import Chem
        from tqdm import tqdm
        from rdkit.Chem import MACCSkeys
        from rdkit.Chem import rdFingerprintGenerator as rfpg
        def calc(smiles):
            mol = Chem.MolFromSmiles(smiles)

            maccs_fp = MACCSkeys.GenMACCSKeys(mol)
            ecfp_fp = gen_ecfp.GetFingerprint(mol)

            return pd.Series([maccs_fp.ToBitString(), ecfp_fp.ToBitString()])
        tqdm.pandas()
        gen_ecfp = rfpg.GetMorganGenerator(radius=2, fpSize=2048)
        df[["Maccs", "ECFP"]] = df["Smiles"].progress_apply(calc)

    @staticmethod
    def merge(csv_dir, output_file):
        import pandas as pd
        import glob
        from pathlib import Path

        # Путь к папке с CSV
        csv_dir = Path(csv_dir)
        # Шаблон поиска
        pattern = "*.csv"
        # Итоговый файл
        output_file = csv_dir / output_file

        # Собираем все пути
        file_paths = sorted(csv_dir.glob(pattern))

        # Считываем и объединяем
        dfs = [pd.read_csv(fp) for fp in file_paths]
        merged = pd.concat(dfs, ignore_index=True)

        # Сохраняем
        merged.to_csv(output_file, index=False)

        print(f"Объединено {len(file_paths)} файлов → {output_file}")