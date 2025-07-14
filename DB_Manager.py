# from DB_Manager import Manager
# и используете по названиям функций
# valid_smiles - принимает csv распакованый через pandas, возвращает csv с только валидными smiles
# calc_all_desc - принимает csv распакованый через pandas, возвращает csv с добавленными дескрипторами
# calc_fps - принимает csv распакованый через pandas, возвращает csv с добавленными фингерпринтами
# merge(in, out) - принимает пть к папке, где лежат csv, всех их объединяет и потом сохраняет в файл out в той же папке
# calc_desc_need - генерирует только нужные 127 дескрипторов, используется в самом конце
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

    @staticmethod
    def calc_desc_need(df):
        import pandas as pd
        import numpy as np
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        from rdkit.ML.Descriptors import MoleculeDescriptors
        from tqdm import tqdm

        R = 0.70

        desc_names = [
            "MaxAbsEStateIndex", "MinAbsEStateIndex", "MinEStateIndex", "qed", "SPS",
            "MaxPartialCharge", "MinPartialCharge", "FpDensityMorgan1", "BCUT2D_MWHI",
            "BCUT2D_LOGPLOW", "BCUT2D_MRHI", "BCUT2D_MRLOW", "AvgIpc", "BalabanJ", "Ipc",
            "PEOE_VSA1", "PEOE_VSA10", "PEOE_VSA11", "PEOE_VSA12", "PEOE_VSA13",
            "PEOE_VSA2", "PEOE_VSA3", "PEOE_VSA4", "PEOE_VSA5", "PEOE_VSA6",
            "PEOE_VSA8", "PEOE_VSA9", "SMR_VSA10", "SMR_VSA2", "SMR_VSA3", "SMR_VSA4",
            "SMR_VSA6", "SMR_VSA8", "SlogP_VSA1", "SlogP_VSA10", "SlogP_VSA11",
            "SlogP_VSA12", "SlogP_VSA2", "SlogP_VSA3", "SlogP_VSA4", "SlogP_VSA5",
            "SlogP_VSA6", "SlogP_VSA7", "SlogP_VSA8", "SlogP_VSA9", "VSA_EState1",
            "VSA_EState10", "VSA_EState2", "VSA_EState3", "VSA_EState4", "VSA_EState5",
            "VSA_EState6", "VSA_EState7", "VSA_EState8", "VSA_EState9", "fr_Al_COO",
            "fr_Al_OH", "fr_Al_OH_noTert", "fr_ArN", "fr_Ar_COO", "fr_Ar_N", "fr_Ar_NH",
            "fr_Ar_OH", "fr_Ar_ring", "fr_ArNHR", "fr_Ar_OHNoCOO", "fr_COO", "fr_COO2",
            "fr_C_O", "fr_C_O_noCOO", "fr_C_S", "fr_F", "fr_Imine", "fr_NH0", "fr_NH1",
            "fr_NH2", "fr_N_O", "fr_Ndealkylation1", "fr_Ndealkylation2", "fr_Nhpyrrole",
            "fr_Npyridinium", "fr_OCN", "fr_OH", "fr_OH_noO", "fr_SH", "fr_SMe",
            "fr_aldehyde", "fr_alkyl_carbamate", "fr_alkyl_halide", "fr_allylic_oxid",
            "fr_amide", "fr_amidine", "fr_aniline", "fr_aryl_methyl", "fr_azide",
            "fr_azo", "fr_barbitur", "fr_benzene", "fr_benzodiazepine", "fr_bicyclic",
            "fr_diazo", "fr_dihydropyridine", "fr_epoxide", "fr_ester", "fr_ether",
            "fr_furan", "fr_guanido", "fr_halogen", "fr_hdrzine", "fr_hdrzone", "fr_imide",
            "fr_imidazole", "fr_isocyan", "fr_isothiocyan", "fr_ketone",
            "fr_ketone_Topliss", "fr_lactam", "fr_lactone", "fr_methoxy", "fr_morpholine",
            "fr_nitrile", "fr_nitro", "fr_nitro_arom", "fr_nitroso", "fr_oxazole",
            "fr_oxime", "fr_para_hydroxylation"]
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
