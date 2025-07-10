import pandas as pd
from rdkit import Chem

data = pd.read_csv("tasks/input_file.csv", sep=';', quotechar='"')

data = data[['Smiles', 'Standard Type', 'Standard Relation', 'Standard Value', 'Standard Units']]
data = data[data["Standard Type"] == "IC50"]

def do_all(row):
    new_row = format_units(row)
    new_row = is_valid_smiles(new_row)
    return new_row
def format_units(row):
    if row['Standard Units'] == "nM":
        return row
    elif row['Standard Units'] == "M":
        row['Standard Units'] = "nM"
        row['Standard Value'] = float(row['Standard Value']) * 10 ** (-9)
        return row
    return

def is_valid_smiles(row):
    try:
        mol = Chem.MolFromSmiles(row['Smiles'].strip())
        if mol != None:
            return row
        return
    except Exception:
        return


new_data = data.apply(format_units, axis=1)
new_data = new_data.apply(is_valid_smiles, axis=1)
new_data = new_data.dropna(subset='Smiles')
new_data = new_data.drop_duplicates(subset='Smiles')
new_data = new_data[new_data['Standard Relation'] == "'='"]
new_data = new_data[['Smiles', 'Standard Value']]

print(len(new_data))

new_data.to_csv("dataset_valid.csv", index=False)