import pandas as pd
import re

def parse_age_rexgradient(age_str: str) -> float:
    """
    Convert age strings to years:
    - '007Y' -> 7
    - '006M' -> 0.5
    - '010D' -> ~0.027
    Returns float years.
    """
    if pd.isna(age_str):
        return -1

    age_str = str(age_str).strip().upper()

    match = re.match(r"(\d+)([YMD])", age_str)
    if not match:
        return -1

    value = int(match.group(1))
    unit = match.group(2)

    if unit == "Y":
        return float(value)
    elif unit == "M":
        return round(value / 12.0, 2)
    elif unit == "D":
        return round(value / 365.0, 2)
    else:
        return -1
    
def generate_age_labels_rexgradient(metadata_path: str, split_type: str):
    df = pd.read_csv(metadata_path)

    # Convert age to years
    df['AgeYears'] = df['PatientAge'].apply(parse_age_rexgradient)

    df['FileName'] = df['StudyInstanceUid'] + '.png'

    filtered_df = df[df['AgeYears'] > 0.0]

    print(f"{df.shape[0] - filtered_df.shape[0]} NaN age values removed")

    output_data = filtered_df[['FileName', 'AgeYears']]
    output_data.to_csv(
        f'age_labels_{split_type}.txt',
        sep=' ',
        index=False,
        header=False
    )

    print(f"Mean age: {filtered_df['AgeYears'].mean():.3f}")
    print(f"Std age: {filtered_df['AgeYears'].std():.3f}")
    print(f"File 'age_labels_{split_type}.txt' created.")

def generate_age_labels_cxr_14(metadata_path: str, data_splits_path: str, split_type: str):
    df = pd.read_csv(metadata_path)

    valid_filenames = []
    with open(data_splits_path, 'r') as f:
        valid_filenames = set(line.strip().split(" ")[0] for line in f)

    filtered_df = df[df['Image Index'].isin(valid_filenames)].copy()
    output_data = filtered_df[['Image Index', 'Patient Age']]

    output_data.to_csv(f'age_labels_{split_type}.txt', sep=' ', index=False, header=False)
    print(f"Mean: {filtered_df['Patient Age'].mean()}, Std Dev: {filtered_df['Patient Age'].std()}")
    print(f"File 'age_labels_{split_type}.txt' has been created.")

generate_age_labels_cxr_14('Chest_X_Rays_Metadata.csv', '../classification/datasets/data_splits/cxr14/test_official.txt', 'test')
generate_age_labels_cxr_14('Chest_X_Rays_Metadata.csv', '../classification/datasets/data_splits/cxr14/train_official.txt', 'train')
generate_age_labels_cxr_14('Chest_X_Rays_Metadata.csv', '../classification/datasets/data_splits/cxr14/val_official.txt', 'val')
generate_age_labels_rexgradient('train_metadata.csv', 'train')
generate_age_labels_rexgradient('test_metadata.csv', 'test')
generate_age_labels_rexgradient('valid_metadata.csv', 'valid')