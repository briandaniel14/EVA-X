import pandas as pd

def generate_sex_labels_cxr_14(metadata_path: str, data_splits_path: str, split_type: str):
    df = pd.read_csv(metadata_path)

    valid_filenames = []
    with open(data_splits_path, 'r') as f:
        valid_filenames = set(line.strip().split(" ")[0] for line in f)

    df['Male'] = (df['Patient Gender'] == 'M').astype(int)
    df['Female'] = (df['Patient Gender'] == 'F').astype(int)
    filtered_df = df[df['Image Index'].isin(valid_filenames)].copy()
    output_data = filtered_df[['Image Index', 'Male', 'Female']]

    output_data.to_csv(f'sex_labels_{split_type}.txt', sep=' ', index=False, header=False)

    print(f"File 'sex_labels_{split_type}.txt' has been created.")

def generate_sex_labels_rexgradient(metadata_path: str, split_type: str):
    df = pd.read_csv(metadata_path)

    # One-hot encoding
    df['Male'] = (df['PatientSex'] == 'M').astype(int)
    df['Female'] = (df['PatientSex'] == 'F').astype(int)
    df['FileName'] = df['StudyInstanceUid'] + '.png'

    filtered_df = df.query('Female == 0 and Male == 0')
    print(f"NOTE: There are {filtered_df.shape[0]} instances where Sex is neither male nor female. Keeping these in for now")

    output_data = df[['FileName', 'Male', 'Female']]
    output_data.to_csv(
        f'sex_labels_{split_type}.txt',
        sep=' ',
        index=False,
        header=False
    )

    print(f"File 'sex_labels_{split_type}.txt' created.")
    print(f"Samples: {len(output_data)}")

generate_sex_labels_cxr_14('Chest_X_Rays_Metadata.csv', './classification/datasets/data_splits/cxr14/test_official.txt', 'test')
generate_sex_labels_cxr_14('Chest_X_Rays_Metadata.csv', './classification/datasets/data_splits/cxr14/train_official.txt', 'train')
generate_sex_labels_cxr_14('Chest_X_Rays_Metadata.csv', './classification/datasets/data_splits/cxr14/val_official.txt', 'val')
generate_sex_labels_rexgradient('train_metadata.csv', 'train')
generate_sex_labels_rexgradient('test_metadata.csv', 'test')
generate_sex_labels_rexgradient('valid_metadata.csv', 'valid')