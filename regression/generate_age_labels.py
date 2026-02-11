import pandas as pd

def generate_age_labels(metadata_path: str, data_splits_path: str, split_type: str):
    df = pd.read_csv(metadata_path)

    valid_filenames = []
    with open(data_splits_path, 'r') as f:
        valid_filenames = set(line.strip().split(" ")[0] for line in f)

    filtered_df = df[df['Image Index'].isin(valid_filenames)].copy()
    output_data = filtered_df[['Image Index', 'Patient Age']]

    output_data.to_csv(f'age_labels_{split_type}.txt', sep=' ', index=False, header=False)
    print(f"Mean: {filtered_df['Patient Age'].mean()}, Std Dev: {filtered_df['Patient Age'].std()}")
    print(f"File 'age_labels_{split_type}.txt' has been created.")

generate_age_labels('Chest_X_Rays_Metadata.csv', '../classification/datasets/data_splits/cxr14/test_official.txt', 'test')
generate_age_labels('Chest_X_Rays_Metadata.csv', '../classification/datasets/data_splits/cxr14/train_official.txt', 'train')
generate_age_labels('Chest_X_Rays_Metadata.csv', '../classification/datasets/data_splits/cxr14/val_official.txt', 'val')