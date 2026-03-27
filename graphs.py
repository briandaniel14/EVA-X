import json
import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


DENORM_FACTOR = 16.60267981756069


def load_metrics(path):
    """Load JSON-lines metrics file into a DataFrame."""
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


def denormalise(df):
    """Apply denormalisation to relevant metrics."""
    df = df.copy()
    df["train_loss_denorm"] = df["train_loss"] * DENORM_FACTOR
    df["test_loss_denorm"] = df["test_loss"] * DENORM_FACTOR
    df["test_mae_denorm"] = df["test_mae"] * DENORM_FACTOR
    return df


def plot_metrics(df, save_path=None):
    """Plot denormalised losses and MAE."""
    sns.set_theme(style="whitegrid", context="talk")

    plot_df = df[
        ["epoch", "train_loss_denorm", "test_loss_denorm", "test_mae_denorm"]
    ].melt(
        id_vars="epoch",
        var_name="metric",
        value_name="value",
    )

    # Pretty legend names
    name_map = {
        "train_loss_denorm": "Train Loss",
        "test_loss_denorm": "Validation Loss",
        "test_mae_denorm": "Validation MAE",
    }
    plot_df["metric"] = plot_df["metric"].map(name_map)

    plt.figure(figsize=(11, 6))
    sns.lineplot(
        data=plot_df,
        x="epoch",
        y="value",
        hue="metric",
        marker="o",
    )

    plt.title("Denormalised Training Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Denormalised Value")
    plt.legend(title="")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logfile", type=Path, help="Path to metrics text file")
    parser.add_argument("--out", type=Path, default=None, help="Optional output image path")
    args = parser.parse_args()

    df = load_metrics(args.logfile)
    df = df.sort_values("epoch")
    df = denormalise(df)

    plot_metrics(df, args.out)


if __name__ == "__main__":
    main()