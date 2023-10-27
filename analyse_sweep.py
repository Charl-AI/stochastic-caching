import argparse
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("--csv-dir", type=str, default="outputs/")
parser.add_argument("--fig-save-dir", type=str, default="assets/")

DATASET_SIZE_GIB = 9.2


def main(args):
    files = glob.glob(os.path.join(args.csv_dir, "*.csv"))

    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)

    for col in df.columns:
        if col not in ["cache_limit_gib", "epoch_0_time", "epoch_1_time"]:
            df = df.drop(col, axis=1)

    df = df.melt(
        id_vars=["cache_limit_gib"],
        value_vars=["epoch_0_time", "epoch_1_time"],
        var_name="epoch",
        value_name="time",
    )

    df["cache_limit_gib"] = df["cache_limit_gib"] / DATASET_SIZE_GIB * 100
    df["cache_limit_gib"] = df["cache_limit_gib"].clip(upper=100)

    df = df.rename(columns={"cache_limit_gib": "Cache Limit (%)"})
    df = df.rename(columns={"time": "Time (s)"})
    df = df.rename(columns={"epoch": "Epoch"})
    df["Epoch"] = df["Epoch"].replace({"epoch_0_time": "0", "epoch_1_time": "1"})

    sns.lmplot(
        data=df,
        x="Cache Limit (%)",
        y="Time (s)",
        hue="Epoch",
        x_jitter=2.0,
    )
    plt.ylim(bottom=0)
    plt.savefig(os.path.join(args.fig_save_dir, "sweep.png"))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
