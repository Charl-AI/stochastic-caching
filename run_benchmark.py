import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from benchmark.dataset import DummyDataset
from benchmark.trainer import train

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--data-dir", type=str, default="/data2/dummy_data")
parser.add_argument("--cache-limit-gib", type=int, default=0)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--num-workers", type=int, default=8)
parser.add_argument("--pin-memory", type=bool, default=True)
parser.add_argument("--output-dir", type=str, default="outputs/")

NUM_EPOCHS = 2


def main(args):
    # first epoch fills cache, second epoch uses cache
    torch.manual_seed(args.seed)
    dataset = DummyDataset(args.data_dir, args.cache_limit_gib)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=args.pin_memory,
    )
    times = train(loader, NUM_EPOCHS)
    assert len(times) == 2
    epoch_0_time = times[0]
    epoch_1_time = times[1]

    os.makedirs(args.output_dir, exist_ok=True)

    run_stats = {
        "seed": [args.seed],
        "data_dir": [args.data_dir],
        "cache_limit_gib": [args.cache_limit_gib],
        "batch_size": [args.batch_size],
        "num_workers": [args.num_workers],
        "pin_memory": [args.pin_memory],
        "epoch_0_time": [epoch_0_time],
        "epoch_1_time": [epoch_1_time],
    }
    print(f"Run stats: {run_stats}")
    df = pd.DataFrame.from_dict(run_stats)
    df.to_csv(
        os.path.join(args.output_dir, f"run_{args.seed}_{args.cache_limit_gib}.csv")
    )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
